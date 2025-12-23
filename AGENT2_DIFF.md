# AGENT 2 DIFF: OLD vs NEW ARCHITECTURE COMPARISON

## Overview

This document provides a detailed file-by-file and function-by-function comparison between:
- **OLD Architecture** (deployed): Polling daemons in `agents/`
- **NEW Architecture** (built, not deployed): Unified orchestrator in `services/orchestrator/`

---

## Architecture Summary

| Aspect | OLD (Deployed) | NEW (Built) |
|--------|----------------|-------------|
| **Pattern** | 3 polling daemons | 1 state machine |
| **Communication** | Database status column | In-memory state |
| **Visibility** | None (check DB) | Real-time SSE |
| **Quality loops** | None | Cyclic retries (max 3) |
| **Motion support** | None | Full pose/camera extraction |
| **Video APIs** | Kie only | Kie + Fal + Runway + Wan 2.1 |
| **K8s deployed** | Yes | No |

---

## File-by-File Differences

### 1. SCRIPT GENERATION

| Aspect | OLD (`agents/director.py`) | NEW (`services/orchestrator/nodes.py`) |
|--------|---------------------------|----------------------------------------|
| **Entry Point** | `run_daemon()` - infinite loop | `ScriptNode.execute(state)` - state machine |
| **Trigger** | Polls DB every 30s for `status='new'` | Called by VideoGraph when `phase=SCRIPT` |
| **LLM Provider** | Gemini via `shared/llm.py` | Gemini 2.0 Flash via `generate_script()` |
| **Output** | Inserts into `script_drafts` table | Updates `VideoState.script` in memory |
| **Visibility** | None (silent until DB update) | SSE event: `script_generated` |
| **Retry Logic** | Catches exception, updates status='failed' | Built-in `max_retries=3` with quality loop |
| **Style Variations** | 2 drafts with hardcoded style hints | Single script with dynamic style from planner |

**Code Comparison:**

```python
# OLD: agents/director.py:118-136
for i in range(num_drafts):
    style_variation = None
    if i == 0:
        style_variation = "Focus on emotional storytelling..."
    elif i == 1:
        style_variation = "Focus on surprising facts..."
    draft = generate_script(topic=campaign.topic, ...)
    drafts.append(draft)

# NEW: services/orchestrator/nodes.py (ScriptNode)
state.script = generate_script(
    topic=state.topic,
    niche=state.niche,
    style_notes=state.creative_direction,  # From PlannerNode
    format=state.format,
)
state.phase = GenerationPhase.STORYBOARD
return state
```

---

### 2. VISUAL GENERATION

| Aspect | OLD (`agents/visualist.py`) | NEW (`services/orchestrator/nodes.py`) |
|--------|----------------------------|----------------------------------------|
| **Entry Point** | `run_daemon()` - infinite loop | `VisualNode.execute(state)` - state machine |
| **Trigger** | Polls for `status='script_approved'` | Called when `phase=VISUAL` |
| **API** | Kie API only via `shared/kie_client.py` | Unified client (Kie/Fal/Runway) via `video_generation/` |
| **Priority Tiers** | Single `GENERATION_PRIORITY` env var | Per-campaign `quality_tier: premium|bulk` |
| **Motion Support** | None | Uses motion blueprint from MotionNode |
| **Output** | Inserts into `visual_scenes` table | Updates `VideoState.scenes` in memory |
| **Visibility** | DB status update only | SSE events per scene: `scene_1_generating`, `scene_1_complete` |

**Code Comparison:**

```python
# OLD: agents/visualist.py:164-167
results = await generate_all_scenes(router_scenes, priority=priority)
# ^^^ Uses hardcoded priority, no motion data

# NEW: services/orchestrator/nodes.py (VisualNode)
for scene in state.storyboard.scenes:
    result = await self.video_client.generate(
        prompt=scene.visual_prompt,
        model=self._select_model(state.quality_tier),
        motion_blueprint=state.motion_blueprint,  # From MotionNode!
        duration=scene.duration_seconds,
    )
    state.scenes.append(result)
```

---

### 3. RENDERING & COMPOSITION

| Aspect | OLD (`agents/renderer.py`) | NEW (`services/orchestrator/nodes.py`) |
|--------|---------------------------|----------------------------------------|
| **Entry Point** | `run_daemon()` - infinite loop | `ComposeNode.execute(state)` |
| **Trigger** | Polls for `status='rendering'` | Called when `phase=COMPOSE` |
| **TTS** | ElevenLabs via `shared/tts_client.py` | Same, but with word-level timestamps |
| **Lip Sync** | Optional via `shared/lipsync_client.py` | Same |
| **Remotion** | Direct HTTP to `api.cckeeper.dev` | Same API, but with motion-aware props |
| **Output Formats** | Single output | YouTube 16:9 + auto-repurposed 9:16 |
| **Visibility** | Status update after complete | SSE: `tts_complete`, `lipsync_complete`, `render_progress_50%` |

**Code Comparison:**

```python
# OLD: agents/renderer.py:236-251
props = {
    "audioUrl": f"data:audio/mp3;base64,{tts_result.audio_base64}",
    "subtitles": format_subtitles_for_remotion(tts_result.subtitles),
    "scenes": [{"order": scene["order"], "url": scene["url"], "type": "video"}...],
}

# NEW: services/orchestrator/nodes.py (ComposeNode)
props = {
    "audioUrl": state.audio_url,
    "subtitles": state.subtitles,
    "scenes": state.scenes,
    "motionBlueprint": state.motion_blueprint,  # NEW: Camera/transition timing
    "outputFormats": ["youtube_16x9", "tiktok_9x16", "reels_9x16"],  # NEW: Multi-format
}
```

---

## NEW Components (Not in OLD Architecture)

### MotionNode (`services/motion/`)
Extracts motion data from reference TikTok videos:

```python
class MotionNode:
    async def execute(self, state: VideoState) -> VideoState:
        if state.reference_video_url:
            # Extract pose sequence (dance/movement)
            state.motion_blueprint.poses = await self.pose_extractor.extract(
                video_url=state.reference_video_url
            )
            # Extract camera movement (pan/zoom/shake)
            state.motion_blueprint.camera = await self.camera_tracker.extract(
                video_url=state.reference_video_url
            )
            # Extract transition timing
            state.motion_blueprint.transitions = await self.transition_detector.detect(
                video_url=state.reference_video_url
            )
        state.phase = GenerationPhase.VISUAL
        return state
```

### QualityNode (`services/orchestrator/nodes.py`)
Implements cyclic quality feedback loop (OmniAgent pattern):

```python
class QualityNode:
    async def execute(self, state: VideoState) -> VideoState:
        score = await self._evaluate_quality(state)
        if score < self.threshold and state.retry_count < state.max_retries:
            state.retry_count += 1
            state.phase = GenerationPhase.VISUAL  # Re-generate visuals
            return state
        state.phase = GenerationPhase.COMPOSE
        return state
```

### RepurposeNode (`services/orchestrator/nodes.py`)
Auto-generates short-form clips from long-form:

```python
class RepurposeNode:
    async def execute(self, state: VideoState) -> VideoState:
        if state.format == "long":
            # Detect viral moments
            moments = await self.moment_detector.find_peaks(state.final_video_url)
            # Generate 9:16 clips for each moment
            for moment in moments:
                clip = await self.auto_clipper.extract(
                    video_url=state.final_video_url,
                    start=moment.start,
                    end=moment.end,
                    aspect_ratio="9:16",
                )
                state.repurposed_clips.append(clip)
        state.phase = GenerationPhase.DONE
        return state
```

---

## State Management Comparison

| Aspect | OLD | NEW |
|--------|-----|-----|
| **State Location** | `video_campaigns.status` column in PostgreSQL | `VideoState` dataclass in memory |
| **Granularity** | 6 status values | 8 phases + sub-steps |
| **Persistence** | Every step writes to DB | Checkpoint at phase boundaries |
| **Recovery** | Poll next cycle picks up failed | Resume from last checkpoint |

### OLD Status Flow:
```
new → in_scripting → script_approved → generating_visuals → rendering → published
                                                                    ↓
                                                                 failed
```

### NEW Phase Flow:
```
PLANNING → SCRIPT → STORYBOARD → MOTION → VISUAL → QUALITY → COMPOSE → REPURPOSE → DONE
                                            ↑         │
                                            └─────────┘ (retry loop, max 3)
```

---

## API & Visibility Comparison

| Aspect | OLD | NEW |
|--------|-----|-----|
| **Progress API** | None (check DB) | SSE stream on port 8765 |
| **CLI Support** | Manual DB queries | `python main.py monitor <campaign_id>` |
| **Event Types** | N/A | 12+ event types |
| **Real-time** | No | Yes, <100ms latency |

### SSE Event Examples (NEW):
```json
{"event": "phase_change", "data": {"from": "SCRIPT", "to": "STORYBOARD", "timestamp": "..."}}
{"event": "scene_generating", "data": {"scene": 1, "model": "runway-gen4.5", "progress": 0.45}}
{"event": "quality_check", "data": {"score": 0.82, "threshold": 0.75, "pass": true}}
```

---

## Deployment Comparison

| Aspect | OLD | NEW |
|--------|-----|-----|
| **Pods** | 3 separate (director, visualist, renderer) | 1 unified (orchestrator) |
| **Scaling** | Horizontal per agent | Vertical (more memory/CPU) |
| **Inter-service** | Via PostgreSQL status column | In-memory state |
| **K8s Manifest** | `k8s/03-agents.yaml` | `k8s/06-orchestrator.yaml` (TO CREATE) |

---

## Shared Code Usage

| File | Used by OLD | Used by NEW | Notes |
|------|-------------|-------------|-------|
| `shared/db.py` | Direct DB calls | For checkpoints only | Same connection logic |
| `shared/llm.py` | `generate_script()` | Same function | No change needed |
| `shared/kie_client.py` | Direct usage | Replaced by unified client | Deprecated by NEW |
| `shared/tts_client.py` | Direct usage | Via ComposeNode | Same API |
| `shared/lipsync_client.py` | Direct usage | Via ComposeNode | Same API |

---

## Migration Path

To move from OLD to NEW without downtime:

1. **Deploy orchestrator alongside old agents** (both running)
2. **Add feature flag** to route new campaigns to orchestrator
3. **Let old agents drain** existing in-progress campaigns
4. **Remove old agents** once all campaigns complete
5. **Full cutover** to orchestrator-only

---

## Summary Table

| Feature | OLD (Deployed) | NEW (Built) | Winner |
|---------|----------------|-------------|--------|
| Simplicity | Simple polling | Complex state machine | OLD |
| Visibility | None | Full SSE streaming | NEW |
| Motion support | None | Full pose/camera | NEW |
| Quality loops | None | Cyclic retries | NEW |
| Multi-model | Kie only | Kie/Fal/Runway/Wan | NEW |
| Repurposing | None | Auto short-form | NEW |
| K8s deployed | Yes | No | OLD |
| Battle-tested | In production | Not tested | OLD |

---

## Required K8s Changes

### 1. NEW DEPLOYMENTS TO CREATE

```yaml
# k8s/06-orchestrator.yaml (NEW FILE)
- name: video-orchestrator
  command: ["python", "main.py", "server"]
  ports: [8765]  # SSE streaming
  env:
    - KIE_API_KEY, FAL_API_KEY, ANTHROPIC_API_KEY
    - DATABASE_URL, REDIS_URL
    - LANGFUSE_* (replaces LANGSMITH)

# k8s/07-motion-extractor.yaml (OPTIONAL - if GPU available)
- name: motion-extractor
  command: ["python", "-m", "services.motion"]
  resources:
    limits:
      nvidia.com/gpu: 1

# k8s/08-intelligence-bridge.yaml (NEW FILE)
- name: trend-trigger
  command: ["python", "-m", "services.intelligence_bridge.trend_trigger"]
  env:
    - ORCHESTRATOR_URL=http://video-orchestrator:8765
```

### 2. SECRETS TO UPDATE

```yaml
# ADD these new secrets to k8s/00-namespace-secrets.yaml:
- FAL_API_KEY: ""
- ANTHROPIC_API_KEY: ""
- LANGFUSE_SECRET_KEY: ""
- LANGFUSE_PUBLIC_KEY: ""
- LANGFUSE_HOST: ""
```

### 3. DEPLOYMENTS TO DEPRECATE

```yaml
# Either remove or refactor k8s/03-agents.yaml:
# - agent-director
# - agent-visualist
# - agent-renderer
# (The new orchestrator replaces their function)
```

---

## Recommendation

**Deploy NEW orchestrator in parallel, migrate gradually.**

The NEW architecture is superior for visibility and motion support, but the OLD architecture is battle-tested. Use feature flags to route traffic incrementally.
