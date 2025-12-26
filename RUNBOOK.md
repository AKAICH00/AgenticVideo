# AgenticVideo - Solo Creator Runbook

> **Mission**: Generate enough viral content to fund your Anthropic Max subscription and beyond.

---

## Quick Status Dashboard

| Component | Status | Notes |
|-----------|--------|-------|
| **Core Orchestrator** | Working | Full pipeline with retry logic |
| **Script Generation** | Working | Gemini 2.0 Flash + viral hooks |
| **Storyboard Generation** | Working | Auto scene breakdown |
| **Video Generation** | Working | Via Kie.ai (Veo 3, Sora 2, Wan) |
| **Auto-Download** | Working | Saves videos before URLs expire |
| **Progress Streaming** | Working | Real-time SSE updates |
| **Motion Extraction** | Not Started | Pose, camera tracking TODO |
| **Video Composition** | Partial | Remotion ready, needs integration |
| **YouTube Publishing** | Partial | OAuth ready, upload needs work |
| **Trend Intelligence** | Working | YouTube trend detection |

---

## Table of Contents

1. [What's Built & Working](#1-whats-built--working)
2. [What's NOT Working Yet](#2-whats-not-working-yet)
3. [Quick Start Guide](#3-quick-start-guide)
4. [Cost Analysis & ROI](#4-cost-analysis--roi)
5. [Available Models](#5-available-models)
6. [CLI Commands](#6-cli-commands)
7. [API Endpoints](#7-api-endpoints)
8. [Configuration Reference](#8-configuration-reference)
9. [Development Roadmap](#9-development-roadmap)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What's Built & Working

### Core Pipeline (Ready to Use)

```
User Input → Planner → Script → Storyboard → Visual Gen → Download
                                                          ↓
                                    Output: MP4 videos in /output/{campaign_id}/
```

**Working Features**:
- Script generation with viral hooks (Gemini 2.0 Flash)
- Automatic storyboard breakdown (6-12 scenes)
- Video generation via Kie.ai aggregator
- Multiple model support (Veo 3.1, Sora 2, Wan, etc.)
- Automatic video download before temp URLs expire
- Real-time progress streaming via SSE
- Circuit breaker protection (prevents API overload)
- Negative prompts to avoid text in videos
- Quality retry loops (max 3 attempts)

### Video Generation Models Available

| Model | Quality | Cost/5s | Best For |
|-------|---------|---------|----------|
| `veo-3.1` | Best | ~$1.25 | Hero content, with audio |
| `veo-3.1-fast` | Great | ~$0.25 | **Default** - fast iteration |
| `runway-aleph` | Great | ~$0.75 | Advanced scene reasoning |
| `sora-2` | Great | ~$1.00 | Realistic motion |
| `wan-2.1` | Good | ~$0.10 | Bulk content, cheap |
| `kling-2.5` | Good | ~$0.50 | ⚠️ Chinese text issues |

### Intelligence Layer

- YouTube trend detection
- Niche-specific analysis (tech, gaming, finance, etc.)
- Content brief generation
- Performance tracking (partial)

---

## 2. What's NOT Working Yet

### Critical Gaps

| Feature | Status | Impact | Effort |
|---------|--------|--------|--------|
| **Motion Transfer** | Not Started | Can't copy TikTok movements | High |
| **Video Composition** | Partial | Raw scenes, no final assembly | Medium |
| **YouTube Upload** | Partial | Manual upload required | Medium |
| **Long→Short Repurposing** | Not Started | No auto-clips from long-form | High |
| **Reference Images** | Not Started | No image-to-video | Low |
| **Audio Generation** | Ready | ElevenLabs integrated, not wired | Low |
| **Lip Sync** | Ready | SyncLabs integrated, not wired | Low |

### What This Means for You

**Today you can**:
- Generate individual video scenes automatically
- Get AI-written viral scripts
- Use multiple premium video models
- Monitor progress in real-time

**Today you cannot**:
- Copy dance moves from TikTok references
- Auto-publish to YouTube/TikTok
- Get a single composed final video
- Generate clips from long-form content

---

## 3. Quick Start Guide

### Prerequisites

```bash
# Required API Keys (minimum)
export GOOGLE_API_KEY="your-gemini-key"      # For script generation
export KIE_API_KEY="your-kie-api-key"        # For video generation

# Optional but recommended
export DATABASE_URL="postgresql://..."        # For persistence
export ELEVENLABS_API_KEY="your-key"         # For voiceover
```

### Installation

```bash
cd /root/projects/AgenticVideo
pip install -r requirements.txt
```

### Generate Your First Video

```bash
# Start the server
python main.py server --port 8765

# In another terminal - generate a video
curl -X POST http://localhost:8765/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "3 morning meditation tips for beginners",
    "niche": "wellness",
    "quality_tier": "premium"
  }'

# Watch progress
curl -N http://localhost:8765/stream/{campaign_id}
```

### Output Location

Videos are saved to:
```
output/{campaign_id}/
├── scene_1.mp4
├── scene_2.mp4
├── scene_3.mp4
└── ...
```

### Copy to Windows (WSL)

```bash
cp -r output/{campaign_id}/* /mnt/c/Users/YourName/Desktop/Videos/
```

---

## 4. Cost Analysis & ROI

### Per-Video Costs (10 scenes × 5 seconds each)

| Quality Tier | Model | Cost | Time |
|--------------|-------|------|------|
| **Premium** | Veo 3.1 Fast | ~$2.50 | 3-5 min |
| **Premium** | Veo 3.1 | ~$12.50 | 5-8 min |
| **Bulk** | Wan 2.1 | ~$1.00 | 5-10 min |

### Anthropic Max Subscription ROI

**Anthropic Max**: ~$100/month

**To Break Even (with YouTube monetization)**:
- Need ~10,000 views/day at $2 CPM = $20/day = $600/month
- Or 1-2 viral videos per month

**Video Production Capacity**:
- At $2.50/video (Veo 3.1 Fast): 40 videos/month for $100
- At $1/video (Wan 2.1): 100 videos/month for $100

### Cost Optimization Tips

1. **Use bulk tier for iteration**: Test concepts with Wan ($0.10/scene)
2. **Premium for finals only**: Use Veo 3.1 Fast for hero content
3. **Shorter scenes**: 5s scenes cost less than 10s
4. **Batch similar content**: Same niche = reusable elements

---

## 5. Available Models

### Video Models (via Kie.ai)

```python
from services.video_generation import VideoModel

# Premium (high quality, higher cost)
VideoModel.VEO_3_1         # Google Veo 3.1 - best, with audio
VideoModel.VEO_3_1_FAST    # Google Veo 3.1 Fast - DEFAULT
VideoModel.RUNWAY_ALEPH    # Runway - scene reasoning
VideoModel.SORA_2          # OpenAI Sora 2 - realistic motion

# Bulk (good quality, lower cost)
VideoModel.WAN_2_1         # Wan 2.1 - best value

# Avoid unless needed
VideoModel.KLING_2_5       # ⚠️ Renders Chinese text
```

### Model Selection in Code

```python
# In services/orchestrator/nodes.py (VisualNode)
model = (
    VideoModel.VEO_3_1_FAST  # Premium tier
    if state.quality_tier == "premium"
    else VideoModel.WAN_2_1   # Bulk tier
)
```

---

## 6. CLI Commands

### Server Management

```bash
# Start SSE server
python main.py server --host 0.0.0.0 --port 8765

# Check health
python main.py status --server http://localhost:8765
```

### Video Generation

```bash
# Basic generation
python main.py generate --topic "Your topic here"

# With quality tier
python main.py generate \
  --topic "5 AI tools for productivity" \
  --quality premium

# With reference video (motion transfer - NOT YET WORKING)
python main.py generate \
  --topic "Dance tutorial" \
  --reference "https://tiktok.com/..."
```

### Monitoring

```bash
# Monitor specific campaign
python main.py monitor {campaign_id} --server http://localhost:8765

# Watch all campaigns
curl http://localhost:8765/status
```

### Intelligence Services

```bash
# Start trend monitoring
python main.py intelligence \
  --niches tech gaming wellness \
  --mode passive
```

---

## 7. API Endpoints

### SSE Server (Default: http://localhost:8765)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Usage info |
| `/health` | GET | Health check |
| `/generate` | POST | Start video generation |
| `/stream/{id}` | GET | SSE stream for campaign |
| `/status` | GET | All campaign statuses |
| `/status/{id}` | GET | Single campaign status |

### Generate Request

```json
POST /generate
{
  "topic": "3 tips for better sleep",
  "niche": "wellness",
  "quality_tier": "premium",
  "style_reference": "calm, minimalist",
  "target_duration_seconds": 60
}
```

### Response

```json
{
  "campaign_id": "abc-123",
  "status": "started",
  "stream_url": "/stream/abc-123"
}
```

---

## 8. Configuration Reference

### Required Environment Variables

```bash
# AI/LLM (Required)
GOOGLE_API_KEY=           # Gemini 2.0 Flash for scripts

# Video Generation (At least one required)
KIE_API_KEY=              # Kie.ai aggregator (recommended)
FAL_API_KEY=              # Fal.ai alternative

# Database (Required for persistence)
DATABASE_URL=postgresql://user:pass@host/db
```

### Optional Environment Variables

```bash
# Audio
ELEVENLABS_API_KEY=       # Text-to-speech

# Lip Sync
SYNC_LABS_API_KEY=        # Lip sync

# Storage
R2_ACCOUNT_ID=            # Cloudflare R2
R2_ACCESS_KEY=
R2_SECRET_KEY=
R2_BUCKET=agentic-video

# YouTube Publishing
YOUTUBE_CLIENT_ID=
YOUTUBE_CLIENT_SECRET=
YOUTUBE_REFRESH_TOKEN=

# Observability
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=http://localhost:3000

# Migration Control
PROCESSOR_MODE=old        # old|new|split

# GPU (for self-hosted Wan 2.1)
GPU_ENABLED=false
GPU_DEVICE=cuda:0
```

---

## 9. Development Roadmap

### Phase 1: Core Pipeline (COMPLETE)
- [x] Script generation with Gemini
- [x] Storyboard auto-generation
- [x] Video generation via Kie.ai
- [x] Auto-download before URL expiry
- [x] Progress streaming (SSE)
- [x] Circuit breaker protection
- [x] Negative prompts (no text in videos)
- [x] Model selection (Veo 3, Sora 2, Wan)

### Phase 2: Composition (IN PROGRESS)
- [x] Remotion server setup
- [ ] Scene assembly into final video
- [ ] Transition effects
- [ ] Audio sync
- [ ] Text overlays (via Remotion, not AI)

### Phase 3: Publishing (PARTIAL)
- [x] YouTube OAuth integration
- [ ] Automatic upload
- [ ] Metadata optimization
- [ ] Thumbnail generation
- [ ] Multi-platform (TikTok, Instagram)

### Phase 4: Motion Transfer (NOT STARTED)
- [ ] TikTok video download
- [ ] Pose extraction (DWPose/OpenPose)
- [ ] Camera motion tracking (CoTracker)
- [ ] Motion application to new content
- [ ] Beat synchronization

### Phase 5: Intelligence (PARTIAL)
- [x] YouTube trend detection
- [x] Content brief generation
- [ ] Performance analytics
- [ ] A/B testing framework
- [ ] Auto-optimization loop

---

## 10. Troubleshooting

### Common Issues

#### "No video API aggregator configured"
```bash
# Solution: Set your API key
export KIE_API_KEY="your-key-here"
```

#### "403 API key leaked"
Your API key was exposed in a public repo. Generate a new one.

#### Videos have Chinese text
The system was using Kling. Fixed by switching to Veo/Wan defaults.
```bash
# Verify fix is applied
python -c "from core.config import get_config; print(get_config().models.default_premium)"
# Should output: veo-3.1-fast
```

#### Scene generation timeout
Kie API can take 2-5 minutes per scene. The circuit breaker has a 300s timeout.
- Check your internet connection
- Try again (transient failures happen)
- Use a faster model (`veo-3.1-fast` instead of `veo-3.1`)

#### Videos not downloading
Check the output directory exists and has write permissions:
```bash
ls -la output/
mkdir -p output
chmod 755 output
```

### Debug Mode

```bash
# Run with verbose logging
python main.py server --port 8765 2>&1 | tee server.log

# Check logs
grep ERROR server.log
grep -i "scene" server.log
```

### Health Check

```bash
# Quick health check
curl http://localhost:8765/health

# Expected response
{
  "status": "healthy",
  "connected_clients": 0,
  "active_campaigns": 0
}
```

---

## File Structure

```
AgenticVideo/
├── main.py                    # CLI entry point
├── core/
│   ├── config.py              # Configuration management
│   ├── circuit_breaker.py     # API resilience
│   └── feature_flags.py       # Migration control
├── services/
│   ├── orchestrator/          # NEW pipeline (V2)
│   │   ├── graph.py           # Workflow orchestrator
│   │   ├── nodes.py           # Pipeline nodes
│   │   └── state.py           # State management
│   ├── video_generation/      # Video API clients
│   │   └── client.py          # Unified video client
│   ├── streaming/             # Progress streaming
│   │   └── sse_server.py      # SSE server
│   ├── rendering/             # Video composition
│   │   └── remotion_client.py # Remotion integration
│   ├── publisher/             # Publishing
│   │   └── youtube_client.py  # YouTube API
│   └── motion/                # Motion extraction (TODO)
├── agents/                    # OLD polling agents
├── output/                    # Generated videos
└── RUNBOOK.md                 # This file
```

---

## Support & Contributing

**Issues**: Check logs first, then file in this repo

**The Goal**: Build enough automation to make Anthropic Max pay for itself through content monetization.

**Remember**: Every improvement to this system is an investment in your content empire.

---

*Last updated: December 2025*
*System version: AgenticVideo v2.0*
