# 🏃 ACTIVE SPRINT - AGENTIC VIDEO SYSTEM
## Sprint ID: sprint-20241220-unified
## Goal: Build viral content production pipeline with intelligent optimization
## Started: 2024-12-20

---
## ⚠️ ALL AGENTS: READ THIS FIRST

### Current State Assessment
- **Infrastructure (K8s)**: ✅ DONE - namespace, postgres, nocodb, remotion skeleton
- **Database Schema**: ✅ DONE - campaigns, scripts, scenes, lip sync, performance tables
- **Agent Code**: ✅ DONE - orchestrator with LangGraph-style workflow
- **Intelligence Layer**: ✅ DONE - trend analyzer, content planner, feedback loop
- **Pipeline Integration**: ✅ DONE - intelligence → pipeline bridge connected
- **Remotion Compositions**: ❌ NOT STARTED - empty src/
- **Distribution**: ❌ NOT STARTED - no upload automation

### Critical Connections (DO NOT BREAK)
```yaml
# Tailscale Mesh Network
PostgreSQL: 100.121.2.100:5432      # Campaign state DB
NocoDB:     100.121.2.100:8082      # Campaign management UI
ComfyUI:    100.64.0.2:8188         # Local GPU for image gen
Remotion:   api.cckeeper.dev         # Video rendering API

# Cloud Services
KIE.ai:      api.kie.ai/v1          # Multi-model AI
ElevenLabs:  api.elevenlabs.io      # Voice synthesis
SyncLabs:    api.synclabs.so        # Lip sync
R2 Storage:  <account>.r2.cloudflarestorage.com
```

---
## 📋 PARALLEL WORK ASSIGNMENTS

### STREAM 1: Intelligence Layer (HIGH PRIORITY)
**Assigned to**: Agent handling strategy/analytics
**Files to create**:
```
services/youtube_intelligence/
services/performance_tracker/
services/strategy_engine/
sql/003_add_performance_tables.sql
k8s/05-youtube-intelligence.yaml
```
**Dependencies**: None (greenfield)
**Success Criteria**:
- YouTube trending data flowing in
- Performance metrics stored in time-series DB
- Content recommendations generated

### STREAM 2: Core Agent Implementation (HIGH PRIORITY)
**Assigned to**: Agent handling ML/generation
**Files to create**:
```
agents/director/__main__.py + modules
agents/visualist/__main__.py + modules
agents/renderer/__main__.py + modules
agents/audio/__main__.py + modules
agents/shared/async_db.py (upgrade existing)
```
**Dependencies**: Database schema (✅ exists)
**Success Criteria**:
- Director generates scripts with Claude
- Visualist creates images via KIE
- Audio generates voiceover + music
- All agents poll/update campaign status

### STREAM 3: Remotion + Distribution (MEDIUM PRIORITY)
**Assigned to**: Agent handling frontend/video
**Files to create**:
```
remotion/src/Root.tsx
remotion/src/compositions/*.tsx
services/publisher/*.py
k8s/08-publisher.yaml
```
**Dependencies**: Renderer agent (for testing)
**Success Criteria**:
- Remotion renders vertical + horizontal video
- Videos upload to YouTube automatically
- Metadata generated intelligently

---
## 🔄 Communication Protocol

### How Agents Should Coordinate:
1. Before starting work, check this file for assignments
2. Update status in "Sprint Progress Log" below
3. If blocked, add to "🚨 Blockers" section
4. Commit with message: `[STREAM-N] description`

### Shared Resources (DO NOT DUPLICATE):
- `agents/shared/` - Common utilities
- `sql/` - All schema changes go here
- `k8s/` - Infrastructure configs
- `.env.example` - Document all required env vars

---
## 🧪 Test Commands

```bash
# Verify K8s namespace exists
kubectl get ns viral-video-agents

# Check PostgreSQL connectivity
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -c "SELECT 1"

# Verify NocoDB
kubectl port-forward -n viral-video-agents svc/nocodb-service 8080:80

# Test Remotion (once implemented)
kubectl port-forward -n viral-video-agents svc/remotion-service 8000:8000
curl http://localhost:8000/health
```

---
## 🚨 Known Issues / Don't Touch
- K8s secrets use placeholder values - need Doppler integration
- Database migrations in sql/ must be run in order
- Remotion needs Chrome dependencies in Docker image

---
## 📊 Sprint Progress Log

### 2024-12-20 14:30 - Sprint Created
- Reviewed existing agent work
- Identified 3 parallel work streams
- Created unified architecture plan

### [STREAM-1] 2024-12-21 - Intelligence Layer COMPLETE
- **Completed**:
  - `services/youtube_intelligence/trend_analyzer.py` - YouTube Data API v3 trend detection
  - `services/strategy_engine/content_planner.py` - Gemini-powered content briefs
  - `services/intelligence_bridge/` - Full integration layer:
    - `campaign_factory.py` - Creates VideoState from trend data
    - `trend_trigger.py` - Automated campaign triggering
    - `feedback_loop.py` - Performance → strategy feedback
  - `sql/003_add_performance_tables.sql` - Performance tracking schema
  - `k8s/05-intelligence-layer.yaml` - K8s deployments
  - Updated `nodes.py` - All nodes now use Gemini 2.0 Flash
  - `tests/test_intelligence_pipeline.py` - Integration tests

### [STREAM-2] [Timestamp] - Status
- Orchestrator pipeline exists and is connected to intelligence layer
- Nodes updated to use Gemini instead of Anthropic

### [STREAM-3] 2024-12-22 - Remotion Integration Complete ✅
- **Completed**:
  - `remotion/src/Root.tsx` - Main entry point registering compositions
  - `remotion/src/YouTubeVideo.tsx` - Horizontal 16:9 composition for YouTube
  - `remotion/src/ViralShort.tsx` - Vertical 9:16 composition (existed, verified)
  - `remotion/src/index.ts` - Remotion entry point
  - `services/rendering/remotion_client.py` - API client for Remotion server
  - `services/rendering/__init__.py` - Service exports
  - Updated `ComposeNode` in `nodes.py` to use Remotion API

- **Remotion Compositions**:
  - `ViralShort`: 1080x1920 (9:16) for TikTok/Shorts/Reels
  - `YouTubeVideo`: 1920x1080 (16:9) for YouTube with intro/outro support

- **Integration Points**:
  - ComposeNode calls `compose_video()` from rendering service
  - Falls back to first scene if Remotion unavailable
  - Supports audio sync, subtitles, avatar overlay

### [INTEGRATION] 2024-12-21 - Integration Audit Complete
- **Created**: `DIFF.md` - Comprehensive integration fix documentation
- **Issues Found**:
  - Schema mismatch: `feedback_loop.py` uses columns not in `003_add_performance_tables.sql`
  - Redis using emptyDir (needs PVC for persistence)
  - No `intelligence` command in main.py to start background services
- **Required Fixes**:
  1. `sql/005_fix_intelligence_columns.sql` - Add missing columns, rename snapshot_at→collected_at
  2. `k8s/05-intelligence-layer.yaml` - Add Redis PVC
  3. `main.py` - Add `intelligence` command for TrendTrigger + FeedbackLoop
- **See**: `DIFF.md` for exact line-by-line changes

### [INTEGRATION] 2024-12-21 - Fixes Implemented
- **Completed**:
  - `sql/005_fix_intelligence_columns.sql` - Created idempotent migration with IF EXISTS guards
  - `k8s/05-intelligence-layer.yaml` - Added Redis PVC (1Gi), removed redundant ANTHROPIC_API_KEY
  - `main.py` - Added `python main.py intelligence --niches tech --mode passive` command
- **Deployed and tested**

### [INTEGRATION] 2024-12-22 - Integration Testing Complete ✅
- **Issues Found & Fixed**:
  1. `trending_topics` table missing - syntax error in migration 003 UNIQUE constraint
     - Fix: Created table with corrected constraint (removed ::date cast)
  2. SQL INTERVAL parameter bug - `INTERVAL '$N days'` doesn't work with asyncpg
     - Fix: Changed to `make_interval(days => $N)` in trend_trigger.py and feedback_loop.py
  3. ON CONFLICT mismatch - feedback_loop.py constraint didn't match schema
     - Fix: Updated to use `ON CONFLICT (campaign_id, platform, collected_at)`
  4. Missing __init__.py exports - youtube_intelligence and strategy_engine importing non-existent modules
     - Fix: Updated to only export existing classes

- **Test Results**:
  ```
  $ python main.py intelligence --niches tech --mode passive
  ✅ Database connection: SUCCESS (PostgreSQL via port-forward)
  ✅ TrendTrigger: Started and monitoring
  ✅ PerformanceTracker: Started with feedback cycle
  ✅ FeedbackLoop: Completed cycle (0 topics - no data yet)
  ✅ Graceful shutdown: SUCCESS
  ```

- **All 15 tables verified**:
  - video_campaigns, video_performance, trending_topics
  - content_recommendations, competitor_content, ab_tests
  - generation_costs, generation_steps, script_drafts
  - visual_scenes, motion_data, reference_videos
  - short_form_clips, circuit_breaker_state, video_generation_jobs

### [STREAM-3] 2024-12-22 - Publisher Service Complete ✅
- **Completed**:
  - `services/publisher/youtube_client.py` - YouTube Data API v3 client with:
    - Resumable video uploads
    - OAuth2 token refresh
    - Thumbnail uploads
    - Playlist management
    - Metadata updates
  - `services/publisher/metadata_generator.py` - SEO optimization:
    - AI-powered title/description generation (Gemini 2.0 Flash)
    - Niche-specific tag generation
    - A/B variant generation for testing
    - Thumbnail text suggestions
  - `services/publisher/scheduler.py` - Optimal posting times:
    - Niche-specific timing patterns (tech, finance, gaming, etc.)
    - Historical performance analysis
    - Competition level scoring
    - Shorts-specific adjustments
  - `services/publisher/worker.py` - Background processing:
    - Publish queue monitoring
    - Retry logic with configurable limits
    - Scheduled video processing
  - `services/publisher/__init__.py` - Service exports
  - `k8s/08-publisher.yaml` - K8s deployment with:
    - YouTube secrets configuration
    - CronJob for scheduled publishing (every 15 min)
    - Upload cache volume
  - Updated orchestrator:
    - Added `PublishNode` to `nodes.py`
    - Added `PUBLISHING` phase to `state.py`
    - Updated `graph.py` workflow: REPURPOSE → PUBLISH → COMPLETE

- **Publisher Workflow**:
  1. `PublishNode` generates SEO metadata via AI
  2. Calculates optimal posting time based on niche patterns
  3. Sets campaign status to `ready_to_publish`
  4. Background worker picks up and uploads to YouTube
  5. Updates database with `youtube_video_id` and `published_url`

- **Configuration Required**:
  - `BLOTATO_API_KEY` - Multi-platform publishing (recommended)
  - `GOOGLE_API_KEY` - For metadata generation
  - `YOUTUBE_CLIENT_ID` - OAuth2 client ID (fallback if no Blotato)
  - `YOUTUBE_CLIENT_SECRET` - OAuth2 client secret (fallback)
  - `YOUTUBE_REFRESH_TOKEN` - OAuth2 refresh token (fallback)

### [STREAM-3] 2024-12-22 - Blotato Multi-Platform Integration ✅
- **Completed**:
  - `services/publisher/blotato_client.py` - Full Blotato API client:
    - Multi-platform support: YouTube, TikTok, LinkedIn, Pinterest, Threads, Bluesky
    - YouTube-specific: title, description, tags, privacy, AI disclosure
    - TikTok-specific: privacy level, duet/stitch settings, AI disclosure
    - Scheduled posting support
    - Rate limiting (30 posts/min, 10 media/min)
  - Updated `services/publisher/__init__.py` - Added Blotato exports
  - Updated `services/orchestrator/nodes.py` - `PublishNode` now uses Blotato:
    - Publishes to YouTube + TikTok (for shorts) automatically
    - Falls back to queue-only mode if Blotato not configured
    - Tracks publish results in state metadata
  - Updated `k8s/08-publisher.yaml`:
    - Added `blotato-secrets` Secret
    - Made YouTube secrets optional (fallback)
    - Added `BLOTATO_API_KEY` env to both Deployment and CronJob

- **Blotato API Reference**:
  - Base URL: `https://backend.blotato.com/v2`
  - Auth: `blotato-api-key` header
  - POST `/v2/posts` - Publish content
  - POST `/v2/media` - Upload media (optional - can use public URLs)
  - Docs: https://help.blotato.com/

- **Publishing Workflow (Updated)**:
  1. `PublishNode` generates SEO metadata via Gemini
  2. Calculates optimal posting time based on niche patterns
  3. If `BLOTATO_API_KEY` set:
     - Publishes directly to YouTube (all videos)
     - Publishes to TikTok (shorts <= 60s only)
     - Records publish results in state metadata
  4. If no Blotato:
     - Queues for background worker with YouTube OAuth

---
## 💡 Architecture Decisions

### Why This Structure:
1. **Intelligence-First**: Without trend data, we're guessing what to make
2. **Agent Separation**: Director/Visualist/Renderer can scale independently
3. **LangGraph over Polling**: State machine > while-true loops
4. **Remotion for Rendering**: Programmatic video > manual editing

### Technology Choices:
| Component | Choice | Reason |
|-----------|--------|--------|
| Orchestration | LangGraph | State machine with memory |
| Script Gen | Gemini 2.0 Flash | Fast, cheap, good quality |
| Visual Gen | KIE.ai | Video + image in one API |
| Voice | ElevenLabs | Best quality TTS |
| Music | Suno | Context-aware generation |
| Rendering | Remotion | Programmatic, scalable |
| Distribution | Blotato | Multi-platform (YT, TikTok, etc.) |
| Analytics | YouTube APIs | Native ecosystem |
| Time-series | QuestDB | Fast for metrics |

---
## 🏁 Definition of Done

- [x] Can create campaign in NocoDB → video published to YouTube ✅ (Blotato integration complete)
- [x] Multi-platform publishing (YouTube + TikTok) ✅ (Blotato API)
- [ ] Performance data feeds back to strategy engine
- [ ] A/B testing running on thumbnails
- [ ] <$5 cost per video produced
- [ ] 24-hour trend-to-publish pipeline

