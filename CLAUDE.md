# AgenticVideo - Viral Content Production System

## 🎯 System Purpose
Autonomous AI-powered system for producing viral long-form and short-form video content at scale.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE LAYER                           │
│  YouTube Intel → Performance Tracker → Strategy Engine          │
│  (What's trending)  (What worked)     (What to make)           │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT LAYER                                   │
│  Director → Visualist → Audio → Renderer → Publisher           │
│  (Script)   (Images)    (Voice) (Assembly) (Upload)            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                    │
│  PostgreSQL (state) │ Redis (queues) │ R2 (assets)             │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
AgenticVideo/
├── agents/                    # Production agents (Director, Visualist, etc.)
│   ├── director/              # Script generation with Claude
│   ├── visualist/             # Image/video generation with KIE
│   ├── renderer/              # Remotion video assembly
│   ├── audio/                 # ElevenLabs + Suno
│   └── shared/                # Common utilities (db.py)
├── services/                  # Support services
│   ├── youtube_intelligence/  # Trend analysis
│   ├── performance_tracker/   # Analytics collection
│   ├── strategy_engine/       # Content planning
│   └── publisher/             # Multi-platform upload
├── remotion/                  # Video templates
│   └── src/compositions/      # React video components
├── k8s/                       # Kubernetes deployments
├── sql/                       # Database migrations
└── ACTIVE_SPRINT.md           # Current sprint context
```

## 🔑 Critical Rules

### 1. ALWAYS CHECK ACTIVE_SPRINT.md FIRST
Before any work, read ACTIVE_SPRINT.md to understand:
- Current parallel work assignments
- What other agents are building
- Which files NOT to modify

### 2. DATA LAYER RULES
```python
# CORRECT: Use async database operations
async with db.connection() as conn:
    result = await conn.fetch("SELECT * FROM video_campaigns")

# WRONG: Sync operations block the event loop
result = psycopg2.connect(url).cursor().fetchall()
```

### 3. INTELLIGENCE BEFORE GENERATION
Never generate content blindly. Always:
1. Query `trending_topics` for current trends
2. Check `video_performance` for what worked
3. Use `content_recommendations` for prioritized topics

### 4. COST TRACKING IS MANDATORY
Every API call must log to `generation_costs`:
```python
await log_cost(
    campaign_id=campaign.id,
    service="claude",
    operation="script_gen",
    tokens_used=response.usage.total_tokens,
    api_cost_usd=calculate_cost(response.usage)
)
```

## 🔧 Key Integrations

### Google Ecosystem (Leverage This!)
```
YouTube Data API v3  → Trend detection, competitor analysis
YouTube Analytics    → Our video performance
Vertex AI           → Pattern recognition on trends
BigQuery            → Long-term analytics storage
```

### Technology Stack
| Use Case | Technology | Reason |
|----------|------------|--------|
| Script/Brief Gen | Google Gemini 2.0 | Fast, cost-effective |
| Visual Gen | KIE.ai + ComfyUI | Multi-model + local GPU |
| Voice | ElevenLabs | Best quality TTS |
| Lip Sync | SyncLabs | Realistic face sync |
| Video Assembly | Remotion | Programmatic rendering |
| Observability | LangSmith | Agent decision traces |

### Infrastructure (Tailscale Mesh)
| Service | Address | Purpose |
|---------|---------|---------|
| PostgreSQL | `100.121.2.100:5432` | Campaign state |
| NocoDB | `100.121.2.100:8082` | Campaign UI |
| ComfyUI | `100.64.0.2:8188` | Local GPU rendering |
| Remotion | `api.cckeeper.dev` | Video export API |

## 📊 Database Schema Quick Reference

```sql
-- Core Pipeline
video_campaigns     -- Main campaign state (new → script → visual → render → published)
script_drafts       -- Script versions with critic feedback
visual_scenes       -- Storyboard with generation status

-- Intelligence Layer
trending_topics     -- Current trends by platform/niche
video_performance   -- Performance metrics over time
content_recommendations -- AI-generated content briefs

-- Operations
generation_costs    -- API cost tracking
ab_tests           -- Thumbnail/title experiments
```

## 🚨 Anti-Patterns to Avoid

### DON'T: Generate Without Strategy
```python
# WRONG
prompt = "Write a viral video script about AI"
script = await claude.generate(prompt)

# CORRECT
trends = await get_top_trends(niche="tech")
recommendation = await strategy_engine.get_recommendation(trends)
script = await director.generate_script(recommendation)
```

### DON'T: Ignore Performance Feedback
```python
# WRONG
# Just keep generating content with same approach

# CORRECT
performance = await get_recent_performance()
if performance.avg_ctr < 0.05:
    await adjust_thumbnail_strategy()
if performance.avg_retention < 0.3:
    await improve_hook_quality()
```

### DON'T: Scatter Secrets
```python
# WRONG
api_key = "sk-ant-api03-..."  # Hardcoded

# WRONG
api_key = os.getenv("ANTHROPIC_API_KEY")  # Different env vars everywhere

# CORRECT
from agents.shared.config import settings
api_key = settings.anthropic_api_key  # Centralized config
```

## 🎬 Viral Content Formula

```
VIRALITY = (HOOK_STRENGTH × RETENTION × SHAREABILITY) / TIME_TO_VALUE

Where:
- HOOK_STRENGTH: First 3 seconds engagement (0-1)
- RETENTION: Average view percentage (0-1)
- SHAREABILITY: Emotional trigger strength (0-1)
- TIME_TO_VALUE: Seconds until "aha moment"
```

### Platform-Specific Parameters

| Platform | Duration | Hook Window | Key Factor |
|----------|----------|-------------|------------|
| YouTube Long | 8-12 min | 8 seconds | Watch time |
| YouTube Shorts | 45-58 sec | 1.5 seconds | Loop rate |
| TikTok | 21-34 sec | 0.8 seconds | Completion rate |

## 🔄 Development Workflow

1. **Check Sprint**: Read ACTIVE_SPRINT.md for assignments
2. **Branch**: `git checkout -b feature/STREAM-N-description`
3. **Implement**: Follow architecture in this file
4. **Test**: Run against test database
5. **Commit**: `[STREAM-N] description`
6. **Update Sprint**: Log progress in ACTIVE_SPRINT.md

## 📈 Success Metrics

- **Cost per video**: Target < $5
- **Trend-to-publish time**: Target < 24 hours
- **Quality gate pass rate**: Target > 80%
- **Views per dollar spent**: Target > 1000

## 🔗 Related Documentation

- [ACTIVE_SPRINT.md](./ACTIVE_SPRINT.md) - Current sprint context
- [sql/001_initial_schema.sql](./sql/001_initial_schema.sql) - Core schema
- [sql/003_add_performance_tables.sql](./sql/003_add_performance_tables.sql) - Analytics schema
- [k8s/](./k8s/) - Kubernetes deployments
