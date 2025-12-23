# DIFF.md - Intelligence Layer Integration Fixes

## For Agent Coordination - AgenticVideo Sprint

**Generated**: 2024-12-21
**Sprint**: sprint-20241220-unified
**Purpose**: Document exact changes needed for seamless intelligence layer integration

---

## Executive Summary

| Priority | File | Change Type | Status |
|----------|------|-------------|--------|
| P1 | `sql/005_fix_intelligence_columns.sql` | NEW | Required - BLOCKING |
| P2 | `k8s/05-intelligence-layer.yaml` | MODIFY | Required (Redis PVC) |
| P3 | `main.py` | MODIFY | Required (intelligence cmd) |
| P4 | `services/intelligence_bridge/feedback_loop.py` | OPTIONAL | Alternative to P1 |

---

## CHANGE 1: SQL Migration (BLOCKING)

### File: `sql/005_fix_intelligence_columns.sql` [NEW]

**Reason**: `feedback_loop.py` uses column names that don't exist in `sql/003_add_performance_tables.sql`

| Code Expects (feedback_loop.py:144-155) | Schema Has (003:38-64) | Action |
|----------------------------------------|------------------------|--------|
| `youtube_video_id` | `external_video_id` | Add alias column |
| `collected_at` | `snapshot_at` | Rename |
| `engagement_rate` | (missing) | Add column |
| `target_duration` (campaigns) | (missing) | Add column |

```sql
-- sql/005_fix_intelligence_columns.sql
-- Migration to align schema with intelligence layer code expectations

-- 1. Add engagement_rate to video_performance (feedback_loop.py:141)
ALTER TABLE video_performance
  ADD COLUMN IF NOT EXISTS engagement_rate FLOAT;

-- 2. Add youtube_video_id alias (feedback_loop.py:147)
ALTER TABLE video_performance
  ADD COLUMN IF NOT EXISTS youtube_video_id TEXT;

-- 3. Rename snapshot_at → collected_at (feedback_loop.py:149,154)
ALTER TABLE video_performance
  RENAME COLUMN snapshot_at TO collected_at;

-- 4. Update unique constraint for new column name
ALTER TABLE video_performance
  DROP CONSTRAINT IF EXISTS video_performance_campaign_id_platform_snapshot_at_key;
ALTER TABLE video_performance
  ADD CONSTRAINT video_performance_campaign_id_collected_at_key
  UNIQUE(campaign_id, platform, collected_at);

-- 5. Add target_duration to video_campaigns (feedback_loop.py:190)
ALTER TABLE video_campaigns
  ADD COLUMN IF NOT EXISTS target_duration INT;

-- 6. Backfill youtube_video_id from external_video_id
UPDATE video_performance
  SET youtube_video_id = external_video_id
  WHERE youtube_video_id IS NULL AND external_video_id IS NOT NULL;

-- 7. Create index for common query pattern
CREATE INDEX IF NOT EXISTS idx_performance_collected
  ON video_performance(collected_at DESC);
```

---

## CHANGE 2: K8s Redis PVC (User Requested)

### File: `k8s/05-intelligence-layer.yaml` [MODIFY lines 29-41]

**Current** (volatile emptyDir):
```yaml
# Lines 29-41
spec:
  containers:
  - name: redis
    image: redis:7-alpine
    ports:
    - containerPort: 6379
    volumeMounts:
    - name: redis-data
      mountPath: /data
  volumes:
  - name: redis-data
    emptyDir: {}  # DATA LOST ON RESTART
```

**Change to** (persistent):
```yaml
spec:
  containers:
  - name: redis
    image: redis:7-alpine
    command: ["redis-server", "--appendonly", "yes"]
    ports:
    - containerPort: 6379
    volumeMounts:
    - name: redis-data
      mountPath: /data
    resources:
      requests:
        memory: "128Mi"
        cpu: "100m"
      limits:
        memory: "256Mi"
        cpu: "200m"
  volumes:
  - name: redis-data
    persistentVolumeClaim:
      claimName: redis-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: viral-video-agents
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

---

## CHANGE 3: K8s Anthropic Key Cleanup (Optional)

### File: `k8s/05-intelligence-layer.yaml` [MODIFY lines 125-129]

**Current** (redundant - code uses Gemini now):
```yaml
# Lines 125-129 - Strategy Engine env
- name: ANTHROPIC_API_KEY
  valueFrom:
    secretKeyRef:
      name: api-secrets
      key: anthropic_api_key
```

**Action**: Remove these 5 lines. GOOGLE_API_KEY already defined at lines 130-134.

**Note**: This is optional - having both keys doesn't break anything, it's just redundant.

---

## CHANGE 4: Add Intelligence Command to main.py

### File: `main.py` [MODIFY]

**Insert after `status_parser` definition (line 266)**:

```python
    # Intelligence command
    intel_parser = subparsers.add_parser("intelligence", help="Start intelligence services")
    intel_parser.add_argument(
        "--niches",
        nargs="+",
        default=["tech", "finance"],
        help="Niches to monitor"
    )
    intel_parser.add_argument(
        "--mode",
        choices=["passive", "active"],
        default="passive",
        help="passive=queue only, active=auto-generate",
    )
```

**Add to command handling section (after line 313)**:

```python
    elif args.command == "intelligence":
        asyncio.run(start_intelligence(args.niches, args.mode))


async def start_intelligence(niches: list, mode: str):
    """Start intelligence background services."""
    import asyncpg
    from core.config import get_config
    from services.intelligence_bridge.trend_trigger import TrendTrigger
    from services.intelligence_bridge.feedback_loop import FeedbackLoop, PerformanceTracker

    config = get_config()
    logger.info(f"Starting intelligence services for niches: {niches}")

    db_pool = await asyncpg.create_pool(
        config.database.url,
        min_size=2,
        max_size=10,
    )

    trigger = TrendTrigger(db_pool=db_pool)
    feedback = FeedbackLoop(db_pool=db_pool)
    tracker = PerformanceTracker(feedback, niches)

    # Handle shutdown
    stop_event = asyncio.Event()

    def handle_signal():
        logger.info("Shutting down intelligence services...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    # Start services
    await asyncio.gather(
        trigger.start(niches, mode),
        tracker.start(),
    )

    await stop_event.wait()

    # Cleanup
    await trigger.stop()
    await tracker.stop()
    await feedback.close()
    await db_pool.close()
```

---

## Implementation Checklist

### For STREAM-1 Agent (Intelligence):
- [ ] Create `sql/005_fix_intelligence_columns.sql`
- [ ] Run migration via psql

### For STREAM-2 Agent (Core):
- [ ] Add `intelligence` command to `main.py`
- [ ] Test: `python main.py intelligence --niches tech --mode passive`

### For STREAM-3 Agent (Infrastructure):
- [ ] Update `k8s/05-intelligence-layer.yaml` with Redis PVC
- [ ] Apply: `kubectl apply -f k8s/05-intelligence-layer.yaml`
- [ ] Verify: `kubectl get pvc -n viral-video-agents`

---

## Verification Commands

```bash
# 1. Check migration applied
kubectl exec -n viral-video-agents deploy/postgres -- \
  psql -U postgres -c "\d video_performance" | grep -E "collected_at|engagement_rate|youtube_video_id"

# 2. Check Redis PVC
kubectl get pvc -n viral-video-agents | grep redis

# 3. Test intelligence command
python main.py intelligence --niches tech --mode passive

# 4. Verify full pipeline test
python tests/test_intelligence_pipeline.py
```

---

## Architecture After Fixes

```
                    python main.py intelligence
                              |
        +---------------------+---------------------+
        v                     v                     v
+---------------+    +------------------+   +----------------+
| TrendTrigger  |    |PerformanceTracker|   |  FeedbackLoop  |
| (30min poll)  |    |   (1hr poll)     |   |  (on-demand)   |
+-------+-------+    +--------+---------+   +-------+--------+
        |                     |                     |
        v                     |                     |
+---------------+             |                     |
|CampaignFactory|             |                     |
|  + Planner    |             |                     |
+-------+-------+             |                     |
        |                     |                     |
        v                     v                     v
+--------------------------------------------------------------+
|                    PostgreSQL (video_campaigns,               |
|                    video_performance, trending_topics)        |
+--------------------------------------------------------------+
        |
        v
+--------------------------+
| VideoGraph.run(state)    |
|   +- PlannerNode         |
|   +- ScriptNode          |
|   +- StoryboardNode      |
|   +- VisualNode (KIE)    |
|   +- QualityNode         |
|   +- ComposeNode         |
|   +- RepurposeNode       |
+--------------------------+
```

---

## Files Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `sql/005_fix_intelligence_columns.sql` | CREATE | ~30 |
| `k8s/05-intelligence-layer.yaml` | MODIFY | ~40 (Redis PVC) |
| `main.py` | MODIFY | ~50 (intelligence cmd) |

---

## Ready for Implementation

All changes documented above. Agents can work in parallel:
- **STREAM-1**: SQL migration
- **STREAM-2**: main.py additions
- **STREAM-3**: K8s config updates

No dependencies between streams - all can proceed simultaneously.
