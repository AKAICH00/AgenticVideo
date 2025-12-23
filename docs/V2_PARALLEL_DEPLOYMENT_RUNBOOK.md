# V2 Parallel Deployment Runbook

## Overview

This runbook provides step-by-step instructions for safely deploying the V2 Video Orchestrator alongside the existing OLD polling agents with **zero downtime** and **complete isolation**.

### Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    PARALLEL DEPLOYMENT                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OLD AGENTS (deployed)          V2 ORCHESTRATOR (new)           │
│  ┌─────────────────────┐        ┌─────────────────────┐         │
│  │ agent-director      │        │ video-orchestrator  │         │
│  │ agent-visualist     │        │ (single process)    │         │
│  │ agent-renderer      │        │                     │         │
│  └─────────────────────┘        └─────────────────────┘         │
│           │                              │                       │
│           ▼                              ▼                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     PostgreSQL                               ││
│  │  ┌───────────────────┐    ┌────────────────────┐            ││
│  │  │ status = 'new'    │    │ status = 'v2_*'    │            ││
│  │  │ processor = 'old' │    │ processor = 'new'  │            ││
│  │  └───────────────────┘    └────────────────────┘            ││
│  │       ↑ OLD polls              ↑ V2 polls                    ││
│  │       ZERO OVERLAP - COMPLETE ISOLATION                      ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Isolation Guarantee

| System | Status Values | Processor | Query Pattern |
|--------|---------------|-----------|---------------|
| OLD | `new`, `in_scripting`, `script_approved`, `generating_visuals`, `rendering` | `old` or NULL | `WHERE status = 'new'` |
| V2 | `v2_pending`, `v2_planning`, `v2_scripting`, `v2_storyboarding`, `v2_motion`, `v2_visual`, `v2_quality`, `v2_composing`, `v2_repurposing` | `new` | `WHERE status = 'v2_pending'` |
| Shared | `published`, `failed` | preserves original | Final states only |

**CRITICAL**: There is **ZERO OVERLAP** between status values. OLD agents will **NEVER** pick up V2 campaigns.

---

## Pre-Deployment Checklist

### 1. Verify Current State

```bash
# Check current deployments
kubectl get pods -n viral-video-agents

# Verify OLD agents are healthy
kubectl logs -n viral-video-agents deployment/agent-director --tail=20
kubectl logs -n viral-video-agents deployment/agent-visualist --tail=20
kubectl logs -n viral-video-agents deployment/agent-renderer --tail=20

# Check current campaign distribution
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
SELECT status, COUNT(*)
FROM video_campaigns
GROUP BY status
ORDER BY status;
"
```

### 2. Verify Required Secrets Exist

```bash
# Check api-secrets has required keys
kubectl get secret api-secrets -n viral-video-agents -o jsonpath='{.data}' | jq 'keys'

# Required keys for V2:
# - google_api_key
# - kie_api_key
# - elevenlabs_api_key
# - sync_labs_api_key (optional)
# - langfuse_secret_key (optional)
# - langfuse_public_key (optional)
# - r2_audit_access_key
# - r2_audit_secret_key
```

### 3. Backup Database

```bash
# Create backup before migration
kubectl exec -n viral-video-agents deploy/postgres -- pg_dump -U postgres viral_video_db > backup_$(date +%Y%m%d_%H%M%S).sql
```

---

## Phase 1: Database Migration

### Apply V2 Isolation Schema

```bash
# Apply the V2 isolation migration
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -f - << 'EOF'
-- V2 Isolation Migration
-- sql/007_v2_isolation.sql

-- Add processor column for tracking which system handles campaigns
ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS processor VARCHAR(10) DEFAULT NULL;

-- Add V2-specific tracking columns
ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS v2_session_id VARCHAR(50) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS v2_started_at TIMESTAMP DEFAULT NULL,
ADD COLUMN IF NOT EXISTS v2_completed_at TIMESTAMP DEFAULT NULL,
ADD COLUMN IF NOT EXISTS v2_node_history JSONB DEFAULT '[]'::jsonb;

-- Index for processor-based queries
CREATE INDEX IF NOT EXISTS idx_campaigns_processor
ON video_campaigns(processor);

-- Index for V2 status queries
CREATE INDEX IF NOT EXISTS idx_campaigns_v2_status
ON video_campaigns(status)
WHERE status LIKE 'v2_%';

-- Mark all existing campaigns as OLD
UPDATE video_campaigns
SET processor = 'old'
WHERE processor IS NULL;

-- Create migration status view
CREATE OR REPLACE VIEW v2_migration_status AS
SELECT
    processor,
    status,
    COUNT(*) as count,
    MIN(created_at) as oldest,
    MAX(created_at) as newest
FROM video_campaigns
GROUP BY processor, status
ORDER BY processor, status;
EOF

echo "✅ V2 isolation schema applied"
```

### Verify Migration

```bash
# Check new columns exist
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
\d video_campaigns
"

# Check all existing campaigns marked as 'old'
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
SELECT processor, COUNT(*) FROM video_campaigns GROUP BY processor;
"
```

---

## Phase 2: Deploy V2 Orchestrator (Passive Mode)

### Apply Secrets Update

```bash
# Update secrets if needed (edit k8s/00-namespace-secrets.yaml first)
kubectl apply -f k8s/00-namespace-secrets.yaml
```

### Deploy Orchestrator with PROCESSOR_MODE=old

This deploys the orchestrator but it **won't process any campaigns** yet.

```bash
# Ensure PROCESSOR_MODE is set to "old" in k8s/06-orchestrator.yaml
grep -A1 "PROCESSOR_MODE" k8s/06-orchestrator.yaml
# Should show: value: "old"

# Deploy the orchestrator
kubectl apply -f k8s/06-orchestrator.yaml

# Wait for deployment
kubectl rollout status deployment/video-orchestrator -n viral-video-agents --timeout=120s

# Verify pod is running
kubectl get pods -n viral-video-agents -l app=video-orchestrator
```

### Verify Health

```bash
# Port-forward to test locally
kubectl port-forward -n viral-video-agents svc/video-orchestrator 8765:8765 &

# Check health endpoint
curl http://localhost:8765/health
# Expected: {"status":"healthy","mode":"old",...}

# Check V2 endpoints exist
curl http://localhost:8765/v2/info
curl http://localhost:8765/v2/isolation
curl http://localhost:8765/v2/pipeline

# Kill port-forward
pkill -f "port-forward.*8765"
```

---

## Phase 3: Verify Complete Isolation

### Run Isolation Monitor

```bash
# Run the isolation monitor script
python scripts/monitor_v2_isolation.py --server-url http://localhost:8765 --database

# Expected output:
# === V2 Isolation Check (CRITICAL) ===
# ✓ OK ISOLATION OK - No violations detected
# ✓ OK Database isolation enabled
```

### Check for Isolation Violations

```bash
# Direct database check for violations
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
SELECT id, topic, status, processor,
       CASE
           WHEN processor = 'new' AND status IN ('new', 'in_scripting', 'script_approved', 'generating_visuals', 'rendering')
               THEN 'V2 processor with OLD status - VIOLATION'
           WHEN processor = 'old' AND status LIKE 'v2_%'
               THEN 'OLD processor with V2 status - VIOLATION'
           WHEN status LIKE 'v2_%' AND processor IS NULL
               THEN 'V2 status without processor - VIOLATION'
           ELSE 'OK'
       END as check_result
FROM video_campaigns
WHERE (processor = 'new' AND status IN ('new', 'in_scripting', 'script_approved', 'generating_visuals', 'rendering'))
   OR (processor = 'old' AND status LIKE 'v2_%')
   OR (status LIKE 'v2_%' AND processor IS NULL);
"

# Expected: 0 rows (no violations)
```

---

## Phase 4: Enable Split Routing

### Switch to PROCESSOR_MODE=split

In split mode:
- **Premium campaigns** → V2 Orchestrator
- **Bulk campaigns** → OLD Agents

```bash
# Update PROCESSOR_MODE to "split"
kubectl set env deployment/video-orchestrator -n viral-video-agents PROCESSOR_MODE=split

# Wait for rollout
kubectl rollout status deployment/video-orchestrator -n viral-video-agents --timeout=60s

# Verify mode changed
curl http://localhost:8765/health
# Expected: {"status":"healthy","mode":"split",...}
```

### Create Test V2 Campaign

```bash
# Create a premium test campaign via V2 API
curl -X POST http://localhost:8765/generate \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": "test-v2-001",
    "topic": "Test V2 Campaign",
    "niche": "test",
    "quality_tier": "premium"
  }'

# Expected response includes:
# "db_status": "v2_pending"
# "processor": "new"
```

### Monitor Test Campaign

```bash
# Stream progress via SSE
curl -N http://localhost:8765/monitor/test-v2-001

# Or use the CLI monitor
python main.py monitor test-v2-001
```

### Verify Pipeline Status

```bash
# Check V2 pipeline
curl http://localhost:8765/v2/pipeline

# Check database
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
SELECT * FROM v2_migration_status;
"
```

---

## Phase 5: Monitor (24-48 Hours)

### Set Up Continuous Monitoring

```bash
# Run continuous monitoring (Ctrl+C to stop)
python scripts/monitor_v2_isolation.py \
  --server-url http://video-orchestrator.viral-video-agents:8765 \
  --continuous \
  --interval 60

# Or deploy as K8s CronJob
kubectl apply -f - << 'EOF'
apiVersion: batch/v1
kind: CronJob
metadata:
  name: v2-isolation-monitor
  namespace: viral-video-agents
spec:
  schedule: "*/5 * * * *"  # Every 5 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: monitor
            image: ghcr.io/your-org/viral-agents:v2
            command: ["python", "scripts/monitor_v2_isolation.py"]
            args: ["--server-url", "http://video-orchestrator:8765", "--database"]
          restartPolicy: OnFailure
EOF
```

### Key Metrics to Watch

```bash
# Campaign distribution by processor
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
SELECT
    processor,
    status,
    COUNT(*) as count,
    ROUND(AVG(EXTRACT(EPOCH FROM (updated_at - created_at))), 2) as avg_seconds
FROM video_campaigns
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY processor, status
ORDER BY processor, status;
"

# Error rate comparison
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
SELECT
    processor,
    COUNT(*) FILTER (WHERE status = 'published') as success,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'published') / NULLIF(COUNT(*), 0), 2) as success_rate
FROM video_campaigns
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY processor;
"
```

---

## Phase 6: Full Cutover to V2

### Switch to PROCESSOR_MODE=new

After successful 24-48h monitoring with no issues:

```bash
# Update PROCESSOR_MODE to "new" (all traffic to V2)
kubectl set env deployment/video-orchestrator -n viral-video-agents PROCESSOR_MODE=new

# Wait for rollout
kubectl rollout status deployment/video-orchestrator -n viral-video-agents --timeout=60s

# Verify mode changed
curl http://localhost:8765/health
# Expected: {"status":"healthy","mode":"new",...}
```

### Wait for OLD Agents to Drain

OLD agents will finish processing any campaigns they already picked up.

```bash
# Monitor OLD agents draining
watch -n 10 'kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
SELECT status, COUNT(*)
FROM video_campaigns
WHERE processor = '\''old'\''
  AND status NOT IN ('\''published'\'', '\''failed'\'')
GROUP BY status;
"'

# Wait until 0 rows (all OLD campaigns complete)
```

---

## Phase 7: Scale Down OLD Agents

### Verify Complete Drain

```bash
# Confirm no in-progress OLD campaigns
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
SELECT COUNT(*) as in_progress
FROM video_campaigns
WHERE processor = 'old'
  AND status NOT IN ('published', 'failed');
"
# Expected: 0
```

### Scale Down OLD Agents

```bash
# Scale down to 0 replicas (keep deployments for rollback)
kubectl scale deployment/agent-director -n viral-video-agents --replicas=0
kubectl scale deployment/agent-visualist -n viral-video-agents --replicas=0
kubectl scale deployment/agent-renderer -n viral-video-agents --replicas=0

# Verify scaled down
kubectl get pods -n viral-video-agents -l app=agent-director
kubectl get pods -n viral-video-agents -l app=agent-visualist
kubectl get pods -n viral-video-agents -l app=agent-renderer
# Expected: No pods running
```

---

## Rollback Procedures

### Rollback Level 1: Switch Back to Split Mode

If V2 has issues but some traffic is working:

```bash
kubectl set env deployment/video-orchestrator -n viral-video-agents PROCESSOR_MODE=split
kubectl rollout status deployment/video-orchestrator -n viral-video-agents
```

### Rollback Level 2: Disable V2 Completely

If V2 is having serious issues:

```bash
# Switch to OLD only
kubectl set env deployment/video-orchestrator -n viral-video-agents PROCESSOR_MODE=old
kubectl rollout status deployment/video-orchestrator -n viral-video-agents

# Scale down orchestrator
kubectl scale deployment/video-orchestrator -n viral-video-agents --replicas=0

# Scale up OLD agents if they were scaled down
kubectl scale deployment/agent-director -n viral-video-agents --replicas=1
kubectl scale deployment/agent-visualist -n viral-video-agents --replicas=1
kubectl scale deployment/agent-renderer -n viral-video-agents --replicas=1
```

### Rollback Level 3: Database Cleanup

If V2 campaigns need to be abandoned:

```bash
# Mark all in-progress V2 campaigns as failed
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
UPDATE video_campaigns
SET status = 'failed',
    error_log = 'Rollback: V2 system disabled',
    updated_at = NOW()
WHERE processor = 'new'
  AND status LIKE 'v2_%'
  AND status NOT IN ('published', 'failed');
"
```

---

## Troubleshooting

### Issue: V2 Orchestrator Not Starting

```bash
# Check pod status
kubectl describe pod -n viral-video-agents -l app=video-orchestrator

# Check logs
kubectl logs -n viral-video-agents deployment/video-orchestrator --tail=100

# Common issues:
# - Missing secrets: Check kubectl get secret api-secrets -n viral-video-agents
# - Database connection: Check DATABASE_URL secret
# - Import errors: Check Python dependencies
```

### Issue: Isolation Violation Detected

```bash
# Immediately switch to OLD mode
kubectl set env deployment/video-orchestrator -n viral-video-agents PROCESSOR_MODE=old

# Investigate violations
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
SELECT id, topic, status, processor, created_at
FROM video_campaigns
WHERE (processor = 'new' AND status IN ('new', 'in_scripting', 'script_approved', 'generating_visuals', 'rendering'))
   OR (processor = 'old' AND status LIKE 'v2_%')
   OR (status LIKE 'v2_%' AND processor IS NULL)
LIMIT 10;
"

# Fix violations manually
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
-- Reset V2 campaigns with wrong status
UPDATE video_campaigns
SET status = 'v2_pending'
WHERE processor = 'new'
  AND status IN ('new', 'in_scripting', 'script_approved', 'generating_visuals', 'rendering');
"
```

### Issue: High Error Rate on V2

```bash
# Check recent V2 failures
kubectl exec -n viral-video-agents deploy/postgres -- psql -U postgres -d viral_video_db -c "
SELECT id, topic, status, error_log, created_at
FROM video_campaigns
WHERE processor = 'new' AND status = 'failed'
ORDER BY created_at DESC
LIMIT 10;
"

# Check orchestrator logs for errors
kubectl logs -n viral-video-agents deployment/video-orchestrator --tail=200 | grep -i error
```

---

## Post-Migration Cleanup

After successful migration (1-2 weeks stable):

### Remove OLD Agent Resources

```bash
# Delete OLD agent deployments
kubectl delete deployment agent-director -n viral-video-agents
kubectl delete deployment agent-visualist -n viral-video-agents
kubectl delete deployment agent-renderer -n viral-video-agents

# Delete their services if any
kubectl delete service agent-director -n viral-video-agents
kubectl delete service agent-visualist -n viral-video-agents
kubectl delete service agent-renderer -n viral-video-agents
```

### Archive OLD Agent Code

```bash
# Move OLD agents to archive directory
mkdir -p archive/agents
mv agents/director.py archive/agents/
mv agents/visualist.py archive/agents/
mv agents/renderer.py archive/agents/

# Update CLAUDE.md to remove OLD references
```

### Update Feature Flags

```bash
# Remove PROCESSOR_MODE from deployment (hardcode "new" in code)
kubectl set env deployment/video-orchestrator -n viral-video-agents PROCESSOR_MODE-
```

---

## Quick Reference

### Key Commands

| Action | Command |
|--------|---------|
| Check mode | `curl http://localhost:8765/health` |
| Check isolation | `curl http://localhost:8765/v2/isolation` |
| Check pipeline | `curl http://localhost:8765/v2/pipeline` |
| Switch to split | `kubectl set env deployment/video-orchestrator -n viral-video-agents PROCESSOR_MODE=split` |
| Switch to new | `kubectl set env deployment/video-orchestrator -n viral-video-agents PROCESSOR_MODE=new` |
| Rollback to old | `kubectl set env deployment/video-orchestrator -n viral-video-agents PROCESSOR_MODE=old` |
| Monitor isolation | `python scripts/monitor_v2_isolation.py --server-url http://localhost:8765` |

### Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Server health and current mode |
| `GET /v2/isolation` | Isolation status (MUST return isolated=true) |
| `GET /v2/pipeline` | V2 campaign counts by status |
| `GET /v2/info` | V2 orchestrator version and config |
| `POST /generate` | Create new V2 campaign |
| `GET /monitor/{id}` | SSE progress stream |

### Status Values

| V2 Status | Description |
|-----------|-------------|
| `v2_pending` | Awaiting processing |
| `v2_planning` | Planner node active |
| `v2_scripting` | Script generation |
| `v2_storyboarding` | Storyboard creation |
| `v2_motion` | Motion extraction |
| `v2_visual` | Visual generation |
| `v2_quality` | Quality check |
| `v2_composing` | Final composition |
| `v2_repurposing` | Short-form clips |
| `published` | Successfully completed |
| `failed` | Processing failed |
