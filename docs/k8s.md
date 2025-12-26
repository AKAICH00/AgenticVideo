# Kubernetes Infrastructure Guide

## Cluster Info

| Node | Role | IP | OS |
|------|------|----|----|
| k8s-master | control-plane | **46.62.253.219** | Ubuntu 22.04.5 |
| k8s-worker-1 | worker | 37.27.222.165 | Ubuntu 22.04.5 |
| k8s-worker-2 | worker | 77.42.26.144 | Ubuntu 22.04.5 |

**K3s Version:** v1.33.6+k3s1
**Container Runtime:** containerd://2.1.5

```bash
# SSH to master
ssh root@46.62.253.219

# Check cluster status
kubectl get nodes
```

---

## Overview

AgenticVideo runs on Kubernetes with the following architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    viral-video-agents namespace                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  PostgreSQL  │    │    Redis     │    │   NocoDB     │      │
│  │  (StatefulSet│    │ (Deployment) │    │ (Deployment) │      │
│  │   + PVC)     │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                    │
│  ┌──────────────────────────┴───────────────────────────────┐  │
│  │                     Service Mesh                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                   │                   │               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Agents     │    │ Orchestrator │    │  Publisher   │      │
│  │  (director,  │    │  (LangGraph) │    │  (Blotato)   │      │
│  │  visualist)  │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                             │                                    │
│  ┌──────────────────────────┴───────────────────────────────┐  │
│  │              Intelligence Layer                           │  │
│  │  (TrendTrigger, FeedbackLoop, StrategyEngine)            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
k8s/
├── 00-namespace-secrets.yaml   # Namespace + all secrets
├── 01-postgres.yaml            # PostgreSQL StatefulSet
├── 02-nocodb.yaml              # NocoDB UI for data
├── 03-agents.yaml              # AI agents (director, visualist)
├── 04-remotion.yaml            # Video rendering service
├── 05-intelligence-layer.yaml  # Trend detection + feedback
├── 06-orchestrator.yaml        # LangGraph workflow engine
├── 08-publisher.yaml           # Multi-platform publishing
├── argocd-application.yaml     # GitOps deployment
├── kustomization.yaml          # Kustomize config
└── cluster/
    └── letsencrypt-prod.yaml   # SSL certificates
```

---

## Deployment Order

Apply manifests in numeric order:

```bash
# 1. Create namespace and secrets
kubectl apply -f k8s/00-namespace-secrets.yaml

# 2. Database layer
kubectl apply -f k8s/01-postgres.yaml
kubectl apply -f k8s/02-nocodb.yaml

# 3. Core services
kubectl apply -f k8s/03-agents.yaml
kubectl apply -f k8s/04-remotion.yaml

# 4. Intelligence layer
kubectl apply -f k8s/05-intelligence-layer.yaml

# 5. Orchestrator
kubectl apply -f k8s/06-orchestrator.yaml

# 6. Publishing
kubectl apply -f k8s/08-publisher.yaml
```

Or use Kustomize:

```bash
kubectl apply -k k8s/
```

---

## Components

### 00-namespace-secrets.yaml

Creates the namespace and all secrets:

| Secret | Purpose |
|--------|---------|
| `db-secrets` | PostgreSQL connection URL |
| `api-secrets` | All API keys (Google, Kie, ElevenLabs, etc.) |
| `ghcr-secret` | GitHub Container Registry pull credentials |

**Update secrets before deploying:**

```yaml
# k8s/00-namespace-secrets.yaml
stringData:
  google_api_key: "YOUR_REAL_KEY"
  kie_api_key: "YOUR_REAL_KEY"
  # ... etc
```

### 01-postgres.yaml

PostgreSQL database with persistent storage.

| Resource | Type | Details |
|----------|------|---------|
| `postgres` | StatefulSet | 1 replica, postgres:16 |
| `postgres-service` | Service | Port 5432, ClusterIP |
| `postgres-storage` | PVC | 10Gi |

**Connect from pod:**
```
postgresql://postgres:password@postgres-service:5432/viral_video_db
```

### 02-nocodb.yaml

NocoDB provides a spreadsheet UI for database management.

| Resource | Type | Details |
|----------|------|---------|
| `nocodb` | Deployment | 1 replica |
| `nocodb-service` | Service | Port 8080 |

### 03-agents.yaml

AI agents for content generation:

| Deployment | Command | Purpose |
|------------|---------|---------|
| `agent-director` | `agents.director` | Campaign orchestration |
| `agent-visualist` | `agents.visualist` | Visual generation |

**Image:** `ghcr.io/akaich00/viral-agents:v2.4`

### 04-remotion.yaml

Video rendering service using Remotion.

| Resource | Details |
|----------|---------|
| `remotion` | Deployment with puppeteer |
| Port | 3000 |

### 05-intelligence-layer.yaml

Trend detection and performance feedback:

| Deployment | Purpose |
|------------|---------|
| `trend-monitor` | Monitors trending topics |
| `strategy-engine` | Generates content strategies |
| `redis` | Caching layer |

**Services:**
- `trend-monitor-service`: Port 8761
- `strategy-engine-service`: Port 8762
- `redis-service`: Port 6379

### 06-orchestrator.yaml

LangGraph workflow engine:

| Resource | Details |
|----------|---------|
| `orchestrator` | Main workflow runner |
| Port | 8760 |

**Workflow nodes:** PLANNER → SCRIPT → STORYBOARD → MOTION → VISUAL → QUALITY → COMPOSE → REPURPOSE → PUBLISH

### 08-publisher.yaml

Multi-platform video publishing:

| Resource | Details |
|----------|---------|
| `video-publisher` | Deployment |
| `publisher-scheduler` | CronJob (every 15 min) |
| Port | 8766 |

**Secrets:**
- `blotato-secrets`: Blotato API key (recommended)
- `youtube-secrets`: YouTube OAuth (fallback)

See [blotato.md](./blotato.md) for publishing setup.

---

## Secrets Reference

### api-secrets

```yaml
stringData:
  # AI Models
  google_api_key: ""      # Gemini API
  kie_api_key: ""         # Kie/Fal video generation

  # Storage
  r2_audit_access_key: "" # Cloudflare R2
  r2_audit_secret_key: ""

  # Audio
  elevenlabs_api_key: ""  # Text-to-speech
  sync_labs_api_key: ""   # Lip sync

  # Observability
  langsmith_api_key: ""   # LangSmith tracing
  langfuse_secret_key: "" # Langfuse (optional)
  langfuse_public_key: ""
```

### blotato-secrets

```yaml
stringData:
  api_key: "blt_your_api_key"
```

### youtube-secrets (optional fallback)

```yaml
stringData:
  client_id: ""
  client_secret: ""
  refresh_token: ""
```

---

## Resource Limits

| Service | Memory Request | Memory Limit | CPU Request | CPU Limit |
|---------|---------------|--------------|-------------|-----------|
| postgres | 256Mi | 1Gi | 100m | 500m |
| orchestrator | 512Mi | 2Gi | 200m | 1000m |
| publisher | 256Mi | 1Gi | 100m | 500m |
| agents | 256Mi | 1Gi | 100m | 500m |
| redis | 128Mi | 256Mi | 100m | 200m |

---

## Health Checks

All services have readiness and liveness probes:

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8760
  initialDelaySeconds: 10
  periodSeconds: 10

livenessProbe:
  httpGet:
    path: /health
    port: 8760
  initialDelaySeconds: 30
  periodSeconds: 30
```

---

## Common Commands

### Check Status

```bash
# All pods
kubectl get pods -n viral-video-agents

# All services
kubectl get svc -n viral-video-agents

# PVCs
kubectl get pvc -n viral-video-agents
```

### View Logs

```bash
# Orchestrator
kubectl logs -f deploy/orchestrator -n viral-video-agents

# Publisher
kubectl logs -f deploy/video-publisher -n viral-video-agents

# Agents
kubectl logs -f deploy/agent-director -n viral-video-agents
```

### Exec into Pod

```bash
# PostgreSQL
kubectl exec -it sts/postgres -n viral-video-agents -- psql -U postgres -d viral_video_db

# Redis
kubectl exec -it deploy/redis -n viral-video-agents -- redis-cli
```

### Restart Deployment

```bash
kubectl rollout restart deploy/orchestrator -n viral-video-agents
```

### Scale

```bash
kubectl scale deploy/agent-visualist --replicas=2 -n viral-video-agents
```

---

## GitOps with ArgoCD

The `argocd-application.yaml` enables GitOps deployment:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: viral-video-agents
  namespace: argocd
spec:
  source:
    repoURL: https://github.com/akaich00/viral-agents
    path: k8s
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: viral-video-agents
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

---

## SSL/TLS

Cluster directory contains Let's Encrypt configuration:

```yaml
# k8s/cluster/letsencrypt-prod.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

---

## Troubleshooting

### Pod Won't Start

```bash
# Check events
kubectl describe pod <pod-name> -n viral-video-agents

# Check image pull
kubectl get events -n viral-video-agents | grep -i pull
```

### Database Connection Issues

```bash
# Test from inside cluster
kubectl run -it --rm debug --image=postgres:16 -n viral-video-agents -- \
  psql postgresql://postgres:password@postgres-service:5432/viral_video_db
```

### Secret Not Found

```bash
# List secrets
kubectl get secrets -n viral-video-agents

# Verify secret content (base64)
kubectl get secret api-secrets -n viral-video-agents -o yaml
```

### CronJob Not Running

```bash
# Check job history
kubectl get jobs -n viral-video-agents

# Check cronjob status
kubectl describe cronjob publisher-scheduler -n viral-video-agents
```

---

## Environment Variables

All services share these common env vars:

| Variable | Source | Required |
|----------|--------|----------|
| `DATABASE_URL` | db-secrets.url | Yes |
| `GOOGLE_API_KEY` | api-secrets.google_api_key | Yes |
| `REDIS_URL` | hardcoded | Yes |
| `LOG_LEVEL` | hardcoded | No |

---

## Network Topology

Internal service DNS:

| Service | DNS | Port |
|---------|-----|------|
| PostgreSQL | `postgres-service:5432` | 5432 |
| Redis | `redis-service:6379` | 6379 |
| NocoDB | `nocodb-service:8080` | 8080 |
| Orchestrator | `orchestrator-service:8760` | 8760 |
| Publisher | `video-publisher:8766` | 8766 |
| Trend Monitor | `trend-monitor-service:8761` | 8761 |
| Strategy Engine | `strategy-engine-service:8762` | 8762 |
