---
description: Deploy AgenticVideo to Kubernetes with ArgoCD
---

# AgenticVideo Deployment Runbook

This runbook provides step-by-step instructions for deploying AgenticVideo to Kubernetes using ArgoCD.

## Prerequisites

- [ ] Access to `k8s-deployments` GitHub repo
- [ ] Access to `AgenticVideo` GitHub repo
- [ ] `kubectl` configured for the cluster
- [ ] Docker installed for local builds (if needed)
- [ ] Domain configured and pointing to cluster ingress IP

---

## Step 1: Fix Docker Image (One-Time)

The `Dockerfile.agents` must include all source directories.

```bash
# Verify Dockerfile.agents contains:
# COPY agents/ ./agents/
# COPY services/ ./services/
# COPY core/ ./core/
# COPY cli/ ./cli/
# COPY main.py .
```

// turbo
```bash
# Build and push manually (until CI/CD is set up)
cd /root/projects/AgenticVideo
docker build -t ghcr.io/akaich00/viral-agents:v2.5 -f Dockerfile.agents .
docker push ghcr.io/akaich00/viral-agents:v2.5
```

---

## Step 2: Set Up CI/CD (One-Time)

// turbo
```bash
# Create GitHub Actions directory
mkdir -p /root/projects/AgenticVideo/.github/workflows
```

Create `.github/workflows/build-push.yaml` with the workflow from the implementation plan.

```bash
# Commit and push
git add .github/
git commit -m "Add CI/CD workflow"
git push origin main
```

---

## Step 3: Add to k8s-deployments Repo

```bash
# Clone the GitOps repo
git clone https://github.com/AKAICH00/k8s-deployments.git
cd k8s-deployments
```

// turbo
```bash
# Create app directory
mkdir -p apps/viral-video-agents
```

Copy manifests from `AgenticVideo/k8s/` to `apps/viral-video-agents/`:

```bash
# Copy all K8s manifests
cp /root/projects/AgenticVideo/k8s/*.yaml apps/viral-video-agents/
```

Create `kustomization.yaml`:

```yaml
# apps/viral-video-agents/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - 00-namespace-secrets.yaml
  - 01-postgres.yaml
  - 02-nocodb.yaml
  - 03-agents.yaml
  - 04-remotion.yaml
  - 05-intelligence-layer.yaml
  - 06-orchestrator.yaml
  - 08-publisher.yaml

images:
  - name: ghcr.io/akaich00/viral-agents
    newTag: v2.5
```

Commit and push:

```bash
git add apps/viral-video-agents/
git commit -m "Add viral-video-agents app"
git push origin main
```

---

## Step 4: Create ArgoCD Application

Create `argocd/viral-video-agents-app.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: viral-video-agents
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/AKAICH00/k8s-deployments.git
    targetRevision: main
    path: apps/viral-video-agents
  destination:
    server: https://kubernetes.default.svc
    namespace: viral-video-agents
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

// turbo
```bash
# Apply ArgoCD Application
kubectl apply -f argocd/viral-video-agents-app.yaml
```

---

## Step 5: Set Up Production TLS

// turbo
```bash
# Create production ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
      - http01:
          ingress:
            class: traefik
EOF
```

Update Ingress in `apps/viral-video-agents/ingress.yaml` with:
- Real domain
- TLS configuration
- `cert-manager.io/cluster-issuer: letsencrypt-prod` annotation

---

## Step 6: Verify Deployment

// turbo
```bash
# Check ArgoCD sync status
kubectl get applications -n argocd
```

// turbo
```bash
# Check all pods are running
kubectl get pods -n viral-video-agents
```

// turbo
```bash
# Check TLS certificate
kubectl get certificates -n viral-video-agents
```

---

## Troubleshooting

### Pods in ImagePullBackOff

```bash
# Check if imagePullSecrets is configured
kubectl get pod <pod-name> -n viral-video-agents -o yaml | grep -A2 imagePullSecrets

# Verify ghcr-secret exists
kubectl get secret ghcr-secret -n viral-video-agents
```

### ArgoCD Out of Sync

```bash
# Force sync
argocd app sync viral-video-agents --force

# Or via kubectl
kubectl patch application viral-video-agents -n argocd --type merge -p '{"operation":{"sync":{}}}'
```

### TLS Certificate Not Issued

```bash
# Check cert-manager logs
kubectl logs -n cert-manager -l app=cert-manager

# Check certificate status
kubectl describe certificate -n viral-video-agents
```

---

## Image Update Process

When code changes are pushed to `main`:

1. GitHub Actions builds and pushes new image with SHA tag
2. Update `apps/viral-video-agents/kustomization.yaml` with new tag:
   ```yaml
   images:
     - name: ghcr.io/akaich00/viral-agents
       newTag: <new-sha-or-version>
   ```
3. Commit and push to `k8s-deployments`
4. ArgoCD auto-syncs the change
