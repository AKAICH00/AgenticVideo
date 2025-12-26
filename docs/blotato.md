# Blotato API Integration Guide

## Overview

Blotato is a multi-platform content publishing API that enables posting videos and content to multiple social platforms from a single API call.

**Supported Platforms:**
- YouTube (videos, shorts)
- TikTok
- LinkedIn
- Pinterest
- Threads
- Bluesky

**Base URL:** `https://backend.blotato.com/v2`

---

## Setup

### 1. Get Your API Key

1. Go to [blotato.com/settings/api](https://blotato.com/settings/api)
2. Generate a new API key
3. Copy the key (format: `blt_xxxxxxxxxxxxxxxxxxxx`)

### 2. Connect Social Accounts

**CRITICAL:** Before you can publish, you must connect your social accounts:

1. Go to [blotato.com/settings/social-accounts](https://blotato.com/settings/social-accounts)
2. Click "Connect" for each platform you want to publish to
3. Complete the OAuth flow for each platform

> **Note:** If you get `401 Unauthorized` when publishing, it means you haven't connected any accounts yet.

### 3. Configure Environment

Set the API key in your environment:

```bash
# In .env.local or .env
BLOTATO_API_KEY=blt_your_api_key_here
```

For Kubernetes deployments, create the secret:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: blotato-secrets
  namespace: viral-video-agents
type: Opaque
stringData:
  api_key: "blt_your_api_key_here"
```

---

## API Reference

### Authentication

All requests require the `blotato-api-key` header:

```bash
curl -H "blotato-api-key: blt_your_api_key" \
     https://backend.blotato.com/v2/users/me/accounts
```

### Rate Limits

- **Posts:** 30 posts/minute
- **Media Uploads:** 10 uploads/minute

Our client automatically adds a 2-second delay between posts.

### Endpoints

#### List Connected Accounts

```
GET /users/me/accounts
```

Returns list of connected social accounts.

**Response:**
```json
{
  "accounts": [
    {
      "id": "acc_xxx",
      "platform": "youtube",
      "name": "My Channel",
      "connected": true
    }
  ]
}
```

#### Publish Content

```
POST /v2/posts
```

**Request Body:**
```json
{
  "targets": [
    {
      "targetType": "youtube",
      "title": "Video Title",
      "privacyStatus": "public",
      "shouldNotifySubscribers": false,
      "containsSyntheticMedia": true
    },
    {
      "targetType": "tiktok",
      "title": "Video Title",
      "privacyLevel": "public",
      "aiGeneratedContent": true
    }
  ],
  "mediaUrls": ["https://example.com/video.mp4"],
  "thumbnailUrl": "https://example.com/thumbnail.jpg",
  "scheduledTime": "2024-01-15T14:00:00Z"
}
```

---

## Platform-Specific Options

### YouTube

| Field | Type | Description |
|-------|------|-------------|
| `targetType` | string | `"youtube"` |
| `title` | string | Video title (max 100 chars) |
| `description` | string | Video description |
| `privacyStatus` | string | `"public"`, `"private"`, `"unlisted"` |
| `shouldNotifySubscribers` | boolean | Send notification to subscribers |
| `containsSyntheticMedia` | boolean | Mark as AI-generated |
| `tags` | string[] | Video tags |

### TikTok

| Field | Type | Description |
|-------|------|-------------|
| `targetType` | string | `"tiktok"` |
| `title` | string | Video title |
| `privacyLevel` | string | `"public"`, `"friends"`, `"private"` |
| `aiGeneratedContent` | boolean | Mark as AI-generated |
| `disableComments` | boolean | Disable comments |
| `disableDuet` | boolean | Disable duets |
| `disableStitch` | boolean | Disable stitch |

### LinkedIn

| Field | Type | Description |
|-------|------|-------------|
| `targetType` | string | `"linkedin"` |
| `text` | string | Post text/description |
| `visibility` | string | `"public"`, `"connections"` |

---

## Code Examples

### Python Client Usage

```python
from services.publisher import BlotaoClient, YouTubeTarget, TikTokTarget

async def publish_video():
    client = BlotaoClient()  # Uses BLOTATO_API_KEY env var

    # Single platform
    result = await client.publish_to_youtube(
        video_url="https://r2.example.com/video.mp4",
        title="My Amazing Video",
        description="Check out this video!",
        privacy="public",
        notify_subscribers=True,
        synthetic_media=True,  # AI-generated content
    )

    # Multi-platform
    targets = [
        YouTubeTarget(
            title="My Video",
            description="Description here",
            privacy="public",
        ),
        TikTokTarget(
            title="My Video",
            privacy="public",
            ai_generated=True,
        ),
    ]

    results = await client.publish_video(
        video_url="https://r2.example.com/video.mp4",
        targets=targets,
        thumbnail_url="https://r2.example.com/thumb.jpg",
    )

    for platform, result in results.items():
        if result.success:
            print(f"{platform}: Published! ID={result.post_id}")
        else:
            print(f"{platform}: Failed - {result.error}")
```

### Quick Publish Helper

```python
from services.publisher import publish_video_simple

# One-liner for YouTube
result = await publish_video_simple(
    video_url="https://example.com/video.mp4",
    title="Quick Video",
    platform="youtube",
)
```

### List Connected Accounts

```python
client = BlotaoClient()
accounts = await client.list_accounts()
for acc in accounts:
    print(f"{acc['platform']}: {acc['name']}")
```

---

## Integration with AgenticVideo

The `PublishNode` in the orchestrator automatically uses Blotato when configured:

```python
# services/orchestrator/nodes.py
class PublishNode(BaseNode):
    async def execute(self, state):
        if self.blotato_enabled:
            client = BlotaoClient()

            # Auto-publish to YouTube
            youtube_target = YouTubeTarget(
                title=state.video_title,
                description=state.description,
                privacy="private",  # Start private for review
                synthetic_media=True,
            )

            # Also TikTok for shorts (<60s)
            targets = [youtube_target]
            if state.duration < 60:
                targets.append(TikTokTarget(
                    title=state.video_title,
                    ai_generated=True,
                ))

            results = await client.publish_video(
                video_url=state.final_video_url,
                targets=targets,
            )
```

---

## Troubleshooting

### 401 Unauthorized

**Cause:** No social accounts connected in Blotato dashboard.

**Fix:**
1. Go to [blotato.com/settings/social-accounts](https://blotato.com/settings/social-accounts)
2. Connect your YouTube/TikTok/etc accounts
3. Try publishing again

### 429 Too Many Requests

**Cause:** Rate limit exceeded.

**Fix:** Wait 1 minute before retrying. The client has built-in rate limiting, but if you're making parallel requests, you may hit the limit.

### Video Upload Failed

**Cause:** Video URL not publicly accessible.

**Fix:** Ensure your video URL is:
- Publicly accessible (no auth required)
- Direct link to video file (not a page)
- Valid video format (mp4, webm, mov)

### Scheduled Post Not Publishing

**Cause:** Time format or timezone issue.

**Fix:** Use ISO 8601 format with UTC timezone:
```
2024-01-15T14:00:00Z
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BLOTATO_API_KEY` | Yes | Your Blotato API key |
| `DEFAULT_PRIVACY_STATUS` | No | Default: `"private"` |
| `TARGET_TIMEZONE` | No | For scheduling, default: `"America/New_York"` |

---

## Files

| File | Description |
|------|-------------|
| `services/publisher/blotato_client.py` | Blotato API client |
| `services/publisher/__init__.py` | Module exports |
| `services/orchestrator/nodes.py` | PublishNode integration |
| `k8s/08-publisher.yaml` | Kubernetes deployment |

---

## Links

- [Blotato Dashboard](https://blotato.com)
- [API Settings](https://blotato.com/settings/api)
- [Social Account Connections](https://blotato.com/settings/social-accounts)
