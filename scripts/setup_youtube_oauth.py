#!/usr/bin/env python3
"""
YouTube OAuth Setup Helper

This script helps you get YouTube API OAuth2 credentials.
It handles the OAuth flow to get a refresh token.

Prerequisites:
1. Create OAuth credentials in Google Cloud Console:
   https://console.cloud.google.com/apis/credentials?project=openmlo

2. Choose "OAuth client ID" -> "Desktop app"

3. Download the JSON credentials file

Usage:
    python setup_youtube_oauth.py --credentials /path/to/client_secret.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials
except ImportError:
    print("Installing required packages...")
    os.system("pip install google-auth-oauthlib google-auth")
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials


# YouTube API scopes needed for upload
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


def create_manual_credentials():
    """Create credentials manually by prompting for client ID and secret."""
    print("\n" + "=" * 60)
    print("MANUAL CREDENTIAL ENTRY")
    print("=" * 60)
    print("\nIf you don't have a credentials JSON file, enter the values manually.")
    print("Get these from: https://console.cloud.google.com/apis/credentials\n")

    client_id = input("Enter Client ID: ").strip()
    client_secret = input("Enter Client Secret: ").strip()

    if not client_id or not client_secret:
        print("Error: Both Client ID and Client Secret are required")
        sys.exit(1)

    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def run_oauth_flow(credentials_data: dict) -> Credentials:
    """Run the OAuth flow to get credentials."""
    flow = InstalledAppFlow.from_client_config(credentials_data, SCOPES)

    print("\n" + "=" * 60)
    print("OAUTH AUTHORIZATION")
    print("=" * 60)
    print("\nA browser window will open for you to authorize access.")
    print("If it doesn't open, copy the URL from the terminal.\n")

    # Run local server for callback
    credentials = flow.run_local_server(
        port=8085,
        prompt="consent",
        access_type="offline",
    )

    return credentials


def main():
    parser = argparse.ArgumentParser(description="YouTube OAuth Setup Helper")
    parser.add_argument(
        "--credentials", "-c",
        help="Path to OAuth client credentials JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        default="youtube_credentials.json",
        help="Output file for credentials (default: youtube_credentials.json)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("YOUTUBE OAUTH SETUP")
    print("=" * 60)

    # Load or create credentials config
    if args.credentials and Path(args.credentials).exists():
        print(f"\nLoading credentials from: {args.credentials}")
        with open(args.credentials) as f:
            credentials_data = json.load(f)
    else:
        print("\nNo credentials file provided.")
        print("\nYou need to create OAuth credentials first:")
        print("1. Go to: https://console.cloud.google.com/apis/credentials?project=openmlo")
        print("2. Click '+ CREATE CREDENTIALS' -> 'OAuth client ID'")
        print("3. Select 'Desktop app' as application type")
        print("4. Name it 'AgenticVideo YouTube'")
        print("5. Click 'Create'")
        print("")

        choice = input("Do you want to enter credentials manually? (y/n): ").strip().lower()
        if choice == 'y':
            credentials_data = create_manual_credentials()
        else:
            print("\nPlease download the JSON file and run again with:")
            print(f"  python {sys.argv[0]} --credentials /path/to/client_secret.json")
            sys.exit(0)

    # Run OAuth flow
    try:
        credentials = run_oauth_flow(credentials_data)
    except Exception as e:
        print(f"\nError during OAuth flow: {e}")
        sys.exit(1)

    # Extract the values we need
    client_id = credentials_data.get("installed", {}).get("client_id", "")
    client_secret = credentials_data.get("installed", {}).get("client_secret", "")
    refresh_token = credentials.refresh_token

    print("\n" + "=" * 60)
    print("SUCCESS! Here are your credentials:")
    print("=" * 60)
    print(f"\nYOUTUBE_CLIENT_ID={client_id}")
    print(f"YOUTUBE_CLIENT_SECRET={client_secret}")
    print(f"YOUTUBE_REFRESH_TOKEN={refresh_token}")

    # Save to file
    output_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "token": credentials.token,
        "token_uri": "https://oauth2.googleapis.com/token",
        "scopes": SCOPES,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nCredentials saved to: {args.output}")

    # Print kubectl commands
    print("\n" + "=" * 60)
    print("KUBERNETES SECRET COMMANDS:")
    print("=" * 60)
    print(f"""
# Update the youtube-secrets in K8s:
kubectl create secret generic youtube-secrets \\
  --namespace=viral-video-agents \\
  --from-literal=client_id='{client_id}' \\
  --from-literal=client_secret='{client_secret}' \\
  --from-literal=refresh_token='{refresh_token}' \\
  --dry-run=client -o yaml | kubectl apply -f -
""")

    # Print .env format
    print("=" * 60)
    print("FOR .env FILE:")
    print("=" * 60)
    print(f"""
YOUTUBE_CLIENT_ID={client_id}
YOUTUBE_CLIENT_SECRET={client_secret}
YOUTUBE_REFRESH_TOKEN={refresh_token}
""")


if __name__ == "__main__":
    main()
