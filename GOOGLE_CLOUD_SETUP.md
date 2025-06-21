# Google Cloud Text-to-Speech Setup Guide

This guide explains how to optionally configure Google Cloud Text-to-Speech for premium voice quality.

## ⚠️ Important Notes

- **Google Cloud TTS requires billing setup** (credit card) even for free tier usage
- **Your personal credentials will be used** (unlike the default gTTS which is anonymous)
- **Costs apply after free tier**: $4-16 per million characters
- **gTTS (default) is completely free** and doesn't require any setup

## Free Tier Limits

- **Standard voices**: 4 million characters free per month
- **WaveNet voices**: 1 million characters free per month

## Setup Steps

### 1. Install Google Cloud TTS Library

```bash
pip install google-cloud-texttospeech
```

### 2. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing one
3. **Enable billing** (required even for free tier)

### 3. Enable Text-to-Speech API

1. Go to APIs & Services > Library
2. Search for "Cloud Text-to-Speech API"
3. Click "Enable"

### 4. Create Service Account

1. Go to IAM & Admin > Service Accounts
2. Click "Create Service Account"
3. Give it a name (e.g., "tts-service")
4. Assign role: "Cloud Text-to-Speech API User"

### 5. Download Credentials

1. Click on your service account
2. Go to "Keys" tab
3. Click "Add Key" > "Create new key"
4. Choose JSON format
5. Download the file to a secure location

### 6. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
# Enable Google Cloud TTS
MCP_TTS_USE_GOOGLE_CLOUD=true

# Set voice type (standard or wavenet)
MCP_TTS_GOOGLE_VOICE_TYPE=standard

# Point to your credentials file
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\service-account-key.json

# Use full locale codes for better voice selection
MCP_TTS_DEFAULT_VOICE=en-US
```

### 7. Restart Claude Desktop

Close Claude Desktop completely and restart to load the new configuration.

## Usage

The server will automatically:
1. Try Google Cloud TTS first (if configured)
2. Fall back to gTTS if Google Cloud fails
3. Use gTTS if Google Cloud is not configured

## Voice Codes

### gTTS (default)
- `en` - English
- `fr` - French
- `es` - Spanish
- `de` - German

### Google Cloud TTS
- `en-US` - US English
- `en-GB` - British English
- `fr-FR` - French (France)
- `es-ES` - Spanish (Spain)
- `de-DE` - German
- And many more...

## Monitoring Usage

Use the `tts_status` tool to check:
- Which service is currently active
- Configuration status
- Available services

## Cost Monitoring

In Google Cloud Console:
1. Go to Billing > Reports
2. Filter by "Cloud Text-to-Speech API"
3. Monitor character usage

## Security

- **Keep your JSON key file secure**
- **Don't share or commit it to git**
- **Store it outside your project directory**
