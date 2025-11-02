# Streamlit Cloud Deployment Setup

## Current Issue

Streamlit Cloud is using old ChromaDB credentials. Update the secrets in Streamlit Cloud.

## Step-by-Step Fix

### 1. Go to Streamlit Cloud Dashboard

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Navigate to your app
3. Click **"Settings"** (⚙️ icon)

### 2. Update Secrets

Click on **"Secrets"** section and update with:

```toml
[chromadb]
api_key = "YOUR_CHROMADB_API_KEY"
tenant = "YOUR_TENANT_ID"
database = "AI Engineer"

[azure_openai]
endpoint = "https://your-endpoint.openai.azure.com/"
api_key = "YOUR_AZURE_API_KEY"
api_version = "2025-01-01-preview"
deployment = "gpt-4o-mini"
embed_deployment = "text-embedding-3-small"

[mongodb]
uri = "mongodb+srv://user:password@cluster.mongodb.net/"
database = "vaia_market_analyst"
```

### 3. Save and Redeploy

1. Click **"Save"**
2. App will automatically restart
3. Wait for deployment (2-3 minutes)
4. Check logs to verify successful startup

### 4. Verify Deployment

Look for these in the logs:
```
✅ Application startup complete
✅ Uvicorn running
✅ No ChromaDB connection errors
```

### 5. Test Ingestion

1. Upload a PDF in the Streamlit UI
2. Should see: "✅ Ingested X chunks"
3. Start chatting!

## Local Testing (Optional)

Test credentials before deploying:

```bash
# Activate environment
source .venv/bin/activate

# Test ChromaDB connection
python test_chromadb.py

# Test PDF ingestion
python run_ingest.py market.pdf
```

## Troubleshooting

### "Could not connect to tenant"

- Check API key is correct in your ChromaDB dashboard
- Verify tenant ID from your ChromaDB account
- Ensure secrets are saved and app restarted

### "Missing required environment variable"

- Azure OpenAI secrets not configured
- Check `.streamlit/secrets.toml` has all required fields

### "Import errors"

- Restart deployment
- Check `requirements.txt` is up to date
- Verify Python 3.11+ is selected

## Quick Checklist

- [ ] Secrets updated in Streamlit Cloud
- [ ] All credentials copied correctly
- [ ] App restarted after saving
- [ ] No errors in deployment logs
- [ ] PDF ingestion works
- [ ] Chat responds properly

## Support

If issues persist:
1. Check Streamlit Cloud logs
2. Verify credentials via `test_chromadb.py` locally
3. Ensure latest code is pushed to `main` branch

