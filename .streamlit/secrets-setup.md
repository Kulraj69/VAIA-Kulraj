# Streamlit Secrets Setup Guide

## Local Development

For local development, create a file `.streamlit/secrets.toml` based on the example:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml` with your actual credentials.

**Note:** This file is already added to `.gitignore` and will not be committed to the repository.

## Streamlit Cloud

For deploying to Streamlit Cloud, you need to add your secrets through the Streamlit Cloud interface:

1. Go to your app settings on [streamlit.io](https://streamlit.io)
2. Click on "Settings" → "Secrets"
3. Add your secrets in the following format:

```toml
[chromadb]
api_key = "your-chromadb-api-key"
tenant = "your-chromadb-tenant"
database = "your-chromadb-database"

[azure_openai]
endpoint = "https://your-endpoint.openai.azure.com/"
api_key = "your-azure-openai-api-key"
api_version = "2025-01-01-preview"
deployment = "gpt-4o-mini"
embed_deployment = "text-embedding-3-small"

[mongodb]
uri = "mongodb+srv://user:password@cluster.mongodb.net/"
database = "vaia_market_analyst"
```

4. Save the secrets
5. Your app will automatically restart with the new secrets

## Required Credentials

### Azure OpenAI
- `endpoint`: Your Azure OpenAI endpoint URL
- `api_key`: Your API key
- `api_version`: API version (typically "2025-01-01-preview")
- `deployment`: Chat deployment name (e.g., "gpt-4o-mini")
- `embed_deployment`: Embedding deployment name (e.g., "text-embedding-3-small")

### MongoDB (Optional)
- `uri`: MongoDB connection string
- `database`: Database name

### ChromaDB (Optional)
- `api_key`: ChromaDB API key
- `tenant`: ChromaDB tenant
- `database`: ChromaDB database name

## Security Notes

- Never commit actual secrets to the repository
- Use different secrets for different environments (dev, staging, prod)
- Rotate credentials regularly
- Use Streamlit Cloud's built-in secrets management for production deployments

