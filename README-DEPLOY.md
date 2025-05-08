# Vercel Deployment Instructions

This document provides instructions for deploying the Final Year Project to Vercel.

## Prerequisites

1. A Vercel account (sign up at https://vercel.com)
2. Vercel CLI installed (optional, for local testing)
3. A PostgreSQL database (for production)

## Deployment Steps

### 1. Set up environment variables in Vercel

- **SECRET_KEY**: A secure Django secret key
- **DATABASE_URL**: Your PostgreSQL connection string
- **DEBUG**: Set to 'False'
- **OPENROUTER_API_KEY**: Your OpenRouter API key for AI functionality
- **OPENROUTER_MODEL_NAME**: The model name for OpenRouter

### 2. Deploy to Vercel

#### Using the Vercel Dashboard:

1. Log in to your Vercel account
2. Import your GitHub repository
3. Configure the project:
   - Build Command: `./build.sh`
   - Output Directory: Leave blank
   - Install Command: Leave blank
4. Configure environment variables as mentioned above
5. Deploy

#### Using the Vercel CLI:

```bash
# Install Vercel CLI if you haven't already
npm i -g vercel

# Login to Vercel
vercel login

# Deploy the project
vercel
```

### 3. Run migrations

After deploying, you need to run migrations on the deployed app:

```bash
vercel run python manage.py migrate
```

### 4. Create a superuser

Create an admin user:

```bash
vercel run python manage.py createsuperuser
```

## Troubleshooting

1. **Static files not loading**: Make sure STATICFILES_STORAGE is set correctly and static files are collected during build
2. **Database connection errors**: Verify your DATABASE_URL is correctly formatted
3. **500 errors**: Check Vercel logs for detailed error messages

## Maintenance

To update the deployment:

1. Push changes to your GitHub repository
2. Vercel will automatically deploy the changes

## Local Testing

Test your Vercel configuration locally:

```bash
vercel dev
```

This will run a local server that mimics the Vercel environment. 