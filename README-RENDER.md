# Render Deployment Instructions

This document provides instructions for deploying the Final Year Project to Render.com using SQLite3 database.

## Prerequisites

1. A Render account (sign up at https://render.com)
2. Your project code in a Git repository (GitHub, GitLab, etc.)

## Deployment Steps

### 1. Deploy a Web Service with SQLite3

1. In the Render dashboard, navigate to "Web Services"
2. Click "New Web Service"
3. Connect your Git repository
4. Configure the service:
   - Name: `final_year_project`
   - Environment: `Python`
   - Build Command: `./build.sh`
   - Start Command: `gunicorn final_year_project.wsgi:application --bind=0.0.0.0:$PORT --timeout 60 --workers 2 --log-level info`
   - Plan: Select "Basic" ($6/month)
5. Add the following environment variables:
   - `SECRET_KEY`: A secure random string for Django
   - `DEBUG`: `False`
   - `USE_SQLITE`: `True`
   - `OPENROUTER_API_KEY`: Your OpenRouter API key for AI functionality
   - `OPENROUTER_MODEL_NAME`: The model name for OpenRouter
6. Configure a disk for database persistence:
   - Name: `sqlite_data`
   - Mount Path: `/data`
   - Size: 1GB (or more as needed)
7. Click "Create Web Service"

### 2. Verify Deployment

1. Wait for the deployment to complete
2. Click the service URL to access your application
3. Visit `https://your-service-url.onrender.com/admin/` to verify the admin interface works

## How SQLite3 Persistence Works

The deployment is configured to:
1. Store your SQLite3 database file in the persistent disk at `/data/db.sqlite3`
2. Create a symbolic link from your project directory to the persistent database
3. Copy your existing database to persistent storage on first deploy

This ensures your database remains intact between deployments and service restarts.

## Manual Setup (if needed)

If you need to set up the database or create an admin user manually:

1. Open a Shell from the Render dashboard
2. Run migrations:
   ```bash
   python manage.py migrate
   ```
3. Create a superuser:
   ```bash
   python manage.py createsuperuser
   ```

## Troubleshooting

1. **Database issues**: Check if the symbolic link to `/data/db.sqlite3` exists and points to a valid file
2. **Static files not loading**: Ensure collectstatic ran successfully in the build process
3. **Timeouts**: Consider adjusting the Gunicorn timeout settings in gunicorn.conf.py
4. **500 errors**: Check the logs in the Render dashboard for detailed error messages

## Maintenance

To update your deployment:

1. Push changes to your Git repository
2. Render will automatically deploy the changes if auto-deploy is enabled
3. Alternatively, trigger a manual deploy from the Render dashboard 