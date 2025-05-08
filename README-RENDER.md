# Render Deployment Instructions

This document provides instructions for deploying the Final Year Project to Render.com.

## Prerequisites

1. A Render account (sign up at https://render.com)
2. Your project code in a Git repository (GitHub, GitLab, etc.)

## Deployment Steps

### 1. Set up a PostgreSQL database on Render

1. Log in to your Render dashboard
2. Navigate to "PostgreSQL" in the left sidebar
3. Click "New PostgreSQL"
4. Configure your database:
   - Name: `final_year_project_db`
   - User: `final_year_project`
   - Database: `final_year_project`
   - Plan: Select a plan that fits your needs (starter for development)
5. Click "Create Database"
6. Once created, note the "Internal Database URL" - you'll need this later

### 2. Deploy a Web Service

1. In the Render dashboard, navigate to "Web Services"
2. Click "New Web Service"
3. Connect your Git repository
4. Configure the service:
   - Name: `final_year_project`
   - Environment: `Python`
   - Build Command: `./build.sh`
   - Start Command: `gunicorn final_year_project.wsgi:application --bind=0.0.0.0:$PORT --timeout 60 --workers 2 --log-level info`
   - Plan: Select a plan that fits your needs (starter for development)
5. Add the following environment variables:
   - `DATABASE_URL`: The Internal Database URL from your PostgreSQL instance
   - `SECRET_KEY`: A secure random string for Django
   - `DEBUG`: `False`
   - `OPENROUTER_API_KEY`: Your OpenRouter API key for AI functionality
   - `OPENROUTER_MODEL_NAME`: The model name for OpenRouter
6. Click "Create Web Service"

### 3. Verify Deployment

1. Wait for the deployment to complete
2. Click the service URL to access your application
3. Visit `https://your-service-url.onrender.com/admin/` to verify the admin interface works

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

1. **Database connection issues**: Make sure your DATABASE_URL environment variable is correct
2. **Static files not loading**: Ensure collectstatic ran successfully in the build process
3. **Timeouts**: Consider adjusting the Gunicorn timeout settings in gunicorn.conf.py
4. **500 errors**: Check the logs in the Render dashboard for detailed error messages

## Maintenance

To update your deployment:

1. Push changes to your Git repository
2. Render will automatically deploy the changes if auto-deploy is enabled
3. Alternatively, trigger a manual deploy from the Render dashboard 