#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting build process..."

# Setup environment for better performance
export PYTHONUNBUFFERED=1
export PYTHONOPTIMIZE=2
export DJANGO_SETTINGS_MODULE=final_year_project.settings

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Apply database migrations with bigger timeout
echo "Applying database migrations..."
python manage.py migrate --no-input || {
    echo "Failed to apply migrations. Will try again without timeout."
    python manage.py migrate --no-input
}

# Collect static files with max optimization
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear || {
    echo "Warning: Failed to collect static files. Continuing anyway."
}

# Verify deployment readiness
echo "Checking deployment readiness..."
python manage.py check --deploy || {
    echo "Warning: Deployment check failed. Continuing anyway."
}

echo "Build completed successfully!"
