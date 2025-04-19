#!/usr/bin/env bash
# exit on error
set -o errexit

# Setup environment for better performance
export PYTHONUNBUFFERED=1
export PYTHONOPTIMIZE=2
export DJANGO_SETTINGS_MODULE=final_year_project.settings

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Apply database migrations with bigger timeout
python manage.py migrate --no-input --timeout 120

# Collect static files with max optimization
python manage.py collectstatic --noinput --clear

# Verify deployment readiness
python manage.py check --deploy
