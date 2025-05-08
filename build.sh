#!/bin/bash
set -o errexit

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setting up environment variables..."
export DJANGO_SETTINGS_MODULE="final_year_project.settings"

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Running database migrations..."
python manage.py migrate --noinput

echo "Build completed successfully!"

# This line is added to trigger a new build on Render
echo "Running build script with python-decouple support..."

