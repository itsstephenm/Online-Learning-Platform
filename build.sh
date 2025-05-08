#!/usr/bin/env bash
# exit on error
set -eo pipefail

echo "Starting build process..."

# Setup environment for better performance
export PYTHONUNBUFFERED=1
export PYTHON_VERSION=${PYTHON_VERSION:-3.11.8}
export PYTHONDONTWRITEBYTECODE=1  # Don't write .pyc files
export MPLCONFIGDIR=/tmp          # For matplotlib
export NUMBA_CACHE_DIR=/tmp       # For numba (if used)

# Check Python version
echo "Python version:"
python --version

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || true

# Install dependencies
echo "Installing dependencies..."
python -m pip install --upgrade pip

# Check if we're on Vercel
if [ -n "$VERCEL" ] || [ -n "$VERCEL_ENV" ]; then
  echo "Detected Vercel environment, using Vercel-specific requirements..."
  if [ -f "requirements-vercel.txt" ]; then
    python -m pip install -r requirements-vercel.txt
  else
    echo "Warning: requirements-vercel.txt not found, using standard requirements..."
    python -m pip install -r requirements.txt
  fi
else
  echo "Using standard requirements..."
  python -m pip install -r requirements.txt
fi

# Create .env file from environment variables if it doesn't exist
if [ ! -f ".env" ]; then
  echo "Creating .env file from environment variables..."
  echo "SECRET_KEY='${SECRET_KEY:-YOUR_SECRET_KEY}'" > .env
  echo "DEBUG=False" >> .env
  echo "PORT=${PORT:-10000}" >> .env
  
  # Add specific settings to avoid proxy issues with OpenAI
  echo "OPENAI_API_BASE=https://openrouter.ai/api/v1" >> .env
  
  if [ -n "$DATABASE_URL" ]; then
    echo "DATABASE_URL='$DATABASE_URL'" >> .env
  fi
  
  # Add Vercel-specific settings if on Vercel
  if [ -n "$VERCEL" ] || [ -n "$VERCEL_ENV" ]; then
    echo "VERCEL=true" >> .env
    echo "VERCEL_ENV=${VERCEL_ENV:-production}" >> .env
  fi
fi

# Apply database migrations
echo "Applying database migrations..."
if [ -n "$VERCEL" ] || [ -n "$VERCEL_ENV" ]; then
  echo "Skipping migrations on Vercel build (will be applied at runtime)"
else
  python manage.py migrate --no-input || {
      echo "Failed to apply migrations. Will try again without timeout."
      python manage.py migrate --no-input
  }
fi

# Collect static files
echo "Collecting static files..."
if [ -n "$VERCEL" ] || [ -n "$VERCEL_ENV" ]; then
  echo "Collecting static files for Vercel..."
  export DJANGO_SETTINGS_MODULE=final_year_project.vercel_settings
  python manage.py collectstatic --noinput --clear
else
  python manage.py collectstatic --noinput --clear || {
    echo "Warning: Failed to collect static files. Continuing anyway."
  }
fi

echo "Build completed successfully!"

# This line is added to trigger a new build on Render
echo "Running build script with python-decouple support..."

