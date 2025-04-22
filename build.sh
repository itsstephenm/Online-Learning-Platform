#!/usr/bin/env bash
# exit on error
set -eo pipefail

echo "Starting build process..."

# Setup environment for better performance
export PYTHONUNBUFFERED=1
export PYTHON_VERSION=${PYTHON_VERSION:-3.8}

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
# Check if we're on Netlify
elif [ -n "$NETLIFY" ]; then
  echo "Detected Netlify environment, using Netlify-specific requirements..."
  if [ -f "requirements-netlify.txt" ]; then
    python -m pip install -r requirements-netlify.txt
  else
    echo "Using standard requirements..."
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
  if [ -n "$DATABASE_URL" ]; then
    echo "DATABASE_URL='$DATABASE_URL'" >> .env
  fi
fi

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate --no-input || {
    echo "Failed to apply migrations. Will try again without timeout."
    python manage.py migrate --no-input
}

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear || {
    echo "Warning: Failed to collect static files. Continuing anyway."
}

echo "Build completed successfully!"
