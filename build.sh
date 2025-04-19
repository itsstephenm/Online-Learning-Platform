#!/usr/bin/env bash
# exit on error
set -eo pipefail

echo "Starting build process..."

# Setup environment for better performance
export PYTHONUNBUFFERED=1

# Check Python version and try to use 3.8 if available
if command -v python3.8 &>/dev/null; then
  echo "Using Python 3.8"
  PYTHON_CMD=python3.8
elif command -v python3 &>/dev/null; then
  echo "Using Python 3"
  PYTHON_CMD=python3
else
  echo "Using system Python"
  PYTHON_CMD=python
fi

echo "Python version:"
$PYTHON_CMD --version

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  $PYTHON_CMD -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
python -m pip install --upgrade pip

# Check if we're on Netlify
if [ -n "$NETLIFY" ]; then
  echo "Detected Netlify environment, using Netlify-specific requirements..."
  python -m pip install -r requirements-netlify.txt
else
  echo "Using standard requirements..."
  python -m pip install -r requirements.txt
fi

# Create .env file from Netlify environment variables if it doesn't exist
if [ ! -f ".env" ]; then
  echo "Creating .env file from environment variables..."
  echo "SECRET_KEY='$SECRET_KEY'" > .env
  echo "DEBUG=False" >> .env
  echo "DATABASE_URL='$DATABASE_URL'" >> .env
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
