"""
Django settings for Vercel deployment.
"""

from .settings import *  # Import base settings

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# Configure allowed hosts for Vercel
ALLOWED_HOSTS = ['.vercel.app', 'localhost', '127.0.0.1']

# Simplified static file serving
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Vercel-specific database settings (SQLite for serverless)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# Update if DATABASE_URL is provided
if os.environ.get('DATABASE_URL'):
    import dj_database_url
    DATABASES['default'] = dj_database_url.config(
        conn_max_age=600,
        conn_health_checks=True,
    )

# Configure CSRF and security settings for Vercel
CSRF_TRUSTED_ORIGINS = [
    'https://*.vercel.app',
]

# Disable password validation for development
AUTH_PASSWORD_VALIDATORS = []

# Set up logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
} 