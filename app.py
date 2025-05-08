# Render WSGI entry point
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    stream=sys.stdout
)

try:
    logging.info("Initializing Django WSGI application")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "final_year_project.settings")
    
    from django.core.wsgi import get_wsgi_application
    app = get_wsgi_application()
    logging.info("Django WSGI application initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Django application: {e}")
    # Re-raise to make the error visible in the logs
    raise

# This file is used by Render's default command - do not delete 