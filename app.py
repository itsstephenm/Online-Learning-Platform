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

# Monkey patch for gevent
try:
    from gevent import monkey
    monkey.patch_all()
    logging.info("Applied gevent monkey patches")
except ImportError:
    logging.warning("Gevent not available, skipping monkey patching")

# Disable pandas and numpy auto-importing at startup
os.environ["PANDAS_IMPORTS"] = "skip"  # Custom env var we'll check before imports
os.environ["NUMPY_IMPORTS"] = "skip"

try:
    logging.info("Initializing Django WSGI application")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "final_year_project.settings")
    
    # Implement lazy module loading pattern
    from django.core.wsgi import get_wsgi_application
    app = get_wsgi_application()
    logging.info("Django WSGI application initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Django application: {e}")
    # Re-raise to make the error visible in the logs
    raise

# This file is used by Render's default command - do not delete 