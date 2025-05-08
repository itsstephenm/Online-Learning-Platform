import os
import sys

# Add your project directory to the sys.path
path = os.path.dirname(os.path.abspath(__file__))
if path not in sys.path:
    sys.path.append(path)

# Point to the correct settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "final_year_project.settings")

# Get the Django WSGI application
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Passenger web server needs this variable
from app import application as passenger_wsgi 