"""
WSGI config for final_year_project project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os
import logging

logger = logging.getLogger(__name__)

# Disable any proxy settings to avoid OpenAI client issues
proxy_vars = [
    'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'no_proxy', 'NO_PROXY'
]

for var in proxy_vars:
    if var in os.environ:
        logger.info(f"Removing proxy environment variable: {var}")
        del os.environ[var]

# Set OpenAI API base URL
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'final_year_project.settings')

application = get_wsgi_application()
