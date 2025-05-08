from final_year_project.wsgi import application
import os

# WSGI application for Vercel serverless functions
app = application

# Handle ASGI if needed
if os.environ.get('ASGI_ENABLED') == '1':
    from final_year_project.asgi import application as asgi_app
    async def asgi_handler(scope, receive, send):
        await asgi_app(scope, receive, send) 