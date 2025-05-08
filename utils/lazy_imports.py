import os
import importlib
import logging

# Dictionary to store lazy-loaded modules
_lazy_modules = {}

def lazy_import(module_name):
    """
    Lazily import a module only when it's first used.
    
    Example usage:
        pandas = lazy_import('pandas')
        # later when you need it:
        df = pandas().DataFrame(...)  # notice the parentheses
    """
    if module_name in _lazy_modules:
        return _lazy_modules[module_name]
        
    def _import_module():
        if module_name not in _lazy_modules:
            logging.info(f"Lazy-loading module: {module_name}")
            try:
                _lazy_modules[module_name] = importlib.import_module(module_name)
            except ImportError as e:
                logging.error(f"Failed to import {module_name}: {e}")
                raise
        return _lazy_modules[module_name]
        
    return _import_module

# Predefined lazy imports for common heavy modules
pandas = lazy_import('pandas')
numpy = lazy_import('numpy')
openai = lazy_import('openai')

# Example of a lazy-loaded OpenAI client
_openai_client = None

def get_openai_client():
    """Get or initialize the OpenAI client on first use"""
    global _openai_client
    
    if _openai_client is None:
        # Import only when needed
        from openai import OpenAI
        import os
        
        logging.info("Initializing OpenAI client")
        api_key = os.environ.get('OPENAI_API_KEY')
        _openai_client = OpenAI(api_key=api_key)
        
    return _openai_client 