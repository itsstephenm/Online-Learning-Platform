import logging
import time
import resource
import gc
import psutil
import tracemalloc
from django.conf import settings

logger = logging.getLogger(__name__)

class MemoryUsageMiddleware:
    """
    Middleware to log memory usage for requests exceeding a threshold.
    This helps identify memory leaks and high-usage views.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.memory_threshold_mb = getattr(settings, 'MEMORY_USAGE_THRESHOLD_MB', 100)
        self.trace_memory = getattr(settings, 'TRACE_MEMORY_USAGE', settings.DEBUG)
        
        if self.trace_memory:
            tracemalloc.start()
    
    def __call__(self, request):
        # Start monitoring
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        if self.trace_memory:
            snapshot1 = tracemalloc.take_snapshot()
        
        # Process the request
        response = self.get_response(request)
        
        # Calculate memory usage
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_memory - start_memory
        execution_time = time.time() - start_time
        
        # Log if memory usage exceeds threshold or request is slow
        if memory_delta > self.memory_threshold_mb or execution_time > 1.0:
            logger.warning(
                f"Memory usage: path={request.path}, "
                f"delta={memory_delta:.2f}MB, "
                f"total={end_memory:.2f}MB, "
                f"time={execution_time:.2f}s"
            )
            
            # Get detailed memory info if tracing is enabled
            if self.trace_memory:
                snapshot2 = tracemalloc.take_snapshot()
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                
                for stat in top_stats[:10]:
                    logger.info(f"{stat}")
                
                # Run garbage collection to free memory
                gc.collect()
        
        return response 