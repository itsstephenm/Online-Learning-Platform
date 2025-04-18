from .models import Student

class StudentMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response
    
    def process_template_response(self, request, response):
        if hasattr(request, 'user') and request.user.is_authenticated:
            if request.user.groups.filter(name='STUDENT').exists():
                if hasattr(response, 'context_data') and response.context_data is not None:
                    try:
                        # Add student to context if not already present
                        if 'student' not in response.context_data:
                            student = Student.objects.get(user=request.user)
                            response.context_data['student'] = student
                    except Student.DoesNotExist:
                        pass
        return response 