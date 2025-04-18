from .models import Teacher

class TeacherMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response
    
    def process_template_response(self, request, response):
        if hasattr(request, 'user') and request.user.is_authenticated:
            if request.user.groups.filter(name='TEACHER').exists():
                if hasattr(response, 'context_data') and response.context_data is not None:
                    try:
                        # Add teacher to context if not already present
                        if 'teacher' not in response.context_data:
                            teacher = Teacher.objects.get(user=request.user)
                            response.context_data['teacher'] = teacher
                    except Teacher.DoesNotExist:
                        pass
        return response 