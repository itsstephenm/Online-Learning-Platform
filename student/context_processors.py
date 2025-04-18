from .models import Student

def student_profile(request):
    """
    Context processor to add student profile to template context
    """
    context = {}
    
    if request.user.is_authenticated:
        try:
            if request.user.groups.filter(name='STUDENT').exists():
                student = Student.objects.get(user=request.user)
                context['student'] = student
        except Student.DoesNotExist:
            pass
            
    return context 