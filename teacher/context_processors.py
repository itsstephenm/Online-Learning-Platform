from .models import Teacher

def teacher_profile(request):
    """
    Context processor to add teacher profile to template context
    """
    context = {}
    
    if request.user.is_authenticated:
        try:
            if request.user.groups.filter(name='TEACHER').exists():
                teacher = Teacher.objects.get(user=request.user)
                context['teacher'] = teacher
        except Teacher.DoesNotExist:
            pass
            
    return context 