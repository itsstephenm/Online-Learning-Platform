"""
Optimized database query functions to improve performance
"""
from django.db.models import Prefetch, Count, Sum, Avg, F, Q
from quiz.models import Question, Course, Result, QuestionAttempt, StudentSkillLevel
from student.models import Student
from teacher.models import Teacher

def get_course_questions(course_id, select_related=False, prefetch_related=False):
    """Get questions for a course with optional optimizations"""
    query = Question.objects.filter(course_id=course_id)
    
    if select_related:
        query = query.select_related('course')
    
    if prefetch_related:
        query = query.prefetch_related('explanations')
    
    return query

def get_student_course_results(student_id, course_id=None):
    """Get optimized student results for a course"""
    query = Result.objects.filter(student_id=student_id)
    
    if course_id:
        query = query.filter(exam_id=course_id)
    
    return query.select_related('student', 'exam')

def get_question_attempts_for_student(student_id, with_feedback=False):
    """Get optimized question attempts for a student"""
    query = QuestionAttempt.objects.filter(student_id=student_id)
    
    if with_feedback:
        query = query.select_related('question', 'student').prefetch_related('feedback')
    else:
        query = query.select_related('question', 'student')
    
    return query

def get_courses_with_question_counts():
    """Get all courses with precomputed question counts"""
    return Course.objects.annotate(question_count=Count('question'))

def get_student_skill_levels(student_id=None):
    """Get optimized student skill levels"""
    query = StudentSkillLevel.objects.select_related('student', 'course')
    
    if student_id:
        query = query.filter(student_id=student_id)
        
    return query

def get_teachers_with_status(status=True):
    """Get teachers filtered by status with optimized query"""
    return Teacher.objects.filter(status=status).select_related('user') 