import math
import random
from django.db.models import Avg, Count, F, Max, Min
from quiz.models import Question, QuestionAttempt, StudentSkillLevel, AdaptiveQuizSettings, Course

def get_student_skill_level(student, course):
    """
    Get or create a student skill level for a specific course
    """
    skill_level, created = StudentSkillLevel.objects.get_or_create(
        student=student,
        course=course,
        defaults={
            'current_level': 5.0,  # Start at medium difficulty
            'confidence': 0.5      # Medium confidence
        }
    )
    return skill_level

def get_adaptive_questions(student, course, num_questions=10):
    """
    Get adaptive questions based on student's current skill level
    """
    # Get student's skill level for this course
    skill_level = get_student_skill_level(student, course)
    
    # Get or create adaptive settings for this course
    settings, _ = AdaptiveQuizSettings.objects.get_or_create(
        course=course,
        defaults={
            'is_adaptive': True,
            'min_difficulty': 1,
            'max_difficulty': 10,
            'difficulty_step': 0.5
        }
    )
    
    if not settings.is_adaptive:
        # If adaptive quizzing is not enabled, return random questions
        available_questions = Question.objects.filter(course=course)
        if available_questions.count() <= num_questions:
            return list(available_questions)
        else:
            return random.sample(list(available_questions), num_questions)
    
    # Get appropriate difficulty level questions
    current_level = skill_level.current_level
    
    # Map 1-10 scale to question marks (assuming higher marks = harder questions)
    # Determine min and max marks for this course
    mark_stats = Question.objects.filter(course=course).aggregate(
        min_marks=Min('marks'),
        max_marks=Max('marks')
    )
    min_marks = mark_stats['min_marks'] if mark_stats['min_marks'] is not None else 1
    max_marks = mark_stats['max_marks'] if mark_stats['max_marks'] is not None else 5
    
    # Normalize student level to question marks scale
    level_range = settings.max_difficulty - settings.min_difficulty
    marks_range = max_marks - min_marks
    
    if level_range == 0:
        normalized_level = min_marks
    else:
        normalized_level = min_marks + ((current_level - settings.min_difficulty) / level_range) * marks_range
    
    # Get questions with marks close to the student's level
    # We'll get a range around the student's level to ensure enough questions
    mark_range = max(1, marks_range / 3)  # Use at least 1 or 1/3 of the total range
    
    available_questions = Question.objects.filter(
        course=course,
        marks__gte=normalized_level - mark_range,
        marks__lte=normalized_level + mark_range
    )
    
    # If not enough questions in the specified range, expand the range
    if available_questions.count() < num_questions:
        available_questions = Question.objects.filter(course=course)
    
    # If still not enough questions, return all available
    if available_questions.count() <= num_questions:
        return list(available_questions)
    
    # Select questions with preference to those matching student's level
    questions = list(available_questions)
    questions.sort(key=lambda q: abs(q.marks - normalized_level))
    
    return questions[:num_questions]

def update_skill_level(student, course, question_attempts):
    """
    Update student's skill level based on their performance
    
    Args:
        student: Student object
        course: Course object
        question_attempts: List of QuestionAttempt objects
    """
    if not question_attempts:
        return
    
    # Get current skill level
    skill_level = get_student_skill_level(student, course)
    
    # Get adaptive settings
    settings, _ = AdaptiveQuizSettings.objects.get_or_create(
        course=course,
        defaults={
            'is_adaptive': True,
            'min_difficulty': 1,
            'max_difficulty': 10,
            'difficulty_step': 0.5
        }
    )
    
    # Calculate performance metrics
    total_questions = len(question_attempts)
    correct_answers = sum(1 for attempt in question_attempts if attempt.is_correct)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # Performance threshold: above this is considered "good"
    threshold = 0.7  # 70% correct answers
    
    # Adjust difficulty based on performance
    if accuracy > threshold:
        # Student did well, increase difficulty
        adjustment = settings.difficulty_step
    else:
        # Student struggled, decrease difficulty
        adjustment = -settings.difficulty_step
    
    # Update level
    new_level = skill_level.current_level + adjustment
    
    # Ensure level stays within bounds
    new_level = max(settings.min_difficulty, min(settings.max_difficulty, new_level))
    
    # Update confidence based on consistency of performance
    # If all answers are correct or all are wrong, high consistency
    # If mixed results, lower consistency
    consistency = abs(accuracy - 0.5) * 2  # Maps 0-1 to 0-1 with 0.5 accuracy giving lowest consistency
    
    # Blend old confidence with new data point
    # Weight based on number of questions (more questions = more weight to new data)
    weight_factor = min(total_questions / 10, 0.8)  # Cap at 80% weight to new data
    new_confidence = (skill_level.confidence * (1 - weight_factor)) + (consistency * weight_factor)
    
    # Update and save
    skill_level.current_level = new_level
    skill_level.confidence = new_confidence
    skill_level.save()
    
    return skill_level

def generate_quiz_analytics(student, course):
    """
    Generate analytics about a student's quiz performance for a course
    
    Returns:
        Dictionary with analytics data
    """
    # Get question attempts for this student and course
    attempts = QuestionAttempt.objects.filter(
        student=student,
        question__course=course
    )
    
    if not attempts.exists():
        return {
            'total_questions': 0,
            'accuracy': 0,
            'average_time': 0,
            'skill_level': 'Not available yet',
            'recommendations': ['Complete some quizzes to see recommendations']
        }
    
    # Calculate statistics
    total_questions = attempts.count()
    correct_answers = attempts.filter(is_correct=True).count()
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    avg_time = attempts.aggregate(avg_time=Avg('time_taken'))['avg_time'] or 0
    
    # Get skill level
    skill_level = get_student_skill_level(student, course)
    
    # Get adaptive settings for this course
    settings, _ = AdaptiveQuizSettings.objects.get_or_create(
        course=course,
        defaults={
            'is_adaptive': True,
            'min_difficulty': 1,
            'max_difficulty': 10,
            'difficulty_step': 0.5
        }
    )
    
    # Determine areas for improvement
    # Group question attempts by marks (difficulty) and calculate accuracy per difficulty
    difficulty_stats = {}
    for attempt in attempts:
        marks = attempt.question.marks
        if marks not in difficulty_stats:
            difficulty_stats[marks] = {'correct': 0, 'total': 0}
        
        difficulty_stats[marks]['total'] += 1
        if attempt.is_correct:
            difficulty_stats[marks]['correct'] += 1
    
    # Find weakest difficulty level
    weakest_difficulty = None
    lowest_accuracy = 100
    
    for difficulty, stats in difficulty_stats.items():
        if stats['total'] >= 3:  # Only consider difficulties with at least 3 attempts
            difficulty_accuracy = (stats['correct'] / stats['total']) * 100
            if difficulty_accuracy < lowest_accuracy:
                lowest_accuracy = difficulty_accuracy
                weakest_difficulty = difficulty
    
    # Generate recommendations
    recommendations = []
    
    if weakest_difficulty is not None:
        recommendations.append(f"Focus on difficulty level {weakest_difficulty} questions")
    
    if accuracy < 60:
        recommendations.append("Review course materials before attempting more quizzes")
    elif accuracy > 90 and skill_level.current_level < settings.max_difficulty:
        recommendations.append("Consider increasing difficulty level for more challenge")
    
    if avg_time > 60:  # If average time per question is over 60 seconds
        recommendations.append("Work on improving your speed in answering questions")
    
    # Add general recommendation if none specific
    if not recommendations:
        recommendations.append("Continue practicing to maintain your skill level")
    
    return {
        'total_questions': total_questions,
        'accuracy': round(accuracy, 1),
        'average_time': round(avg_time, 1),
        'skill_level': round(skill_level.current_level, 1),
        'confidence': round(skill_level.confidence * 100, 1),
        'recommendations': recommendations
    } 