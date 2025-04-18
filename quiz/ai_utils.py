import random
from datetime import datetime

def generate_questions(course, difficulty, num_questions, question_types):
    """
    Generate AI questions for a course
    
    Args:
        course: Course object
        difficulty: Difficulty level (easy, medium, hard)
        num_questions: Number of questions to generate
        question_types: List of question types to generate
        
    Returns:
        List of question dictionaries
    """
    questions = []
    
    # Set difficulty-based marks
    if difficulty == 'easy':
        marks = 1
    elif difficulty == 'medium':
        marks = 2
    else:  # hard
        marks = 3
        
    for i in range(num_questions):
        # Select a random question type from the provided types
        question_type = random.choice(question_types)
        
        # Create question data based on type
        if question_type == 'multiple_choice':
            q_data = {
                'course': course,
                'question_type': 'multiple_choice',
                'question': f"Sample {difficulty} MCQ question #{i+1} for {course.course_name}",
                'option1': f"Option A for question #{i+1}",
                'option2': f"Option B for question #{i+1}",
                'option3': f"Option C for question #{i+1}",
                'option4': f"Option D for question #{i+1}",
                'answer': f"Option A for question #{i+1}",  # Correct answer
                'marks': marks,
                'is_ai_generated': True
            }
            
        elif question_type == 'checkbox':
            q_data = {
                'course': course,
                'question_type': 'checkbox',
                'question': f"Sample {difficulty} checkbox question #{i+1} for {course.course_name}",
                'option1': f"Option A for question #{i+1}",
                'option2': f"Option B for question #{i+1}",
                'option3': f"Option C for question #{i+1}",
                'option4': f"Option D for question #{i+1}",
                'multiple_answers': [f"Option A for question #{i+1}", f"Option B for question #{i+1}"],
                'marks': marks,
                'is_ai_generated': True
            }
            
        else:  # short_answer
            q_data = {
                'course': course,
                'question_type': 'short_answer',
                'question': f"Sample {difficulty} short answer question #{i+1} for {course.course_name}",
                'short_answer_pattern': f"answer|sample|response",
                'marks': marks,
                'is_ai_generated': True
            }
            
        questions.append(q_data)
        
    return questions 