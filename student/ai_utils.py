import requests
from django.conf import settings
import json
from datetime import datetime
from django.utils import timezone
from .models import AIChatHistory, AIUsageAnalytics
import logging
import random
from openai import OpenAI, OpenAIError
from decouple import config

logger = logging.getLogger(__name__)

# Get DeepInfra API configuration from settings
OPENROUTER_API_KEY = config('OPENROUTER_API_KEY')
OPENROUTER_MODEL_NAME = config('OPENROUTER_MODEL_NAME')

# Initialize OpenAI client with DeepInfra configuration
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1/chat/completions",
    default_headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
)

# Mock responses for different subjects
MOCK_RESPONSES = {
    'math': [
        "Let me help you with that math problem. First, let's break it down step by step...",
        "For this mathematical concept, you'll want to remember these key points...",
        "Here's a helpful way to approach this math question..."
    ],
    'science': [
        "In science, this concept is important because...",
        "Let me explain the scientific principles behind this...",
        "Here's how this scientific concept works..."
    ],
    'history': [
        "This historical event is significant because...",
        "Let me explain the historical context of this...",
        "Here's what you need to know about this historical period..."
    ],
    'english': [
        "For this English concept, consider these points...",
        "Let me help you understand this literary concept...",
        "Here's how to approach this English topic..."
    ],
    'physics': [
        "In physics, this principle is fundamental because...",
        "Let me explain the physics behind this...",
        "Here's how this physics concept works..."
    ],
    'chemistry': [
        "In chemistry, this reaction occurs because...",
        "Let me explain the chemical principles involved...",
        "Here's how this chemical process works..."
    ],
    'biology': [
        "In biology, this process is important because...",
        "Let me explain the biological principles...",
        "Here's how this biological system works..."
    ],
    'default': [
        "Let me help you understand this topic better...",
        "Here's what you need to know about this...",
        "Let me explain this concept in detail..."
    ]
}

def get_mock_response(question, course=None):
    """Generate a contextual mock response based on the question and course."""
    # Extract topics from the question
    topics = extract_topics(question)
    
    # If we have a course, add its name to topics
    if course:
        course_name = course.course_name.lower()
        topics.append(course_name)
    
    # Find matching responses
    responses = []
    for topic in topics:
        if topic in MOCK_RESPONSES:
            responses.extend(MOCK_RESPONSES[topic])
    
    # If no specific responses found, use default
    if not responses:
        responses = MOCK_RESPONSES['default']
    
    # Add some contextual information
    context = f"Regarding your question about '{question}'"
    if course:
        context += f" in {course.course_name}"
    
    return f"{context}. {random.choice(responses)}"

def get_ai_response(question, student, course=None):
    # If no API key is configured, use mock responses
    if not OPENROUTER_API_KEY:
        logger.info("No Deepseek API key configured, using mock responses")
        response = get_mock_response(question, course)
    else:
        system_message = """You are a helpful AI tutor assisting students with their quiz preparation. 
        Provide clear, concise, and accurate answers. 
        Format your responses in markdown for better readability.
        Ensure your responses are complete and not cut off.
        If you're explaining code, use proper code blocks with language specification."""
        
        if course:
            system_message += f" The student is asking about the course: {course.course_name}"
        
        try:
            start_time = timezone.now()
            logger.info(f"Sending request to OPENROUTER API for student {student.get_name}")
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'http://localhost:3000'
            }
            payload = {
                "model": OPENROUTER_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.95,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response_data = response.json()
            
            if response.status_code == 200:
                response_text = response_data['choices'][0]['message']['content']
            else:
                logger.error(f"API error: {response_data}")
                response_text = get_mock_response(question, course)
            
            end_time = timezone.now()
            
            # Log usage statistics if available
            if 'usage' in response_data:
                usage = response_data['usage']
                logger.info(f"Token usage - Prompt: {usage.get('prompt_tokens', 0)}, "
                            f"Completion: {usage.get('completion_tokens', 0)}, "
                            f"Total: {usage.get('total_tokens', 0)}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while calling OpenRouter API: {str(e)}")
            response_text = get_mock_response(question, course)
        except Exception as e:
            logger.error(f"Unexpected error in get_ai_response: {str(e)}")
            response_text = get_mock_response(question, course)
    
    # Save chat history and update analytics regardless of response source
    try:
        chat_history = AIChatHistory.objects.create(
            student=student,
            question=question,
            answer=response_text,
            course=course
        )
        
        analytics, created = AIUsageAnalytics.objects.get_or_create(
            student=student,
            date=timezone.now().date()
        )
        
        analytics.total_queries += 1
        analytics.average_response_time = (
            (analytics.average_response_time * (analytics.total_queries - 1) + 1) 
            / analytics.total_queries
        )
        
        topics = extract_topics(question)
        analytics.topics_covered = list(set(analytics.topics_covered + topics))
        analytics.save()
        
        student.ai_usage_count += 1
        student.last_ai_interaction = timezone.now()
        student.save()
        
        logger.info(f"Successfully processed request for student {student.get_name}")
    except Exception as e:
        logger.error(f"Error saving chat history or analytics: {str(e)}")
    
    return response_text

def extract_topics(question):
    # Enhanced topic extraction
    topics = []
    common_subjects = ['math', 'science', 'history', 'english', 'physics', 'chemistry', 'biology']
    question_lower = question.lower()
    
    # Check for subject mentions
    for subject in common_subjects:
        if subject in question_lower:
            topics.append(subject)
    
    # Check for question types
    if any(word in question_lower for word in ['how', 'what', 'why', 'when', 'where']):
        topics.append('question')
    
    # Check for difficulty level
    if any(word in question_lower for word in ['hard', 'difficult', 'easy', 'simple']):
        topics.append('difficulty')
    
    return topics

def get_student_ai_insights(student):
    try:
        analytics = AIUsageAnalytics.objects.filter(student=student).order_by('-date')[:7]
        recent_chats = AIChatHistory.objects.filter(student=student).order_by('-timestamp')[:5]
        
        insights = {
            'total_queries': sum(a.total_queries for a in analytics),
            'average_response_time': sum(a.average_response_time for a in analytics) / len(analytics) if analytics else 0,
            'top_topics': get_top_topics(student),
            'recent_chats': recent_chats
        }
        
        return insights
    except Exception as e:
        logger.error(f"Error getting AI insights for student {student.get_name}: {str(e)}")
        return {
            'total_queries': 0,
            'average_response_time': 0,
            'top_topics': [],
            'recent_chats': []
        }

def get_top_topics(student):
    """Get the most frequent topics a student has asked about"""
    try:
        analytics = AIUsageAnalytics.objects.filter(student=student).order_by('-date').first()
        if analytics and analytics.topics_covered:
            return analytics.topics_covered[:5]
    except Exception as e:
        logger.error(f"Error getting top topics: {str(e)}")
    return []

def get_ai_explanation(question):
    """Generate an AI explanation for a quiz question"""
    # First, check if we already have an explanation for this question
    from quiz.models import QuestionExplanation
    existing_explanation = QuestionExplanation.objects.filter(question=question).first()
    
    if existing_explanation:
        logger.info(f"Using existing explanation for question {question.id}")
        return existing_explanation.explanation_text
    
    # If no explanation exists, generate one
    if not OPENROUTER_API_KEY:
        logger.info("No OpenRouter API key configured, using mock explanation")
        return generate_mock_explanation(question)
    
    system_message = f"""You are an expert educational assistant generating detailed explanations for quiz questions.
    Provide a clear, informative explanation for why the correct answer is right and why the other options are wrong.
    Include relevant concepts, principles, or context that helps understand the question better.
    Format your explanation using markdown for better readability."""
    
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost:3000'
        }
        
        question_text = f"""
        Question: {question.question}
        Option 1: {question.option1}
        Option 2: {question.option2}
        Option 3: {question.option3}
        Option 4: {question.option4}
        Correct answer: {question.answer}
        """
        
        payload = {
            "model": OPENROUTER_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate an explanation for this quiz question: {question_text}"}
            ],
            "temperature": 0.5,
            "max_tokens": 1000,
            "top_p": 0.95
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if response.status_code == 200:
            explanation_text = response_data['choices'][0]['message']['content']
            
            # Save the explanation for future use
            from quiz.models import QuestionExplanation
            QuestionExplanation.objects.create(
                question=question,
                explanation_text=explanation_text
            )
            
            return explanation_text
        else:
            logger.error(f"API error: {response_data}")
            return generate_mock_explanation(question)
            
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return generate_mock_explanation(question)

def generate_mock_explanation(question):
    """Generate a mock explanation when API is not available"""
    correct_option_text = getattr(question, question.answer.lower())
    
    explanations = [
        f"The correct answer is {question.answer}: '{correct_option_text}'. This is because the question is asking about {question.question}, and {correct_option_text} most accurately addresses this.",
        f"The answer {question.answer} ('{correct_option_text}') is correct because it aligns with the core principles related to the question.",
        f"When considering '{question.question}', the best answer is {question.answer}: '{correct_option_text}'. The other options contain information that is either incorrect or not fully relevant to the question."
    ]
    
    return random.choice(explanations)

def get_ai_feedback(question_attempt):
    """Generate AI feedback for a student's question attempt"""
    question = question_attempt.question
    is_correct = question_attempt.is_correct
    selected_answer = question_attempt.answer_selected
    
    # First, check if we already have feedback for this attempt
    from quiz.models import StudentFeedback
    existing_feedback = StudentFeedback.objects.filter(question_attempt=question_attempt).first()
    
    if existing_feedback:
        logger.info(f"Using existing feedback for question attempt {question_attempt.id}")
        return existing_feedback.feedback_text
    
    # If no feedback exists, generate it
    if not OPENROUTER_API_KEY:
        logger.info("No OpenRouter API key configured, using mock feedback")
        return generate_mock_feedback(question_attempt)
    
    # Get the text of the selected answer option
    selected_option_text = getattr(question, selected_answer.lower(), "Unknown")
    correct_option_text = getattr(question, question.answer.lower(), "Unknown")
    
    system_message = f"""You are an expert educational assistant providing helpful feedback to students on their quiz answers.
    Provide encouraging, constructive feedback that helps the student understand why their answer was {'correct' if is_correct else 'incorrect'}.
    If the answer was incorrect, explain why and guide them toward understanding the correct answer without explicitly telling them.
    Format your feedback using markdown for better readability."""
    
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost:3000'
        }
        
        question_text = f"""
        Question: {question.question}
        Student's answer: {selected_answer} - {selected_option_text}
        This answer is {'correct' if is_correct else 'incorrect'}.
        """
        
        if not is_correct:
            question_text += f"\nThe correct answer is: {question.answer} - {correct_option_text}"
        
        payload = {
            "model": OPENROUTER_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Provide feedback on this quiz answer: {question_text}"}
            ],
            "temperature": 0.7,
            "max_tokens": 800,
            "top_p": 0.95
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if response.status_code == 200:
            feedback_text = response_data['choices'][0]['message']['content']
            
            # Save the feedback for future reference
            from quiz.models import StudentFeedback
            StudentFeedback.objects.create(
                question_attempt=question_attempt,
                feedback_text=feedback_text,
                is_correct=is_correct
            )
            
            return feedback_text
        else:
            logger.error(f"API error: {response_data}")
            return generate_mock_feedback(question_attempt)
            
    except Exception as e:
        logger.error(f"Error generating feedback: {str(e)}")
        return generate_mock_feedback(question_attempt)

def generate_mock_feedback(question_attempt):
    """Generate mock feedback when API is not available"""
    question = question_attempt.question
    is_correct = question_attempt.is_correct
    selected_answer = question_attempt.answer_selected
    
    if is_correct:
        feedbacks = [
            f"Great job! Your answer '{selected_answer}' is correct. You've demonstrated a good understanding of this concept.",
            f"Excellent! You chose the right answer. This shows you understand the key principles involved.",
            f"Well done! Your answer is correct. You've mastered this topic."
        ]
    else:
        correct_option = question.answer
        feedbacks = [
            f"Your answer '{selected_answer}' isn't quite right. Try reviewing the topic again and consider why another option might be more accurate.",
            f"That's not correct. Think about the key concepts we've covered and reconsider your approach to this question.",
            f"While '{selected_answer}' isn't correct, this is a good opportunity to revisit the material and strengthen your understanding."
        ]
    
    return random.choice(feedbacks)

def generate_content_recommendations(student, course=None):
    """Generate personalized content recommendations for a student"""
    from quiz.models import ContentRecommendation, QuestionAttempt, Result
    
    # Get student's recent question attempts
    question_attempts = QuestionAttempt.objects.filter(student=student).order_by('-timestamp')
    
    # Filter by course if provided
    if course:
        question_attempts = question_attempts.filter(question__course=course)
    
    # Get incorrect answers to identify struggle areas
    incorrect_attempts = question_attempts.filter(is_correct=False)
    
    # Get AI chat history for topics the student has asked about
    chat_history = AIChatHistory.objects.filter(student=student).order_by('-timestamp')
    if course:
        chat_history = chat_history.filter(course=course)
    
    # Extract topics from question attempts and chat history
    topics = set()
    
    # From incorrect question attempts
    for attempt in incorrect_attempts[:10]:  # Look at recent incorrect attempts
        question_text = attempt.question.question.lower()
        extracted = extract_topics(question_text)
        topics.update(extracted)
    
    # From chat history
    for chat in chat_history[:10]:  # Look at recent chats
        extracted = extract_topics(chat.question)
        topics.update(extracted)
    
    # If we have no topics, use some general educational topics
    if not topics:
        if course:
            # Use course name as base for topics
            course_name = course.course_name.lower()
            base_topics = extract_topics(course_name)
            topics.update(base_topics)
            topics.add(course_name)
        else:
            # Add some general topics
            topics.update(['learning', 'study', 'education'])
    
    # Generate recommendations based on topics
    if not OPENROUTER_API_KEY:
        logger.info("No OpenRouter API key configured, using mock recommendations")
        return generate_mock_recommendations(student, list(topics), course)
    
    system_message = """You are an expert educational assistant providing resource recommendations to students.
    Based on the student's learning history, generate 3-5 specific content recommendations.
    Each recommendation should include:
    1. A title
    2. A brief description of the resource
    3. The type of resource (course, article, video, exercise, book)
    4. A relevance score between 0.0 and 1.0 indicating how relevant this resource is to the student's needs
    
    Format your response as a JSON array of objects with this structure:
    [
        {
            "title": "Resource title",
            "description": "Brief description of what the resource covers and how it will help",
            "resource_type": "article",
            "relevance_score": 0.85,
            "url": "https://example.com/resource" (optional)
        },
        ...more recommendations...
    ]
    """
    
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost:3000'
        }
        
        context = f"Topics the student needs help with: {', '.join(topics)}"
        if course:
            context += f"\nCourse: {course.course_name}"
        
        payload = {
            "model": OPENROUTER_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate content recommendations based on this context: {context}"}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if response.status_code == 200:
            try:
                content = response_data['choices'][0]['message']['content']
                # Parse JSON response to get recommendations
                import json
                recommendations_data = json.loads(content)
                
                # Save recommendations to database
                for rec in recommendations_data:
                    ContentRecommendation.objects.create(
                        student=student,
                        course=course,
                        title=rec.get('title', 'Recommended Resource'),
                        description=rec.get('description', ''),
                        resource_type=rec.get('resource_type', 'other'),
                        url=rec.get('url'),
                        relevance_score=float(rec.get('relevance_score', 0.5))
                    )
                
                return recommendations_data
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing API response: {str(e)}")
                return generate_mock_recommendations(student, list(topics), course)
        else:
            logger.error(f"API error: {response_data}")
            return generate_mock_recommendations(student, list(topics), course)
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return generate_mock_recommendations(student, list(topics), course)

def generate_mock_recommendations(student, topics, course=None):
    """Generate mock content recommendations when API is not available"""
    from quiz.models import ContentRecommendation
    
    recommendations = []
    
    resource_types = ['article', 'video', 'exercise', 'book', 'course']
    
    # Use topics to create relevant recommendations
    for i, topic in enumerate(topics[:5]):  # Use up to 5 topics
        title = f"Understanding {topic.title()}"
        description = f"This resource helps you master the concept of {topic} with clear explanations and examples."
        resource_type = resource_types[i % len(resource_types)]
        relevance_score = round(0.5 + (0.1 * i), 2)  # Scores from 0.5 to 0.9
        
        rec = {
            "title": title,
            "description": description,
            "resource_type": resource_type,
            "relevance_score": relevance_score
        }
        
        if resource_type == 'video':
            rec["url"] = f"https://example.com/video/{topic.replace(' ', '-')}"
        elif resource_type == 'article':
            rec["url"] = f"https://example.com/article/{topic.replace(' ', '-')}"
        
        recommendations.append(rec)
        
        # Save to database
        ContentRecommendation.objects.create(
            student=student,
            course=course,
            title=title,
            description=description,
            resource_type=resource_type,
            url=rec.get("url"),
            relevance_score=relevance_score
        )
    
    return recommendations

def calculate_student_skill_level(student, course, attempt=None):
    """Calculate and update a student's skill level for adaptive quizzing"""
    from quiz.models import StudentSkillLevel, QuestionAttempt
    
    # Get or create the student's skill level for this course
    skill_level, created = StudentSkillLevel.objects.get_or_create(
        student=student,
        course=course,
        defaults={
            'current_level': 5.0,  # Start at medium difficulty
            'confidence': 0.5      # Medium confidence
        }
    )
    
    # If no attempt provided, just return the current level
    if not attempt:
        return skill_level.current_level
    
    # Update level based on the new attempt
    is_correct = attempt.is_correct
    confidence = skill_level.confidence
    
    # Adjust level: increase if correct, decrease if incorrect
    # The adjustment is weighted by our confidence
    adjustment = 0.5 * confidence
    
    if is_correct:
        # Correct answer increases difficulty level
        new_level = min(10.0, skill_level.current_level + adjustment)
        # Increase confidence slightly
        new_confidence = min(1.0, confidence + 0.05)
    else:
        # Incorrect answer decreases difficulty level
        new_level = max(1.0, skill_level.current_level - adjustment)
        # Decrease confidence slightly
        new_confidence = max(0.1, confidence - 0.05)
    
    # Update the skill level
    skill_level.current_level = new_level
    skill_level.confidence = new_confidence
    skill_level.save()
    
    return new_level