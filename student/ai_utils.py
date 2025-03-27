import requests
from django.conf import settings
import json
from datetime import datetime
from django.utils import timezone
from .models import AIChatHistory, AIUsageAnalytics
import logging
import random
from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

# Get DeepInfra API configuration from settings
DEEPINFRA_API_KEY = getattr(settings, 'DEEPSEEK_API_KEY', None)
MODEL_NAME = getattr(settings, 'DEEPSEEK_MODEL_NAME', None)

# Initialize OpenAI client with DeepInfra configuration
client = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
    default_headers={
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}"
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
    if not DEEPINFRA_API_KEY:
        logger.info("No DeepInfra API key configured, using mock responses")
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
            logger.info(f"Sending request to DeepInfra API for student {student.get_name}")
            
            chat_completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=2000,  # Increased for longer responses
                top_p=0.95,
                presence_penalty=0.1,
                frequency_penalty=0.1,
                stop=None  # Ensure no premature stopping
            )
            
            end_time = timezone.now()
            response = chat_completion.choices[0].message.content
            
            # Ensure response is complete
            if response.endswith('...') or response.endswith('---'):
                logger.warning("Response appears to be incomplete, requesting completion")
                # Request completion of the response
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Complete the previous response without repeating what was already said."},
                        {"role": "user", "content": f"Complete this response: {response}"}
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    top_p=0.95
                )
                response = completion.choices[0].message.content
            
            # Log usage statistics
            logger.info(f"Token usage - Prompt: {chat_completion.usage.prompt_tokens}, "
                       f"Completion: {chat_completion.usage.completion_tokens}, "
                       f"Total: {chat_completion.usage.total_tokens}")
                
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            response = get_mock_response(question, course)
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while calling DeepInfra API: {str(e)}")
            response = get_mock_response(question, course)
        except Exception as e:
            logger.error(f"Unexpected error in get_ai_response: {str(e)}")
            response = get_mock_response(question, course)
    
    # Save chat history and update analytics regardless of response source
    try:
        chat_history = AIChatHistory.objects.create(
            student=student,
            question=question,
            answer=response,
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
    
    return response

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
    try:
        analytics = AIUsageAnalytics.objects.filter(student=student)
        topic_frequency = {}
        
        for analytic in analytics:
            for topic in analytic.topics_covered:
                topic_frequency[topic] = topic_frequency.get(topic, 0) + 1
        
        return sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
    except Exception as e:
        logger.error(f"Error getting top topics for student {student.get_name}: {str(e)}")
        return [] 