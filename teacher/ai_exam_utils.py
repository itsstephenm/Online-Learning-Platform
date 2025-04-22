import requests # type: ignore
from django.conf import settings # type: ignore
from django.utils import timezone # type: ignore
import logging
import random
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

# Get OpenRouter API configuration from settings
try:
    from decouple import config
except ImportError:
    # Fallback function if decouple is not available
    def config(key, default=None, cast=None):
        value = os.environ.get(key, default)
        if cast and value is not None and cast != bool:
            value = cast(value)
        elif cast == bool and isinstance(value, str):
            value = value.lower() in ('true', 'yes', '1')
        return value

OPENROUTER_API_KEY = config('OPENROUTER_API_KEY')
OPENROUTER_MODEL_NAME = config('OPENROUTER_MODEL_NAME')

def get_ai_exam_questions(course, difficulty, num_questions=10, reference_text=None):
    """
    Generate exam questions using AI based on the course and difficulty.
    
    Args:
        course: Course object
        difficulty: String indicating difficulty level (easy, medium, hard)
        num_questions: Number of questions to generate
        reference_text: Text from uploaded material to use as context
        
    Returns:
        List of generated question dictionaries
    """
    # If no API key is configured, use mock exam questions
    if not OPENROUTER_API_KEY:
        logger.info("No OpenRouter API key configured, using mock exam questions")
        return get_mock_exam_questions(course, difficulty, num_questions)
    
    system_message = f"""You are an AI assistant that generates multiple-choice questions for educational exams.
    Generate {num_questions} multiple-choice questions for a {difficulty} level exam on {course.course_name}.
    For each question, provide:
    1. The question text
    2. Four answer options labeled as option1, option2, option3, and option4
    3. The correct answer (indicated as Option1, Option2, Option3, or Option4)
    4. The number of marks for the question (between 1-5 based on difficulty)
    
    Format your response as a JSON array of objects with the following structure:
    [
        {{
            "question": "Question text here",
            "option1": "First option",
            "option2": "Second option",
            "option3": "Third option",
            "option4": "Fourth option",
            "answer": "Option1",
            "marks": 2
        }},
        // More questions...
    ]
    """
    
    if reference_text:
        system_message += f"\n\nBase your questions on this reference material: {reference_text}"
    
    try:
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
                {"role": "user", "content": f"Generate {num_questions} multiple-choice questions for a {difficulty} level exam on {course.course_name}"}
            ],
            "temperature": 0.7,
            "max_tokens": 4000,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if response.status_code == 200:
            try:
                content = response_data['choices'][0]['message']['content']
                # Parse JSON response to get questions
                import json
                questions_data = json.loads(content)
                return questions_data
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing API response: {str(e)}")
                return get_mock_exam_questions(course, difficulty, num_questions)
        else:
            logger.error(f"API error: {response_data}")
            return get_mock_exam_questions(course, difficulty, num_questions)
            
    except Exception as e:
        logger.error(f"Error in get_ai_exam_questions: {str(e)}")
        return get_mock_exam_questions(course, difficulty, num_questions)

def get_mock_exam_questions(course, difficulty, num_questions=10):
    """Generate mock exam questions when API is not available"""
    questions = []
    difficulty_marks = {
        'easy': 1,
        'medium': 2,
        'hard': 3
    }
    marks = difficulty_marks.get(difficulty, 2)
    
    # Generate mock questions based on course name
    course_name = course.course_name.lower()
    
    for i in range(num_questions):
        if "math" in course_name:
            questions.append(generate_math_question(i, marks))
        elif "science" in course_name or "physics" in course_name or "chemistry" in course_name:
            questions.append(generate_science_question(i, marks))
        elif "history" in course_name:
            questions.append(generate_history_question(i, marks))
        elif "english" in course_name or "literature" in course_name:
            questions.append(generate_english_question(i, marks))
        else:
            questions.append(generate_general_question(i, course.course_name, marks))
    
    return questions

def generate_math_question(index, marks):
    questions = [
        {
            "question": "What is the value of x in the equation 2x + 5 = 15?",
            "option1": "5",
            "option2": "10",
            "option3": "7",
            "option4": "4",
            "answer": "Option1",
            "marks": marks
        },
        {
            "question": "If a triangle has angles measuring 30°, 60°, and x°, what is the value of x?",
            "option1": "90°",
            "option2": "180°",
            "option3": "60°",
            "option4": "45°",
            "answer": "Option1",
            "marks": marks
        },
        {
            "question": "What is the area of a circle with radius 4 units?",
            "option1": "16π square units",
            "option2": "8π square units",
            "option3": "4π square units",
            "option4": "64π square units",
            "answer": "Option1",
            "marks": marks
        }
    ]
    return questions[index % len(questions)]

def generate_science_question(index, marks):
    questions = [
        {
            "question": "Which of the following is NOT a state of matter?",
            "option1": "Energy",
            "option2": "Solid",
            "option3": "Liquid",
            "option4": "Gas",
            "answer": "Option1",
            "marks": marks
        },
        {
            "question": "What is the chemical symbol for gold?",
            "option1": "Au",
            "option2": "Ag",
            "option3": "Fe",
            "option4": "Gd",
            "answer": "Option1",
            "marks": marks
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "option1": "Mars",
            "option2": "Venus",
            "option3": "Jupiter",
            "option4": "Mercury",
            "answer": "Option1",
            "marks": marks
        }
    ]
    return questions[index % len(questions)]

def generate_history_question(index, marks):
    questions = [
        {
            "question": "In which year did World War II end?",
            "option1": "1945",
            "option2": "1939",
            "option3": "1941",
            "option4": "1950",
            "answer": "Option1",
            "marks": marks
        },
        {
            "question": "Who was the first President of the United States?",
            "option1": "George Washington",
            "option2": "Thomas Jefferson",
            "option3": "Abraham Lincoln",
            "option4": "John Adams",
            "answer": "Option1",
            "marks": marks
        },
        {
            "question": "The Industrial Revolution began in which country?",
            "option1": "Great Britain",
            "option2": "France",
            "option3": "Germany",
            "option4": "United States",
            "answer": "Option1",
            "marks": marks
        }
    ]
    return questions[index % len(questions)]

def generate_english_question(index, marks):
    questions = [
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "option1": "William Shakespeare",
            "option2": "Charles Dickens",
            "option3": "Jane Austen",
            "option4": "Mark Twain",
            "answer": "Option1",
            "marks": marks
        },
        {
            "question": "Which of these is NOT a type of literary device?",
            "option1": "Chromosome",
            "option2": "Metaphor",
            "option3": "Simile",
            "option4": "Alliteration",
            "answer": "Option1",
            "marks": marks
        },
        {
            "question": "What is the main function of a preposition in a sentence?",
            "option1": "To show the relationship between a noun and other words",
            "option2": "To describe an action",
            "option3": "To replace a noun",
            "option4": "To modify a verb",
            "answer": "Option1",
            "marks": marks
        }
    ]
    return questions[index % len(questions)]

def generate_general_question(index, course_name, marks):
    questions = [
        {
            "question": f"Which of the following is most relevant to {course_name}?",
            "option1": f"Core principles of {course_name}",
            "option2": "Unrelated concept",
            "option3": "Tangentially related idea",
            "option4": "None of the above",
            "answer": "Option1",
            "marks": marks
        },
        {
            "question": f"Who is considered the founder of modern {course_name}?",
            "option1": f"The pioneering researcher in {course_name}",
            "option2": "A politician",
            "option3": "A celebrity",
            "option4": "A fictional character",
            "answer": "Option1",
            "marks": marks
        },
        {
            "question": f"What is the primary focus of {course_name}?",
            "option1": f"The main subject matter of {course_name}",
            "option2": "An unrelated topic",
            "option3": "A broad general concept",
            "option4": "None of the above",
            "answer": "Option1",
            "marks": marks
        }
    ]
    return questions[index % len(questions)]

def save_ai_generated_exam(course, title, difficulty, questions, description=None, time_limit=60):
    """
    Save an AI-generated exam and its questions to the database.
    
    Args:
        course: Course object
        title: Exam title
        difficulty: Difficulty level
        questions: List of question dictionaries
        description: Optional exam description
        time_limit: Time limit in minutes
        
    Returns:
        The created AIGeneratedExam object
    """
    try:
        # Create AI generated exam record
        ai_exam = AIGeneratedExam.objects.create(
            course=course,
            title=title,
            description=description,
            difficulty=difficulty,
            time_limit=time_limit
        )
        
        # Add questions to the course
        for q_data in questions:
            Question.objects.create(
                course=course,
                marks=q_data.get('marks', 1),
                question=q_data.get('question', ''),
                option1=q_data.get('option1', ''),
                option2=q_data.get('option2', ''),
                option3=q_data.get('option3', ''),
                option4=q_data.get('option4', ''),
                answer=q_data.get('answer', 'Option1')
            )
            
        return ai_exam
    except Exception as e:
        logger.error(f"Error saving AI generated exam: {str(e)}")
        raise

def process_reference_document(document):
    """
    Extract text from an uploaded reference document.
    
    Args:
        document: ReferenceDocument object
        
    Returns:
        Extracted text content
    """
    from quiz.models import ReferenceDocument
    
    if document.extracted_text:
        logger.info(f"Using cached extracted text for document {document.id}")
        return document.extracted_text
    
    try:
        file_path = document.document_file.path
        file_type = document.file_type
        
        if file_type == 'pdf':
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
        elif file_type == 'docx':
            import docx2txt
            text = docx2txt.process(file_path)
        elif file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            logger.error(f"Unsupported file type: {file_type}")
            text = ""
        
        # Update the document with extracted text
        document.extracted_text = text
        document.save()
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from document: {str(e)}")
        return ""

def get_ai_exam_questions_from_references(course, difficulty, reference_documents, num_questions=10):
    """
    Generate exam questions using AI based on reference documents.
    
    Args:
        course: Course object
        difficulty: String indicating difficulty level (easy, medium, hard)
        reference_documents: List of ReferenceDocument objects
        num_questions: Number of questions to generate
        
    Returns:
        List of generated question dictionaries
    """
    # If no API key is configured, use mock exam questions
    if not OPENROUTER_API_KEY:
        logger.info("No OpenRouter API key configured, using mock exam questions")
        return get_mock_exam_questions(course, difficulty, num_questions)
    
    # Extract text from reference documents
    reference_texts = []
    for document in reference_documents:
        text = process_reference_document(document)
        if text:
            # Truncate very long documents if needed
            if len(text) > 10000:
                text = text[:10000] + "...[content truncated]..."
            reference_texts.append(text)
    
    # If no reference texts were successfully extracted, fall back to standard generation
    if not reference_texts:
        logger.warning("No reference texts extracted, falling back to standard question generation")
        return get_ai_exam_questions(course, difficulty, num_questions)
    
    combined_text = "\n\n===DOCUMENT SEPARATOR===\n\n".join(reference_texts)
    
    system_message = f"""You are an expert educational assistant generating multiple-choice questions for exams.
    Generate {num_questions} multiple-choice questions for a {difficulty} level exam on {course.course_name}.
    
    Base your questions ONLY on the reference material provided. Do not include concepts or information not covered in the reference material.
    
    For each question, provide:
    1. The question text
    2. Four answer options labeled as option1, option2, option3, and option4
    3. The correct answer (indicated as Option1, Option2, Option3, or Option4)
    4. The number of marks for the question (between 1-5 based on difficulty)
    
    Format your response as a JSON array of objects with the following structure:
    [
        {{
            "question": "Question text here",
            "option1": "First option",
            "option2": "Second option",
            "option3": "Third option",
            "option4": "Fourth option",
            "answer": "Option1",
            "marks": 2
        }},
        // More questions...
    ]
    """
    
    try:
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
                {"role": "user", "content": f"Reference material:\n\n{combined_text}\n\nGenerate {num_questions} multiple-choice questions for a {difficulty} level exam on {course.course_name} based on this material."}
            ],
            "temperature": 0.7,
            "max_tokens": 4000,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if response.status_code == 200:
            try:
                content = response_data['choices'][0]['message']['content']
                # Parse JSON response to get questions
                import json
                questions_data = json.loads(content)
                return questions_data
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing API response: {str(e)}")
                return get_mock_exam_questions(course, difficulty, num_questions)
        else:
            logger.error(f"API error: {response_data}")
            return get_mock_exam_questions(course, difficulty, num_questions)
            
    except Exception as e:
        logger.error(f"Error in get_ai_exam_questions_from_references: {str(e)}")
        return get_mock_exam_questions(course, difficulty, num_questions) 