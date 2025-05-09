# Online Examination System - Sequence Diagrams

## Student Taking an Exam Sequence

```mermaid
sequenceDiagram
    actor Student
    participant AuthSystem
    participant Dashboard
    participant CourseManager
    participant ExamEngine
    participant AdaptiveSystem
    participant ResultProcessor
    participant AIExplainer
    
    Student->>AuthSystem: Login credentials
    AuthSystem->>Dashboard: Authentication token
    Dashboard->>Student: Show available courses
    
    Student->>CourseManager: Select course
    CourseManager->>Student: Display available exams
    Student->>ExamEngine: Start exam
    
    ExamEngine->>ExamEngine: Check if adaptive
    
    alt Adaptive Exam
        ExamEngine->>AdaptiveSystem: Initialize adaptive settings
        AdaptiveSystem->>ExamEngine: Initial question set (medium difficulty)
    else Standard Exam
        ExamEngine->>ExamEngine: Load all questions
    end
    
    loop For each question
        ExamEngine->>Student: Present question
        Student->>ExamEngine: Submit answer
        
        alt Adaptive Exam
            ExamEngine->>AdaptiveSystem: Process answer
            AdaptiveSystem->>ExamEngine: Adjust difficulty
            AdaptiveSystem->>ExamEngine: Next question based on performance
        end
    end
    
    Student->>ExamEngine: Submit exam
    ExamEngine->>ResultProcessor: Process answers
    ResultProcessor->>ExamEngine: Calculated results
    ExamEngine->>Student: Display results
    
    opt Request Explanation
        Student->>AIExplainer: Request explanation for specific questions
        AIExplainer->>AIExplainer: Generate personalized explanation
        AIExplainer->>Student: Provide detailed explanation with resources
    end
```

## Teacher Creating AI-Generated Exam Sequence

```mermaid
sequenceDiagram
    actor Teacher
    participant AuthSystem
    participant Dashboard
    participant CourseManager
    participant AIExamGenerator
    participant ExamEditor
    participant NotificationSystem
    
    Teacher->>AuthSystem: Login credentials
    AuthSystem->>Dashboard: Authentication token
    Dashboard->>Teacher: Show teaching courses
    
    Teacher->>CourseManager: Select course to manage
    CourseManager->>Teacher: Display course options
    
    Teacher->>AIExamGenerator: Request AI exam generation
    Teacher->>AIExamGenerator: Specify parameters (topic, difficulty, etc.)
    AIExamGenerator->>AIExamGenerator: Generate questions
    AIExamGenerator->>Teacher: Present generated questions
    
    loop Review and Edit
        Teacher->>ExamEditor: Modify questions if needed
        ExamEditor->>Teacher: Updated questions
    end
    
    Teacher->>ExamEditor: Set exam settings (time, security, adaptive options)
    ExamEditor->>CourseManager: Save exam configuration
    CourseManager->>Teacher: Confirm exam creation
    
    opt Notify Students
        Teacher->>NotificationSystem: Request student notification
        NotificationSystem->>NotificationSystem: Prepare notifications
        NotificationSystem-->>Student: Send exam availability notice
    end
```

## AI Tutoring Interaction Sequence

```mermaid
sequenceDiagram
    actor Student
    participant AuthSystem
    participant Dashboard
    participant AITutor
    participant KnowledgeBase
    participant QuestionGenerator
    participant AnalyticsSystem
    
    Student->>AuthSystem: Login credentials
    AuthSystem->>Dashboard: Authentication token
    Dashboard->>Student: Show dashboard options
    
    Student->>AITutor: Access AI tutor
    Student->>AITutor: Ask study question
    
    AITutor->>KnowledgeBase: Query relevant information
    KnowledgeBase->>AITutor: Return knowledge context
    
    AITutor->>AITutor: Process question with context
    AITutor->>Student: Provide answer
    AITutor->>Student: Suggest related topics
    
    alt Request Practice
        Student->>AITutor: Request practice questions
        AITutor->>QuestionGenerator: Generate questions on topic
        QuestionGenerator->>AITutor: Return practice questions
        AITutor->>Student: Present practice questions
        
        loop Practice Session
            Student->>AITutor: Submit practice answers
            AITutor->>AITutor: Evaluate answers
            AITutor->>Student: Provide feedback
        end
    end
    
    Student->>AITutor: End session
    AITutor->>AnalyticsSystem: Log interaction data
    AnalyticsSystem->>AnalyticsSystem: Update student profile
    AnalyticsSystem->>AnalyticsSystem: Identify learning patterns
```

## Adaptive Exam Assessment Sequence

```mermaid
sequenceDiagram
    actor Student
    participant ExamEngine
    participant AdaptiveSystem
    participant QuestionBank
    participant DifficultyCalculator
    participant ResultAnalyzer
    participant RecommendationEngine
    
    Student->>ExamEngine: Start adaptive exam
    ExamEngine->>AdaptiveSystem: Initialize adaptive session
    AdaptiveSystem->>QuestionBank: Request medium difficulty question
    QuestionBank->>AdaptiveSystem: Return initial question
    AdaptiveSystem->>ExamEngine: Deliver question
    ExamEngine->>Student: Present question
    
    loop Adaptive Question Flow
        Student->>ExamEngine: Submit answer
        ExamEngine->>AdaptiveSystem: Pass answer for evaluation
        AdaptiveSystem->>DifficultyCalculator: Calculate next difficulty level
        
        alt Correct Answer
            DifficultyCalculator->>AdaptiveSystem: Increase difficulty
        else Incorrect Answer
            DifficultyCalculator->>AdaptiveSystem: Decrease difficulty
        end
        
        AdaptiveSystem->>QuestionBank: Request question at new difficulty
        QuestionBank->>AdaptiveSystem: Return appropriate question
        AdaptiveSystem->>ExamEngine: Deliver next question
        ExamEngine->>Student: Present question
    end
    
    Student->>ExamEngine: Complete exam
    ExamEngine->>ResultAnalyzer: Process adaptive exam data
    ResultAnalyzer->>ResultAnalyzer: Generate proficiency profile
    ResultAnalyzer->>RecommendationEngine: Send performance data
    RecommendationEngine->>RecommendationEngine: Generate personalized recommendations
    
    ResultAnalyzer->>Student: Display performance analysis
    RecommendationEngine->>Student: Provide study recommendations
```

## Sequence Diagram Description

These sequence diagrams illustrate the detailed interactions between actors (users) and system components in the Online Examination System:

### Student Taking an Exam Sequence
Shows the complete flow of a student taking an exam, highlighting:
- Authentication and course/exam selection process
- Differences between standard and adaptive exam flows
- Question presentation and answer submission
- Results processing and optional AI explanations

### Teacher Creating AI-Generated Exam Sequence
Demonstrates how teachers use AI to streamline exam creation:
- Course selection and management options
- AI-based question generation with parameter specification
- Review and modification of AI-generated content
- Exam configuration and student notification

### AI Tutoring Interaction Sequence
Illustrates the student-AI interaction for learning support:
- Access to AI tutor from dashboard
- Question processing using knowledge base
- Related topic suggestions
- Practice question generation and feedback
- Analytics tracking for personalized learning

### Adaptive Exam Assessment Sequence
Details the sophisticated adaptive testing mechanism:
- Initialization with medium difficulty questions
- Dynamic difficulty adjustment based on performance
- Question selection from appropriate difficulty levels
- Comprehensive result analysis and recommendation generation

These sequence diagrams complement the activity diagrams by showing the specific system components involved in each process and the messages passed between them, providing a detailed technical view of the system interactions. 