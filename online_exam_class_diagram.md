# Online Examination System - Class Diagram

```mermaid
classDiagram
    %% User Management Classes
    User <|-- Student
    User <|-- Teacher
    User <|-- Admin
    Teacher "many" -- "1" Department
    
    %% Course Management Classes
    Course "1" *-- "many" Question
    Course "1" o-- "0..1" AdaptiveQuizSettings
    
    %% Exam and Assessment Classes
    Student "1" -- "many" Result
    Course "1" -- "many" Result
    Question "1" -- "many" QuestionAttempt
    Student "1" -- "many" QuestionAttempt
    
    %% AI-Enhanced Features
    Course "1" -- "many" AIGeneratedExam
    Student "1" -- "many" AIChatHistory
    Course "1" -- "many" AIChatHistory
    Student "1" -- "many" AIUsageAnalytics
    Question *-- QuestionExplanation
    
    class User {
        -int id
        -string username
        -string password
        -string firstName
        -string lastName
        -string email
        -boolean isActive
        +authenticate()
        +updateProfile()
        +resetPassword()
    }
    
    class Admin {
        +manageUsers()
        +viewSystemStats()
        +configureSecurity()
        +manageSettings()
    }
    
    class Student {
        -string profilePic
        -string address
        -string mobile
        -int aiUsageCount
        -datetime lastAIInteraction
        +takeExam()
        +viewResults()
        +askAIAssistant()
        +viewAnalytics()
        +getAvgScore()
        +getQuizCompletionRate()
    }
    
    class Teacher {
        -string profilePic
        -string address
        -string mobile
        -boolean status
        -int salary
        -Department department
        +createExam()
        +generateAIExam()
        +evaluateResults()
        +manageQuestions()
        +setAdaptiveSettings()
        +monitorStudentProgress()
    }
    
    class Department {
        -int id
        -string name
        -boolean status
        +getTeachers()
        +getCoursesOffered()
    }
    
    class Course {
        -int id
        -string courseName
        -int questionNumber
        -int totalMarks
        -boolean allowBacktracking
        -boolean isTimed
        -string securityLevel
        -boolean sequentialQuestions
        -int totalTimeMinutes
        +addQuestion()
        +removeQuestion()
        +getExamStats()
        +generateAIQuestions()
        +configureAdaptiveSettings()
    }
    
    class Question {
        -int id
        -Course course
        -string question
        -string option1
        -string option2
        -string option3
        -string option4
        -string answer
        -int marks
        -string questionType
        -boolean isAIGenerated
        -string aiGenerationPrompt
        -string multipleAnswers
        -string shortAnswerPattern
        +checkAnswer()
        +getExplanation()
        +regenerateQuestion()
    }
    
    class QuestionAttempt {
        -int id
        -Question question
        -Student student
        -string answerSelected
        -boolean isCorrect
        -int timeTaken
        -datetime timestamp
        +getAIFeedback()
        +analyzePerformance()
    }
    
    class Result {
        -int id
        -Student student
        -Course exam
        -int marks
        -datetime date
        +generateReport()
        +getDetailedAnalysis()
        +suggestImprovements()
    }
    
    class AdaptiveQuizSettings {
        -int id
        -Course course
        -boolean isAdaptive
        -int minDifficulty
        -int maxDifficulty
        -float difficultyStep
        +calculateNextQuestionDifficulty()
        +adjustDifficulty()
        +getAdaptiveStats()
    }
    
    class AIGeneratedExam {
        -int id
        -Course course
        -string title
        -string description
        -string difficulty
        -int timeLimit
        -datetime createdAt
        -boolean approved
        +generateQuestions()
        +adjustDifficulty()
        +getModelUsed()
    }
    
    class AIChatHistory {
        -int id
        -Student student
        -Course course
        -string question
        -string answer
        -datetime timestamp
        +analyzeTopics()
        +getRelevantQuestions()
    }
    
    class AIUsageAnalytics {
        -int id
        -Student student
        -date date
        -int totalQueries
        -float averageResponseTime
        -json topicsCovered
        +generateReport()
        +identifyTrends()
        +suggestTopics()
    }
    
    class QuestionExplanation {
        -int id
        -Question question
        -string explanation
        -string furtherResources
        +getRelatedQuestions()
        +generateSimilarExplanation()
    }
```

## Class Diagram Description

This class diagram represents the object-oriented structure of the Online Examination System, including AI-enhanced features. Below are the key components and their relationships:

### User Management Classes

- **User** (Abstract): Base class for all user types with common attributes like username, password, email
  - **Student**: Users who take exams and interact with AI study tools
  - **Teacher**: Users who create and manage exams, generate AI questions
  - **Admin**: System administrators with full control

- **Department**: Organizes teachers by academic department

### Course & Question Management

- **Course**: Represents academic subjects with exam configuration options
- **Question**: Individual exam questions with various formats (multiple choice, short answer)
- **QuestionExplanation**: Detailed explanations of questions for learning purposes

### Assessment Classes

- **QuestionAttempt**: Records of student attempts at specific questions
- **Result**: Exam results linking students to courses and performance metrics

### AI-Enhanced Feature Classes

- **AdaptiveQuizSettings**: Configuration for adaptive difficulty exams
- **AIGeneratedExam**: Exams created using AI with difficulty settings
- **AIChatHistory**: Record of student interactions with AI assistant
- **AIUsageAnalytics**: Analytics of student AI feature usage

### Key Relationships

1. Inheritance relationships (Student, Teacher, Admin inherit from User)
2. Aggregation (Courses contain Questions)
3. Association (Students take Exams, Teachers manage Courses)
4. Composition (Questions have Explanations)

This class diagram illustrates how the system integrates traditional learning management capabilities with AI-powered features to enhance both teaching and learning experiences. 