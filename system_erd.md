# Online Examination System - Entity Relationship Diagram

```mermaid
erDiagram
    AUTH_USER ||--o{ STUDENT : has
    AUTH_USER ||--o{ TEACHER : has
    TEACHER }|--|{ DEPARTMENT : belongs_to
    COURSE ||--|{ QUESTION : contains
    COURSE ||--o{ ADAPTIVE_QUIZ_SETTINGS : has
    COURSE ||--o{ AI_GENERATED_EXAM : has
    COURSE ||--o{ RESULT : has
    STUDENT ||--o{ RESULT : takes
    STUDENT ||--o{ QUESTION_ATTEMPT : answers
    STUDENT ||--o{ AI_CHAT_HISTORY : interacts
    STUDENT ||--o{ AI_USAGE_ANALYTICS : generates
    QUESTION ||--o{ QUESTION_ATTEMPT : includes
    COURSE ||--o{ AI_CHAT_HISTORY : relates_to

    AUTH_USER {
        integer id PK
        string username
        string password
        string first_name
        string last_name
        string email
        boolean is_superuser
        boolean is_staff
        boolean is_active
        datetime date_joined
        datetime last_login
    }

    STUDENT {
        integer id PK
        integer user_id FK
        string profile_pic
        string address
        string mobile
        integer ai_usage_count
        datetime last_ai_interaction
    }

    TEACHER {
        integer id PK
        integer user_id FK
        integer department_id FK
        string profile_pic
        string address
        string mobile
        boolean status
        integer salary
    }

    DEPARTMENT {
        integer id PK
        string name
        boolean status
    }

    COURSE {
        integer id PK
        string course_name
        integer question_number
        integer total_marks
        boolean allow_backtracking
        boolean is_timed
        string security_level
        boolean sequential_questions
        integer total_time_minutes
    }

    QUESTION {
        integer id PK
        integer course_id FK
        string question
        string option1
        string option2
        string option3
        string option4
        string answer
        integer marks
        string question_type
        boolean is_ai_generated
        string ai_generation_prompt
        string multiple_answers
        string short_answer_pattern
    }

    QUESTION_ATTEMPT {
        integer id PK
        integer question_id FK
        integer student_id FK
        string answer_selected
        boolean is_correct
        integer time_taken
        datetime timestamp
    }

    RESULT {
        integer id PK
        integer student_id FK
        integer exam_id FK
        integer marks
        datetime date
    }

    AI_GENERATED_EXAM {
        integer id PK
        integer course_id FK
        string title
        string description
        string difficulty
        integer time_limit
        datetime created_at
        boolean approved
    }

    ADAPTIVE_QUIZ_SETTINGS {
        integer id PK
        integer course_id FK
        boolean is_adaptive
        integer min_difficulty
        integer max_difficulty
        float difficulty_step
    }

    AI_CHAT_HISTORY {
        integer id PK
        integer student_id FK
        integer course_id FK
        string question
        string answer
        datetime timestamp
    }

    AI_USAGE_ANALYTICS {
        integer id PK
        integer student_id FK
        date date
        integer total_queries
        float average_response_time
        json topics_covered
    }
```

## ERD Description

The Entity Relationship Diagram above illustrates the database schema for the Online Examination System with AI features. Here's an explanation of the key entities and their relationships:

### User Management
- **AUTH_USER**: Central user entity containing authentication details
- **STUDENT**: Extends AUTH_USER with student-specific attributes
- **TEACHER**: Extends AUTH_USER with teacher-specific attributes
- **DEPARTMENT**: Organizes teachers by academic department

### Course and Examination
- **COURSE**: Represents academic courses or subjects
- **QUESTION**: Stores exam questions associated with courses
- **QUESTION_ATTEMPT**: Records student attempts at answering questions
- **RESULT**: Tracks student exam results for specific courses

### AI Features
- **AI_GENERATED_EXAM**: Stores exams created by AI for specific courses
- **ADAPTIVE_QUIZ_SETTINGS**: Configures adaptive learning parameters for courses
- **AI_CHAT_HISTORY**: Records interactions between students and the AI assistant
- **AI_USAGE_ANALYTICS**: Tracks and analyzes student usage of AI features

### Key Relationships
1. Each AUTH_USER can be either a STUDENT or TEACHER (or admin)
2. TEACHERs belong to DEPARTMENTs
3. COURSEs contain QUESTIONs
4. STUDENTs take exams (RESULT) for COURSEs
5. STUDENTs answer QUESTIONs (QUESTION_ATTEMPT)
6. COURSEs may have ADAPTIVE_QUIZ_SETTINGS for personalized learning
7. AI_GENERATED_EXAMs are created for specific COURSEs
8. STUDENTs interact with AI and generate usage analytics

This schema supports all the AI-enhanced features described in the system documentation, including AI exam generation, adaptive exams, AI study assistance, and usage analytics. 