# Online Examination System - Entity Relationship Diagram

```mermaid
erDiagram
    %% Core Entities
    USER {
        int id PK
        string username
        string password
        string first_name
        string last_name
        string email
        boolean is_active
        datetime date_joined
        boolean is_staff
        boolean is_superuser
    }
    
    STUDENT {
        int id PK
        int user_id FK
        string profile_pic
        string address
        string mobile
        int ai_usage_count
        datetime last_ai_interaction
    }
    
    TEACHER {
        int id PK
        int user_id FK
        int department_id FK
        string profile_pic
        string address
        string mobile
        boolean status
        int salary
    }
    
    DEPARTMENT {
        int id PK
        string name
        boolean status
    }
    
    COURSE {
        int id PK
        string course_name
        int question_number
        int total_marks
        boolean allow_backtracking
        boolean is_timed
        string security_level
        boolean sequential_questions
        int total_time_minutes
    }
    
    QUESTION {
        int id PK
        int course_id FK
        string question
        string option1
        string option2
        string option3
        string option4
        string answer
        int marks
        string question_type
        boolean is_ai_generated
        string ai_generation_prompt
        string multiple_answers
        string short_answer_pattern
    }
    
    QUESTION_ATTEMPT {
        int id PK
        int question_id FK
        int student_id FK
        string answer_selected
        boolean is_correct
        int time_taken
        datetime timestamp
    }
    
    QUESTION_EXPLANATION {
        int id PK
        int question_id FK
        string explanation
        string further_resources
    }
    
    RESULT {
        int id PK
        int student_id FK
        int course_id FK
        int marks
        datetime date
    }
    
    %% AI Feature Entities
    ADAPTIVE_QUIZ_SETTINGS {
        int id PK
        int course_id FK
        boolean is_adaptive
        int min_difficulty
        int max_difficulty
        float difficulty_step
    }
    
    AI_GENERATED_EXAM {
        int id PK
        int course_id FK
        string title
        string description
        string difficulty
        int time_limit
        datetime created_at
        boolean approved
    }
    
    AI_CHAT_HISTORY {
        int id PK
        int student_id FK
        int course_id FK
        string question
        string answer
        datetime timestamp
    }
    
    AI_USAGE_ANALYTICS {
        int id PK
        int student_id FK
        date date
        int total_queries
        float average_response_time
        string topics_covered
    }
    
    %% Relationships
    USER ||--o{ STUDENT : "has"
    USER ||--o{ TEACHER : "has"
    DEPARTMENT ||--o{ TEACHER : "employs"
    
    COURSE ||--|{ QUESTION : "contains"
    COURSE ||--o| ADAPTIVE_QUIZ_SETTINGS : "configures"
    COURSE ||--o{ AI_GENERATED_EXAM : "generates"
    
    STUDENT ||--o{ RESULT : "receives"
    COURSE ||--o{ RESULT : "produces"
    
    QUESTION ||--o{ QUESTION_ATTEMPT : "attempted as"
    STUDENT ||--o{ QUESTION_ATTEMPT : "attempts"
    
    QUESTION ||--|| QUESTION_EXPLANATION : "explained by"
    
    STUDENT ||--o{ AI_CHAT_HISTORY : "generates"
    COURSE ||--o{ AI_CHAT_HISTORY : "relates to"
    
    STUDENT ||--o{ AI_USAGE_ANALYTICS : "produces"
```

## Entity Relationship Diagram Description

This ERD represents the database structure for the Online Examination System, with a focus on displaying entities, their attributes, and relationships in a database context.

### Core Entities

#### User Management
- **USER**: Central entity storing authentication information
  - Primary users are further categorized into specific roles
- **STUDENT**: Extends USER with student-specific attributes
  - Tracks AI usage metrics and personal information
- **TEACHER**: Extends USER with teacher-specific attributes
  - Links to department and includes status and salary information
- **DEPARTMENT**: Organizational unit for teachers

#### Course Management
- **COURSE**: Represents subjects or exams
  - Contains configuration options for backtracking, timing, security
- **QUESTION**: Individual assessment items
  - Includes various question formats and AI generation flags
- **QUESTION_EXPLANATION**: Extended explanations for questions
  - Provides learning resources related to questions

#### Assessment
- **QUESTION_ATTEMPT**: Records individual question responses
  - Captures correctness, time taken, and selected answers
- **RESULT**: Stores overall exam performance
  - Links students to courses with timestamp and score

### AI Feature Entities

- **ADAPTIVE_QUIZ_SETTINGS**: Configuration for personalized exams
  - Controls difficulty progression parameters
- **AI_GENERATED_EXAM**: AI-created assessments for courses
  - Includes metadata about generation parameters and approval status
- **AI_CHAT_HISTORY**: Student interactions with AI assistant
  - Tracks questions, answers, and related courses
- **AI_USAGE_ANALYTICS**: Metrics about student AI usage
  - Aggregates usage patterns and topic coverage

### Key Relationships

1. **User Hierarchy**: USER entities can be either STUDENT or TEACHER
2. **Organizational**: TEACHERs belong to DEPARTMENTs
3. **Content**: COURSEs contain QUESTIONs, which have QUESTION_EXPLANATIONs
4. **Assessment**: STUDENTs make QUESTION_ATTEMPTs and receive RESULTs
5. **AI Interactions**: STUDENTs generate AI_CHAT_HISTORY and AI_USAGE_ANALYTICS
6. **Smart Features**: COURSEs have ADAPTIVE_QUIZ_SETTINGS and generate AI_GENERATED_EXAMs

### Cardinality Notes

- One USER can have at most one STUDENT or TEACHER profile (one-to-one)
- One COURSE can have many QUESTIONs (one-to-many)
- Each QUESTION can have exactly one QUESTION_EXPLANATION (one-to-one)
- STUDENTs can have many QUESTION_ATTEMPTs and RESULTs (one-to-many)
- COURSEs can have at most one ADAPTIVE_QUIZ_SETTINGS (one-to-one)

This database structure supports all the system requirements, including both traditional examination features and AI-enhanced capabilities. 