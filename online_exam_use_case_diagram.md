# Online Examination System - Use Case Diagram

```mermaid
graph LR
    %% Actor
    User((User))
    
    %% Use cases in horizontal ovals
    UC1>"create account"]
    UC2>"Login"]
    UC3>"View Available Courses"]
    UC4>"Take Standard Exam"]
    UC5>"Take Adaptive Exam"]
    UC6>"View Exam Results"]
    UC7>"Request AI Explanations"]
    UC8>"Use AI Tutor"]
    UC9>"Practice with Generated Questions"]
    UC10>"Manage Courses"]
    UC11>"Create Exam Manually"]
    UC12>"Generate Exam with AI"]
    
    %% System boundary rectangle
    subgraph System
        UC1
        UC2
        UC3
        UC4
        UC5
        UC6
        UC7
        UC8
        UC9
        UC10
        UC11
        UC12
    end
    
    %% Connections
    User --- UC1
    User --- UC2
    User --- UC3
    User --- UC4
    User --- UC5
    User --- UC6
    User --- UC7
    User --- UC8
    User --- UC9
    User --- UC10
    User --- UC11
    User --- UC12
    
    %% Styling - black and white only
    classDef default fill:white,stroke:black,stroke-width:1px
    classDef actor fill:white,stroke:black,stroke-width:1px
    classDef boundary fill:none,stroke:black,stroke-width:1px
    
    class User actor
    class System boundary
```

## Student View Use Case Diagram

```mermaid
graph LR
    %% Actor
    Student((Student))
    
    %% Use cases in horizontal ovals
    UC1>"create account"]
    UC2>"Login"]
    UC3>"View Available Courses"]
    UC4>"Take Standard Exam"]
    UC5>"Take Adaptive Exam"]
    UC6>"View Exam Results"]
    UC7>"Request AI Explanations"]
    UC8>"Use AI Tutor"]
    UC9>"Practice with Generated Questions"]
    UC10>"View profile"]
    UC11>"Update profile"]
    UC12>"Log out"]
    
    %% System boundary rectangle
    subgraph System
        UC1
        UC2
        UC3
        UC4
        UC5
        UC6
        UC7
        UC8
        UC9
        UC10
        UC11
        UC12
    end
    
    %% Connections
    Student --- UC1
    Student --- UC2
    Student --- UC3
    Student --- UC4
    Student --- UC5
    Student --- UC6
    Student --- UC7
    Student --- UC8
    Student --- UC9
    Student --- UC10
    Student --- UC11
    Student --- UC12
    
    %% Styling - black and white only
    classDef default fill:white,stroke:black,stroke-width:1px
    classDef actor fill:white,stroke:black,stroke-width:1px
    classDef boundary fill:none,stroke:black,stroke-width:1px
    
    class Student actor
    class System boundary
```

## Teacher View Use Case Diagram

```mermaid
graph LR
    %% Actor
    Teacher((Teacher))
    
    %% Use cases in horizontal ovals
    UC1>"create account"]
    UC2>"Login"]
    UC3>"Manage Courses"]
    UC4>"Create Exam Manually"]
    UC5>"Generate Exam with AI"]
    UC6>"Review and Edit Exams"]
    UC7>"Configure Adaptive Settings"]
    UC8>"View Student Results"]
    UC9>"Notify Students"]
    UC10>"View profile"]
    UC11>"Update profile"]
    UC12>"Log out"]
    
    %% System boundary rectangle
    subgraph System
        UC1
        UC2
        UC3
        UC4
        UC5
        UC6
        UC7
        UC8
        UC9
        UC10
        UC11
        UC12
    end
    
    %% Connections
    Teacher --- UC1
    Teacher --- UC2
    Teacher --- UC3
    Teacher --- UC4
    Teacher --- UC5
    Teacher --- UC6
    Teacher --- UC7
    Teacher --- UC8
    Teacher --- UC9
    Teacher --- UC10
    Teacher --- UC11
    Teacher --- UC12
    
    %% Styling - black and white only
    classDef default fill:white,stroke:black,stroke-width:1px
    classDef actor fill:white,stroke:black,stroke-width:1px
    classDef boundary fill:none,stroke:black,stroke-width:1px
    
    class Teacher actor
    class System boundary
```

## Admin View Use Case Diagram

```mermaid
graph LR
    %% Actor
    Admin((Admin))
    
    %% Use cases in horizontal ovals
    UC1>"Login"]
    UC2>"Manage User Accounts"]
    UC3>"Configure System Settings"]
    UC4>"Manage Departments"]
    UC5>"View Usage Analytics"]
    UC6>"View profile"]
    UC7>"Update profile"]
    UC8>"Log out"]
    
    %% System boundary rectangle
    subgraph System
        UC1
        UC2
        UC3
        UC4
        UC5
        UC6
        UC7
        UC8
    end
    
    %% Connections
    Admin --- UC1
    Admin --- UC2
    Admin --- UC3
    Admin --- UC4
    Admin --- UC5
    Admin --- UC6
    Admin --- UC7
    Admin --- UC8
    
    %% Styling - black and white only
    classDef default fill:white,stroke:black,stroke-width:1px
    classDef actor fill:white,stroke:black,stroke-width:1px
    classDef boundary fill:none,stroke:black,stroke-width:1px
    
    class Admin actor
    class System boundary
```

## Use Case Description

This use case diagram illustrates the primary actors and their interactions with the Online Examination System.

### Primary Actors

1. **Student**
   - Core users who take exams and use the system for learning
   - Primary interactions include exam-taking and AI learning support

2. **Teacher**
   - Create and manage exams and courses
   - Use AI tools to generate and configure assessments

3. **Admin**
   - Manage system settings and user accounts
   - Oversee departments and system analytics

4. **AI System**
   - Technical actor that provides intelligence services
   - Supports both students and teachers through various features

### Key Use Case Groups

#### Authentication
Common functionality for all human actors to access the system and manage their profiles.

#### Exam Taking
Student-centered use cases focused on course access, exam completion, and results review:
- Standard and adaptive exam modes represent different assessment approaches
- AI explanation requests extend the results viewing experience

#### AI Learning Support
Student-focused learning assistance outside formal exams:
- AI tutoring for question answering
- Practice sessions with AI-generated questions

#### Course & Exam Management
Teacher-focused administrative functions:
- Manual and AI-assisted exam creation
- Course management and student result monitoring
- Adaptive settings configuration for personalized assessments

#### System Administration
Admin-focused system configuration and oversight:
- User account management
- Department configuration
- System settings and analytics monitoring

#### AI System Services
Background services provided by the AI actor:
- Question generation for exams and practice
- Difficulty adaptation for personalized assessment
- Explanation provision for learning support
- Query responses for tutoring interactions
- Pattern tracking for analytics

### Relationships
The diagram shows both direct associations between actors and use cases (solid lines) and relationships between use cases (dotted lines) that represent extensions or inclusions:

- Taking adaptive exams extends standard exam taking
- AI explanations extend results viewing
- Adaptive settings and AI generation feed into exam creation
- AI services support their respective student and teacher features

This comprehensive use case diagram maps the full range of system functionality across all user types, highlighting both traditional examination features and AI-enhanced capabilities. 