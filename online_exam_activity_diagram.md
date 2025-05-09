# Online Examination System - Activity Diagrams

## Student Exam Taking Flow

```mermaid
stateDiagram-v2
    [*] --> Login
    Login --> Dashboard
    Dashboard --> SelectCourse
    SelectCourse --> ViewExams
    
    ViewExams --> StartExam
    StartExam --> CheckAdaptive
    
    CheckAdaptive --> StandardExam : Not Adaptive
    CheckAdaptive --> AdaptiveExam : Adaptive
    
    StandardExam --> LoadQuestions
    AdaptiveExam --> LoadInitialQuestions
    LoadInitialQuestions --> AnswerQuestion
    LoadQuestions --> AnswerQuestion
    
    AnswerQuestion --> UpdateAdaptiveDifficulty : If Adaptive
    UpdateAdaptiveDifficulty --> NextQuestion : Has more questions
    
    AnswerQuestion --> NextQuestion : If Standard & Has more questions
    NextQuestion --> AnswerQuestion : Loop
    
    NextQuestion --> SubmitExam : No more questions
    SubmitExam --> CalculateResults
    CalculateResults --> ViewResults
    ViewResults --> RequestAIExplanation : Optional
    RequestAIExplanation --> AIGeneratesExplanation
    AIGeneratesExplanation --> ViewExplanation
    ViewExplanation --> Dashboard
    
    ViewResults --> Dashboard : Return to dashboard
    
    state AnswerQuestion {
        [*] --> ReadQuestion
        ReadQuestion --> SelectAnswer
        SelectAnswer --> SaveAnswer
        SaveAnswer --> [*]
    }
```

## Teacher Exam Creation Flow

```mermaid
stateDiagram-v2
    [*] --> TeacherLogin
    TeacherLogin --> TeacherDashboard
    TeacherDashboard --> SelectManageCourse
    SelectManageCourse --> CourseOptions
    
    CourseOptions --> CreateNewExam
    CourseOptions --> UseAIGeneration : Choose AI Assistance
    
    UseAIGeneration --> SpecifyExamParameters
    SpecifyExamParameters --> AIGeneratesQuestions
    AIGeneratesQuestions --> ReviewAIQuestions
    ReviewAIQuestions --> EditQuestions : Modify if needed
    ReviewAIQuestions --> AcceptQuestions : Approve
    EditQuestions --> AcceptQuestions
    AcceptQuestions --> SetExamSettings
    
    CreateNewExam --> ManualQuestionCreation
    ManualQuestionCreation --> AddQuestion
    AddQuestion --> MoreQuestions
    MoreQuestions --> AddQuestion : Yes
    MoreQuestions --> SetExamSettings : No
    
    SetExamSettings --> ConfigureAdaptiveSettings : If Adaptive
    ConfigureAdaptiveSettings --> SetExamTimeLimit
    
    SetExamSettings --> SetExamTimeLimit : If Standard
    SetExamTimeLimit --> SetExamSecurityLevel
    SetExamSecurityLevel --> PublishExam
    PublishExam --> NotifyStudents : Optional
    PublishExam --> TeacherDashboard
    
    state AddQuestion {
        [*] --> EnterQuestionText
        EnterQuestionText --> AddOptions : For MCQ
        EnterQuestionText --> SetAnswerPattern : For Short Answer
        AddOptions --> MarkCorrectAnswers
        MarkCorrectAnswers --> SetQuestionMarks
        SetAnswerPattern --> SetQuestionMarks
        SetQuestionMarks --> AddExplanation : Optional
        AddExplanation --> [*]
    }
```

## AI-Enhanced Learning Flow

```mermaid
stateDiagram-v2
    [*] --> StudentLogin
    StudentLogin --> StudentDashboard
    StudentDashboard --> AccessAITutor
    
    AccessAITutor --> AskQuestion
    AskQuestion --> AIProcessesQuestion
    AIProcessesQuestion --> ProvidingAnswer
    ProvidingAnswer --> SuggestRelatedTopics
    SuggestRelatedTopics --> StudentChoosesAction
    
    StudentChoosesAction --> AskFollowup : Ask follow-up
    AskFollowup --> AIProcessesQuestion
    
    StudentChoosesAction --> RequestPracticeQuestions : Practice
    RequestPracticeQuestions --> AIGeneratesPracticeQuestions
    AIGeneratesPracticeQuestions --> AttemptPracticeQuestions
    AttemptPracticeQuestions --> ReceiveFeedback
    ReceiveFeedback --> StudentChoosesAction
    
    StudentChoosesAction --> EndSession
    EndSession --> UpdateAIUsageAnalytics
    UpdateAIUsageAnalytics --> StudentDashboard
```

## Adaptive Exam Flow

```mermaid
stateDiagram-v2
    [*] --> BeginAdaptiveExam
    BeginAdaptiveExam --> LoadInitialQuestion : Medium Difficulty
    
    LoadInitialQuestion --> StudentAnswers
    StudentAnswers --> EvaluateResponse
    
    EvaluateResponse --> IncreaseDifficulty : Correct Answer
    EvaluateResponse --> DecreaseDifficulty : Incorrect Answer
    
    IncreaseDifficulty --> CheckExamCompletion
    DecreaseDifficulty --> CheckExamCompletion
    
    CheckExamCompletion --> LoadNextQuestion : More Questions
    CheckExamCompletion --> CalculateAdaptiveScore : Complete
    
    LoadNextQuestion --> StudentAnswers
    
    CalculateAdaptiveScore --> GenerateDetailedAnalysis
    GenerateDetailedAnalysis --> ShowStudentStrengthsWeaknesses
    ShowStudentStrengthsWeaknesses --> RecommendStudyResources
    RecommendStudyResources --> [*]
```

## Activity Diagram Description

The activity diagrams above illustrate the key workflows within the Online Examination System:

### Student Exam Taking Flow
This diagram depicts how students navigate through the process of taking an exam:
- Login and access their dashboard
- Select a course and view available exams
- Take either standard or adaptive exams
- Answer questions sequentially
- Submit the exam and view results
- Optionally request AI-generated explanations for difficult questions

### Teacher Exam Creation Flow
This diagram shows how teachers create and manage exams:
- Login to their dashboard and select a course to manage
- Choose between manual creation or AI-assisted generation
- Add questions with various configurations (multiple choice, short answer)
- Set exam parameters (time limits, security settings)
- Configure adaptive settings if applicable
- Publish the exam to students

### AI-Enhanced Learning Flow
This diagram illustrates the AI tutoring capabilities:
- Student accesses the AI tutor from their dashboard
- Asks questions and receives answers with related topic suggestions
- Can request practice questions based on weak areas
- System tracks AI usage analytics throughout the session

### Adaptive Exam Flow
This diagram shows the dynamic nature of adaptive assessments:
- Questions adjust in difficulty based on student performance
- System evaluates responses and increases/decreases difficulty accordingly
- Generates detailed analysis of student strengths and weaknesses
- Provides personalized study resource recommendations

These activity diagrams comprehensively model the dynamic behavior of the Online Examination System, highlighting both traditional features and AI-enhanced capabilities that support personalized learning experiences. 