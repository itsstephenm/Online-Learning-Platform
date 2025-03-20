# Online Quiz
![developer]()
---
## screenshots
### Homepage
![homepage snap]()
### Admin Dashboard
![dashboard snap]()
### Exam Rules
![invoice snap]()
### Exam
![exam snap]()
### Teacher
![teacher snap]()
---
## Functions
### Admin
- Create Admin account using command
```
py manage.py createsuperuser
```
- After Login, can see Total Number Of Student, Teacher, Course, Questions are there in system on Dashboard.
- Can View, Update, Delete, Approve Teacher.
- Can View, Update, Delete Student.
- Can Also See Student Marks.
- Can Add, View, Delete Course/Exams.
- Can Add Questions To Respective Courses With Options, Correct Answer, And Marks.
- Can View And Delete Questions Too.

### Teacher
- Apply for job in System. Then Login (Approval required by system admin, Then only teacher can login).
- After Login, can see Total Number Of Student, Course, Questions are there in system on Dashboard.
- Can Add, View, Delete Course/Exams.
- Can Add Questions To Respective Courses With Options, Correct Answer, And Marks.
- Can View And Delete Questions Too.
> **_NOTE:_**  Basically Admin Will Hire Teachers To Manage Courses and Questions.

### Student
- Create account (No Approval Required By Admin, Can Login After Signup)
- After Login, Can See How Many Courses/Exam And Questions Are There In System On Dashboard.
- Can Give Exam Any Time, There Is No Limit On Number Of Attempt.
- Can View Marks Of Each Attempt Of Each Exam.
- Question Pattern Is MCQ With 4 Options And 1 Correct Answer.
---

## HOW TO RUN THIS PROJECT
- Install Python(3.7.6) (Dont Forget to Tick Add to Path while installing Python)
- Open Terminal and Execute Following Commands :
```
python3 -m pip install -r requirements.txt
```
- Download This Project Zip Folder and Extract it
- Move to project folder in Terminal. Then run following Commands :
```
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py runserver
```
- Now enter following URL in Your Browser Installed On Your Pc
```
http://127.0.0.1:8000/
```

## HOW TO ACCESS DJANGO ADMIN PANEL
- Create a superuser if not already created:
```
python3 manage.py createsuperuser
```
- Open your browser and go to:
```
http://127.0.0.1:8000/admin/
```
- Log in with your superuser credentials.

## HOW TO ACCESS SQLITE EXPLORER
- Make sure you have `django-sql-explorer` installed:
```
pip install django-sql-explorer
```
- Add the following to `settings.py`:
```python
INSTALLED_APPS = [
    ...
    'explorer',
]
EXPLORER_CONNECTIONS = {'Default': 'default'}
EXPLORER_DEFAULT_CONNECTION = 'default'
```
- Run the server:
```
python3 manage.py runserver
```
- Open your browser and navigate to:
```
http://127.0.0.1:8000/explorer/
```
- Log in with an admin/staff account to execute queries.

## CHANGES REQUIRED FOR CONTACT US PAGE
- In `settings.py` file, You have to give your email and password
```
EMAIL_HOST_USER = 'youremail@gmail.com'
EMAIL_HOST_PASSWORD = 'your email password'
EMAIL_RECEIVING_USER = 'youremail@gmail.com'
```

## Drawbacks/LoopHoles
- Admin/Teacher can add any number of questions to any course, But while adding course, admin provide question number.

