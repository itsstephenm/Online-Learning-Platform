import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from django.conf import settings
import re
from .models import AIAdoptionData, AIPrediction, InsightTopic, AIInsight, AIModel
import logging
from datetime import datetime
from django.db.models import Count
import pickle
import requests
from decouple import config

logger = logging.getLogger(__name__)

# OpenRouter API configuration
OPENROUTER_API_KEY = config('OPENROUTER_API_KEY', default=None)
OPENROUTER_MODEL_NAME = config('OPENROUTER_MODEL_NAME', default="openai/gpt-3.5-turbo")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Path to save and load models
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ai_models')
os.makedirs(MODEL_DIR, exist_ok=True)
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, 'adoption_model_latest.joblib')
MODEL_VERSION = datetime.now().strftime("%Y%m%d")

# Model file paths
MODEL_PATH = os.path.join(MODEL_DIR, 'ai_adoption_model.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'target_encoder.joblib')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.joblib')

def get_chart_data(chart_type):
    """
    Generate chart data for various analytics visualizations
    
    Args:
        chart_type (str): Type of chart data to generate
        
    Returns:
        dict: Chart data in format suitable for frontend visualization
    """
    try:
        if chart_type == 'adoption_by_faculty':
            # Get faculty adoption data
            faculty_data = AIAdoptionData.objects.values('faculty').annotate(
                count=Count('id')
            ).order_by('-count')[:5]
            
            return {
                'labels': [item['faculty'] for item in faculty_data],
                'datasets': [{
                    'label': 'AI Adoption by Faculty',
                    'data': [item['count'] for item in faculty_data],
                    'backgroundColor': [
                        '#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#e74a3b'
                    ]
                }]
            }
        
        elif chart_type == 'adoption_by_study_level':
            # Get study level adoption data
            level_data = AIAdoptionData.objects.values('level_of_study').annotate(
                count=Count('id')
            ).order_by('-count')
            
            return {
                'labels': [item['level_of_study'] for item in level_data],
                'datasets': [{
                    'label': 'AI Adoption by Study Level',
                    'data': [item['count'] for item in level_data],
                    'backgroundColor': [
                        '#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#e74a3b'
                    ]
                }]
            }
        
        elif chart_type == 'prediction_distribution':
            # Get prediction distribution
            predictions = AIPrediction.objects.all()
            positive_count = sum(1 for p in predictions if p.prediction_class == 1)
            negative_count = len(predictions) - positive_count
            
            return {
                'labels': ['Positive Adoption', 'Negative Adoption'],
                'datasets': [{
                    'label': 'AI Adoption Predictions',
                    'data': [positive_count, negative_count],
                    'backgroundColor': ['#1cc88a', '#e74a3b']
                }]
            }
        
        elif chart_type == 'tools_usage':
            # Get tools usage data
            all_tools = []
            for data in AIAdoptionData.objects.all():
                if data.tools_used:
                    tools = [t.strip() for t in data.tools_used.split(',')]
                    all_tools.extend(tools)
            
            tool_counts = {}
            for tool in all_tools:
                if tool:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            # Sort and get top 5
            top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'labels': [tool for tool, count in top_tools],
                'datasets': [{
                    'label': 'Popular AI Tools',
                    'data': [count for tool, count in top_tools],
                    'backgroundColor': [
                        '#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#e74a3b'
                    ]
                }]
            }
        
        elif chart_type == 'familiarity_distribution':
            # Get familiarity distribution
            familiarity_counts = {
                1: 0, 2: 0, 3: 0, 4: 0, 5: 0
            }
            
            for data in AIAdoptionData.objects.all():
                familiarity_counts[data.ai_familiarity] = familiarity_counts.get(data.ai_familiarity, 0) + 1
            
            labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            
            return {
                'labels': labels,
                'datasets': [{
                    'label': 'AI Familiarity Distribution',
                    'data': [familiarity_counts.get(i, 0) for i in range(1, 6)],
                    'backgroundColor': [
                        '#e74a3b', '#f6c23e', '#1cc88a', '#36b9cc', '#4e73df'
                    ]
                }]
            }
        
        else:
            logger.warning(f"Unknown chart type requested: {chart_type}")
            return {
                'error': f"Unknown chart type: {chart_type}"
            }
    
    except Exception as e:
        logger.error(f"Error generating chart data: {str(e)}")
        return {
            'error': f"Could not generate chart data: {str(e)}"
        }

def process_nl_query(query_text):
    """
    Process a natural language query and generate a response using AI
    
    Args:
        query_text (str): The user's natural language query
        
    Returns:
        dict: Response containing text and/or visualization data
    """
    try:
        # Create a record of the query
        from django.contrib.auth.models import User
        query_obj = NLQuery(
            user=User.objects.get(username='admin'),  # Default to admin for now
            query=query_text,
            processed_query=query_text,  # No preprocessing for now
            response="Processing...",
        )
        query_obj.save()
        
        # Determine what type of query it is
        query_lower = query_text.lower()
        
        # Check if this is an analytics/visualization request
        viz_keywords = ['show', 'chart', 'graph', 'plot', 'visualize', 'display', 'distribution']
        data_keywords = ['adoption', 'faculty', 'study level', 'tools', 'familiarity', 'prediction']
        
        is_viz_request = any(keyword in query_lower for keyword in viz_keywords)
        
        # If it looks like a visualization request, determine the chart type
        if is_viz_request:
            if 'faculty' in query_lower:
                chart_data = get_chart_data('adoption_by_faculty')
                response = "Here's the AI adoption distribution by faculty."
                chart_type = 'bar'
                
            elif 'study level' in query_lower:
                chart_data = get_chart_data('adoption_by_study_level')
                response = "Here's the AI adoption distribution by study level."
                chart_type = 'bar'
                
            elif 'prediction' in query_lower:
                chart_data = get_chart_data('prediction_distribution')
                response = "Here's the distribution of AI adoption predictions."
                chart_type = 'pie'
                
            elif 'tools' in query_lower:
                chart_data = get_chart_data('tools_usage')
                response = "Here are the most popular AI tools used."
                chart_type = 'bar'
                
            elif 'familiar' in query_lower:
                chart_data = get_chart_data('familiarity_distribution')
                response = "Here's the distribution of AI familiarity levels."
                chart_type = 'bar'
                
            else:
                # Default to faculty chart if we can't determine
                chart_data = get_chart_data('adoption_by_faculty')
                response = "I'm showing you AI adoption by faculty, but you can ask for other visualizations like study level, predictions, tools usage, or familiarity levels."
                chart_type = 'bar'
            
            # Update the query record
            query_obj.response = response
            query_obj.response_type = 'chart'
            query_obj.chart_data = chart_data
            query_obj.save()
            
            return {
                'text': response,
                'chart': chart_data,
                'chart_type': chart_type
            }
            
        # If it's not a visualization request, use OpenRouter (if available)
        elif OPENROUTER_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": OPENROUTER_MODEL_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an AI assistant specializing in educational technology and AI adoption in education. You have access to data about AI adoption patterns among students. Answer questions in a helpful, concise, and accurate manner."
                        },
                        {
                            "role": "user",
                            "content": query_text
                        }
                    ]
                }
                
                response = requests.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    ai_response = response_data['choices'][0]['message']['content']
                    
                    # Update the query record
                    query_obj.response = ai_response
                    query_obj.response_type = 'text'
                    query_obj.save()
                    
                    return {
                        'text': ai_response
                    }
                else:
                    raise Exception(f"API returned status code {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error calling OpenRouter API: {str(e)}")
                fallback_response = "I'm sorry, I couldn't process your query through our AI service. Please try again later or ask a different question."
                
                # Update the query record with the error
                query_obj.response = f"Error: {str(e)}"
                query_obj.response_type = 'text'
                query_obj.save()
                
                return {
                    'text': fallback_response
                }
        
        # Fallback response if no OpenRouter API key
        else:
            fallback_response = (
                "I can show you visualizations about AI adoption data. Try asking something like: "
                "Show me AI adoption by faculty, "
                "Display AI familiarity distribution, or "
                "What tools are most popular?"
            )
            
            # Update the query record
            query_obj.response = fallback_response
            query_obj.response_type = 'text'
            query_obj.save()
            
            return {
                'text': fallback_response
            }
            
    except Exception as e:
        logger.error(f"Error processing natural language query: {str(e)}")
        return {
            'text': f"I'm sorry, an error occurred: {str(e)}"
        }

def clean_dataframe(df):
    """
    Clean and preprocess the CSV data
    
    Args:
        df (pd.DataFrame): Raw survey data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Rename columns to match the database model
    column_mapping = {
        '2. Level of study': 'level_of_study',
        '3. Faculty': 'faculty',
        '4. AI familiarity': 'ai_familiarity',
        '5. Used AI tools': 'uses_ai_tools',
        '6. Tools used': 'tools_used',
        '7. Usage frequency': 'usage_frequency',
        '8. Challenges': 'challenges',
        '9. Helpful tools': 'helpful_tools',
        '10. Improves learning?': 'improves_learning',
        '11. Suggestions': 'suggestions'
    }
    data = data.rename(columns=column_mapping)
    
    # Extract email domain
    if 'Email' in data.columns:
        data['email_domain'] = data['Email'].apply(lambda x: x.split('@')[-1] if pd.notnull(x) and '@' in x else 'unknown')
    
    # Convert AI familiarity to numeric
    familiarity_map = {
        'Very low': 1,
        'Low': 2,
        'Medium': 3,
        'High': 4,
        'Very high': 5
    }
    data['ai_familiarity'] = data['ai_familiarity'].map(familiarity_map)
    
    # Convert usage frequency to ordered categories
    frequency_map = {
        'Never': 'never',
        'Rarely': 'rarely',
        'Monthly': 'monthly',
        'Weekly': 'weekly',
        'Daily': 'daily'
    }
    data['usage_frequency'] = data['usage_frequency'].map(frequency_map)
    
    # Convert yes/no columns
    for col in ['uses_ai_tools', 'improves_learning']:
        if col in data.columns:
            data[col] = data[col].str.lower()
            data[col] = data[col].apply(lambda x: 'yes' if pd.notnull(x) and x.lower() == 'yes' else 'no')
    
    # Create feature for number of tools used
    if 'tools_used' in data.columns:
        data['tools_count'] = data['tools_used'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
        )
    
    # Create feature for number of challenges
    if 'challenges' in data.columns:
        data['challenges_count'] = data['challenges'].apply(
            lambda x: len(str(x).split('.')) if pd.notnull(x) else 0
        )
    
    # Create target feature - positive adoption if:
    # 1. Uses AI tools is Yes
    # 2. Usage frequency is at least Weekly
    # 3. AI familiarity is at least Medium (3)
    data['adoption_positive'] = (
        (data['uses_ai_tools'] == 'yes') & 
        (data['usage_frequency'].isin(['weekly', 'daily'])) & 
        (data['ai_familiarity'] >= 3)
    ).astype(int)
    
    # Fill missing values
    data = data.fillna({
        'level_of_study': 'Unknown',
        'faculty': 'Unknown',
        'ai_familiarity': 2,  # Default to Low
        'usage_frequency': 'never',
        'uses_ai_tools': 'no',
        'improves_learning': 'no',
        'tools_count': 0,
        'challenges_count': 0,
        'email_domain': 'unknown'
    })
    
    return data

def prepare_features(data):
    """
    Prepare features for the prediction model
    
    Args:
        data (pd.DataFrame): Processed survey data
        
    Returns:
        tuple: X (features) and y (target)
    """
    # Select features
    features = [
        "level_of_study", 
        "faculty", 
        "ai_familiarity", 
        "usage_frequency", 
        "tools_count", 
        "challenges_count"
    ]
    
    # Select target
    target = "adoption_positive"
    
    # Create feature matrix and target vector
    X = data[features]
    y = data[target]
    
    return X, y

def train_model(data, hyperparameter_tuning=False):
    """
    Train a prediction model for AI adoption
    
    Args:
        data (pd.DataFrame): Processed survey data
        hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        
    Returns:
        tuple: Trained model, preprocessor, accuracy score, and classification report
    """
    # Prepare features and target
    X, y = prepare_features(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define categorical and numerical features
    categorical_features = ["level_of_study", "faculty"]
    ordinal_features = ["ai_familiarity", "usage_frequency"]
    numerical_features = ["tools_count", "challenges_count"]
    
    # Create frequency encoder for usage frequency
    freq_encoder = {
        'never': 0,
        'rarely': 1,
        'monthly': 2,
        'weekly': 3,
        'daily': 4
    }
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ]), numerical_features),
            ('ord', Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ]), ordinal_features)
        ]
    )
    
    # Create pipeline
    if hyperparameter_tuning:
        # Define hyperparameter grid
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        
        # Create base pipeline
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Create grid search
        model = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    else:
        # Create simple pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Get the best model if hyperparameter tuning was done
    if hyperparameter_tuning:
        best_model = model.best_estimator_
        print(f"Best parameters: {model.best_params_}")
    else:
        best_model = model
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save model and preprocessor
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    # Create and save target encoder
    target_encoder = LabelEncoder()
    target_encoder.fit(['very_low', 'low', 'medium', 'high', 'very_high'])
    joblib.dump(target_encoder, ENCODER_PATH)
    
    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    return best_model, preprocessor, accuracy, classification_rep

def predict_adoption_level(data):
    """
    Make predictions about AI adoption level
    
    Args:
        data (pd.DataFrame or dict): Data to predict
        
    Returns:
        tuple: Prediction, confidence, features used
    """
    # Load model if it exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model has not been trained yet")
    
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    
    # If data is a dictionary, convert to DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Prepare features
    X, _ = prepare_features(data)
    
    # Make prediction (probability of positive class)
    proba = model.predict_proba(X)[0]
    
    # Map probability to adoption level (very_low to very_high)
    if proba[1] < 0.2:
        prediction = 'very_low'
    elif proba[1] < 0.4:
        prediction = 'low'
    elif proba[1] < 0.6:
        prediction = 'medium'
    elif proba[1] < 0.8:
        prediction = 'high'
    else:
        prediction = 'very_high'
    
    # Get confidence
    confidence = max(proba)
    
    # Get features used
    features_used = X.columns.tolist()
    
    return prediction, confidence, features_used

def analyze_text_fields(data, field='challenges'):
    """
    Analyze text fields using TF-IDF for topic extraction
    
    Args:
        data (pd.DataFrame): Survey data
        field (str): Field to analyze ('challenges' or 'suggestions')
        
    Returns:
        dict: Topics and their keywords
    """
    # Filter out empty entries
    text_data = data[field].dropna().astype(str)
    text_data = text_data[text_data != '']
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        min_df=2,
        ngram_range=(1, 2)
    )
    
    # Create TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(text_data)
    
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    
    # Get top keywords for each document
    topics = {}
    for idx, doc in enumerate(tfidf_matrix):
        # Get top 10 keywords
        top_idx = doc.toarray()[0].argsort()[-10:][::-1]
        topics[idx] = {
            'text': text_data.iloc[idx],
            'keywords': [feature_names[i] for i in top_idx]
        }
    
    return topics

def generate_insights_from_data():
    """Generate insights from the AI adoption data using OpenRouter AI"""
    try:
        insights_generated = 0
        
        # Get all adoption data
        adoption_data = AIAdoptionData.objects.all()
        
        if not adoption_data.exists():
            return {
                'success': False,
                'message': 'No adoption data available for insight generation'
            }
        
        # Convert to DataFrame
        data_list = []
        for item in adoption_data:
            data_list.append({
                'level_of_study': item.level_of_study,
                'faculty': item.faculty,
                'ai_familiarity': item.ai_familiarity,
                'uses_ai_tools': item.uses_ai_tools,
                'tools_used': item.tools_used,
                'usage_frequency': item.usage_frequency,
                'challenges': item.challenges,
                'improves_learning': item.improves_learning,
                'adoption_level': item.adoption_level
            })
        
        df = pd.DataFrame(data_list)
        
        # Generate insights using OpenRouter if available
        if OPENROUTER_API_KEY:
            # Create summary statistics for OpenRouter AI
            summary = {
                'total_records': len(df),
                'adoption_level_counts': df['adoption_level'].value_counts().to_dict(),
                'faculty_distribution': df['faculty'].value_counts().to_dict(),
                'study_level_distribution': df['level_of_study'].value_counts().to_dict(),
                'ai_familiarity_avg': float(df['ai_familiarity'].mean()),
                'uses_ai_tools_percent': float((df['uses_ai_tools'] == 'yes').mean() * 100),
                'top_tools': df['tools_used'].str.split(',').explode().value_counts().head(5).to_dict(),
                'top_challenges': df['challenges'].str.split('.').explode().value_counts().head(5).to_dict()
            }
            
            # Use OpenRouter to generate enhanced insights
            headers = {
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            prompt = f"""
            As an AI education expert, analyze this data on student AI adoption:
            
            {json.dumps(summary, indent=2)}
            
            Generate 3 key insights about:
            1. Faculty-based adoption patterns
            2. Correlation between AI familiarity and adoption
            3. Main challenges affecting adoption
            
            For each insight, provide:
            - A concise title
            - A brief explanation (2-3 sentences)
            - 1-2 actionable recommendations for educators
            
            Format each insight as a JSON object with keys: "topic", "title", "content", "recommendations".
            Return all insights in a JSON array.
            """
            
            data = {
                "model": OPENROUTER_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are an AI education expert specializing in analyzing educational technology adoption data."},
                    {"role": "user", "content": prompt}
                ]
            }
            
            try:
                response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=15)
                response.raise_for_status()
                
                # Try to parse JSON from the response
                response_data = response.json()
                insights_text = response_data['choices'][0]['message']['content']
                
                try:
                    # Parse the JSON from the response text
                    insights_data = json.loads(insights_text)
                    
                    for insight_data in insights_data:
                        # Create or get topic
                        topic, created = InsightTopic.objects.get_or_create(
                            name=insight_data.get('topic', 'AI Adoption Analysis'),
                            defaults={'description': f"Analysis of {insight_data.get('topic', 'AI adoption').lower()}"}
                        )
                        
                        # Create content with recommendations
                        content = insight_data.get('content', '')
                        if 'recommendations' in insight_data:
                            content += "\n\nRecommendations:\n" + insight_data.get('recommendations')
                        
                        # Create the insight
                        insight = AIInsight(
                            topic=topic,
                            title=insight_data.get('title', 'AI Adoption Insight'),
                            content=content,
                            source_data={"summary": summary},
                            relevance_score=0.9,
                            is_generated=True
                        )
                        insight.save()
                        insights_generated += 1
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse OpenRouter response as JSON, falling back to basic insight generation")
                    insights_generated = generate_basic_insights(df)
                
            except (requests.RequestException, KeyError) as e:
                logger.error(f"Error calling OpenRouter API: {str(e)}")
                insights_generated = generate_basic_insights(df)
        else:
            # Fallback to basic insight generation
            insights_generated = generate_basic_insights(df)
        
        return {
            'success': True,
            'insights_generated': insights_generated
        }
    
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def generate_basic_insights(df):
    """Generate basic insights without using OpenRouter AI"""
    insights_generated = 0
    
    # 1. Faculty Adoption Insights
    faculty_adoption = df.groupby('faculty')['adoption_level'].value_counts().unstack().fillna(0)
    
    if not faculty_adoption.empty:
        topic, created = InsightTopic.objects.get_or_create(
            name="Faculty Adoption",
            defaults={'description': "Analysis of AI adoption across different faculties"}
        )
        
        insight = AIInsight(
            topic=topic,
            title="AI Adoption Patterns Across Faculties",
            content="Analysis of adoption levels across different faculties shows variations in how students from different academic backgrounds are adopting AI technologies.",
            source_data={"faculty_adoption": faculty_adoption.to_dict()},
            chart_type="bar",
            relevance_score=0.9
        )
        insight.save()
        insights_generated += 1
    
    # 2. Familiarity vs Adoption
    familiarity_adoption = df.groupby(['ai_familiarity', 'adoption_level']).size().reset_index()
    familiarity_adoption.columns = ['ai_familiarity', 'adoption_level', 'count']
    
    if not familiarity_adoption.empty:
        topic, created = InsightTopic.objects.get_or_create(
            name="Familiarity vs Adoption",
            defaults={'description': "Analysis of how AI familiarity correlates with adoption levels"}
        )
        
        insight = AIInsight(
            topic=topic,
            title="Correlation Between AI Familiarity and Adoption Levels",
            content="There appears to be a correlation between students' familiarity with AI and their adoption levels. Higher familiarity tends to lead to more advanced adoption patterns.",
            source_data={"familiarity_adoption": familiarity_adoption.to_dict('records')},
            chart_type="heatmap",
            relevance_score=0.85
        )
        insight.save()
        insights_generated += 1
    
    # Add more basic insights as needed...
    
    return insights_generated

def get_data_counts():
    """
    Get counts of various data points for the dashboard
    
    Returns:
        dict: Counts of records, predictions, and insights
    """
    try:
        records_count = AIAdoptionData.objects.count()
        insights_count = AIInsight.objects.count()
        topics_count = InsightTopic.objects.count()
        
        # Adoption level distribution
        adoption_levels = AIAdoptionData.objects.values('adoption_level').annotate(
            count=Count('id')
        )
        
        # Faculty distribution
        faculty_distribution = AIAdoptionData.objects.values('faculty').annotate(
            count=Count('id')
        )
        
        # Study level distribution
        study_level_distribution = AIAdoptionData.objects.values('level_of_study').annotate(
            count=Count('id')
        )
        
        return {
            'success': True,
            'records_count': records_count,
            'insights_count': insights_count,
            'topics_count': topics_count,
            'adoption_levels': list(adoption_levels),
            'faculty_distribution': list(faculty_distribution),
            'study_level_distribution': list(study_level_distribution)
        }
    
    except Exception as e:
        logger.error(f"Error getting data counts: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def process_training_data(file_path, target_column='success'):
    """
    Process a CSV file containing training data
    
    Args:
        file_path (str): Path to the CSV file
        target_column (str): Name of the target column
        
    Returns:
        dict: Dictionary containing processing results
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get basic stats
        rows = len(df)
        columns = len(df.columns)
        
        # Check if target column exists
        if target_column not in df.columns:
            return {
                'success': False,
                'error': f"Target column '{target_column}' not found in the data"
            }
            
        # Get class distribution
        class_distribution = df[target_column].value_counts().to_dict()
        
        # Calculate missing values
        missing_values = df.isnull().sum().to_dict()
        
        # Return processing results
        return {
            'success': True,
            'rows': rows,
            'columns': columns,
            'columns_list': list(df.columns),
            'class_distribution': class_distribution,
            'missing_values': missing_values
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def train_model(training_data_obj):
    """Train a model using the uploaded training data"""
    try:
        # Read the CSV file
        file_path = os.path.join(settings.MEDIA_ROOT, str(training_data_obj.file))
        df = pd.read_csv(file_path)
        
        # Check if 'target' column exists
        if 'target' not in df.columns:
            return {
                'success': False,
                'message': 'CSV file must contain a "target" column'
            }
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Create and train the model
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_accuracy = 0
        best_model_name = ''
        model_reports = {}
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train each model and find the best one
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            model_reports[name] = {
                'accuracy': accuracy,
                'report': report
            }
            
            # Update best model if this one is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = pipeline
                best_model_name = name
        
        # Create a model object
        model_file_path = f'models/model_{training_data_obj.id}.pkl'
        full_model_path = os.path.join(settings.MEDIA_ROOT, model_file_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_model_path), exist_ok=True)
        
        # Save the model
        with open(full_model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Create insights
        insights = []
        if best_model_name == 'random_forest':
            # Extract feature importance
            feature_names = (
                numeric_features.tolist() + 
                list(best_model.named_steps['preprocessor']
                    .transformers_[1][1]
                    .get_feature_names_out(categorical_features))
            )
            
            feature_importances = best_model.named_steps['classifier'].feature_importances_
            
            # Sort by importance
            feature_importance_dict = dict(zip(feature_names, feature_importances))
            sorted_importances = sorted(
                feature_importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Create insights for top 5 features
            for i, (feature, importance) in enumerate(sorted_importances[:5]):
                insights.append({
                    'title': f'Top Feature #{i+1}',
                    'description': f'"{feature}" has an importance score of {importance:.4f}',
                    'importance': importance
                })
        
        # Create AIModel instance
        model_obj = AIModel.objects.create(
            training_data=training_data_obj,
            model_type=best_model_name,
            accuracy=best_accuracy,
            model_file=model_file_path,
            is_active=True,  # Set this model as active
        )
        
        # Deactivate all other models
        AIModel.objects.exclude(id=model_obj.id).update(is_active=False)
        
        # Add insights
        for insight in insights:
            AIInsight.objects.create(
                model=model_obj,
                title=insight['title'],
                description=insight['description'],
                importance=insight['importance']
            )
        
        return {
            'success': True,
            'message': 'Model trained successfully',
            'model_id': model_obj.id,
            'accuracy': best_accuracy,
            'insights': insights
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Error training model: {str(e)}'
        }

def prepare_features(student_data):
    """
    Prepare features for prediction from student data
    This assumes student_data contains information about the student
    """
    # Extract relevant features from student data
    features = {
        'quiz_score_avg': student_data.get('quiz_score_avg', 0),
        'exam_score_avg': student_data.get('exam_score_avg', 0),
        'attendance_rate': student_data.get('attendance_rate', 0),
        'study_time_weekly': student_data.get('study_time_weekly', 0),
        'participation_score': student_data.get('participation_score', 0),
        'assignments_completed': student_data.get('assignments_completed', 0),
        'major': student_data.get('major', 'Unknown'),
        'year_level': student_data.get('year_level', 1)
    }
    
    return features

def generate_enhanced_explanation(prediction_data, feature_importance):
    """Generate an enhanced explanation using OpenRouter AI"""
    try:
        # Import config here to avoid circular imports
        from decouple import config
        
        # Get OpenRouter API configuration
        OPENROUTER_API_KEY = config('OPENROUTER_API_KEY', default=None)
        OPENROUTER_MODEL_NAME = config('OPENROUTER_MODEL_NAME', default="openai/gpt-3.5-turbo")
        OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
        
        if not OPENROUTER_API_KEY:
            logger.info("No OpenRouter API key configured, using basic explanation")
            return generate_basic_explanation(prediction_data, feature_importance)
            
        # Format the prediction data for the prompt
        prediction_class = prediction_data.get('prediction_class', 0)
        success_probability = prediction_data.get('success_probability', 0)
        risk_probability = prediction_data.get('risk_probability', 0)
        
        # Sort feature importance by value
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Format feature importance for the prompt
        feature_importance_text = "\n".join([
            f"- {feature}: {importance:.4f}" 
            for feature, importance in sorted_features[:5]
        ])
        
        # Input data formatted as text
        input_data_text = "\n".join([
            f"- {key}: {value}" 
            for key, value in prediction_data.get('input_data', {}).items()
        ])
        
        # Create the prompt
        prompt = f"""
        As an AI education expert, analyze this student prediction:
        
        Prediction class: {'Success' if prediction_class == 1 else 'At Risk'}
        Success probability: {success_probability:.2f}%
        Risk probability: {risk_probability:.2f}%
        
        Top influencing factors:
        {feature_importance_text}
        
        Student data:
        {input_data_text}
        
        Provide a concise, helpful explanation of:
        1. What this prediction means
        2. Why the model made this prediction (based on the feature importance)
        3. Specific actionable recommendations for educators
        
        Keep your response under 250 words.
        """
        
        # Call the OpenRouter API
        import requests
        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": OPENROUTER_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are an AI education expert specializing in analyzing student data."},
                {"role": "user", "content": prompt}
            ]
        }
        
        # Make the API request
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        explanation = result['choices'][0]['message']['content'].strip()
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating enhanced explanation: {str(e)}")
        return generate_basic_explanation(prediction_data, feature_importance)

def generate_basic_explanation(prediction_data, feature_importance):
    """Generate a basic explanation without using OpenRouter API"""
    prediction_class = prediction_data.get('prediction_class', 0)
    success_probability = prediction_data.get('success_probability', 0)
    risk_probability = prediction_data.get('risk_probability', 0)
    
    # Sort feature importance by value
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]  # Top 3 features
    
    # Create basic explanation
    if prediction_class == 1:
        explanation = f"This student has a {success_probability:.1f}% probability of success. "
        explanation += "Key factors contributing to this prediction are "
        explanation += ", ".join([f"{feature.replace('_', ' ')}" for feature, _ in sorted_features])
        explanation += ". Regular check-ins and advanced materials are recommended."
    else:
        explanation = f"This student has a {risk_probability:.1f}% probability of being at risk. "
        explanation += "Key factors contributing to this prediction are "
        explanation += ", ".join([f"{feature.replace('_', ' ')}" for feature, _ in sorted_features])
        explanation += ". Early intervention with additional tutoring is recommended."
    
    return explanation

def make_prediction(input_data, model_id=None):
    """Make a prediction using the input data and specified model"""
    try:
        # Get the active model
        if model_id:
            model_obj = AIModel.objects.get(id=model_id)
        else:
            model_obj = AIModel.objects.filter(is_active=True).first()
        
        if not model_obj:
            return {
                'success': False,
                'message': 'No active model found'
            }
        
        # Load the model
        model_path = os.path.join(settings.MEDIA_ROOT, str(model_obj.model_file))
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        
        # Prepare input data
        if model_id:
            # If model_id is provided, get data from AIAdoptionData model
            adoption_data = AIAdoptionData.objects.get(id=model_id)
            
            input_data = {
                'level_of_study': adoption_data.level_of_study,
                'faculty': adoption_data.faculty,
                'ai_familiarity': adoption_data.ai_familiarity,
                'uses_ai_tools': adoption_data.uses_ai_tools,
                'tools_used': adoption_data.tools_used,
                'usage_frequency': adoption_data.usage_frequency,
                'challenges': adoption_data.challenges,
                'improves_learning': adoption_data.improves_learning
            }
        else:
            # Use provided data
            input_data = input_data
            
            # Convert gender to numerical value if necessary
            if 'gender' in input_data and isinstance(input_data['gender'], str):
                gender_map = {'M': 0, 'F': 1, 'O': 2}
                input_data['gender'] = gender_map.get(input_data['gender'], 0)
        
        # Create DataFrame with input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Scale the input data
        input_scaled = scaler.transform(input_df[feature_cols])
        
        # Make prediction
        prediction_class = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Create prediction result
        success_probability = prediction_proba[1] * 100
        risk_probability = prediction_proba[0] * 100
        
        # Get feature importance for this prediction (if available)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            # Multiply feature values by their importance
            for i, col in enumerate(feature_cols):
                scaled_value = input_scaled[0][i]
                importance = model.feature_importances_[i]
                feature_importance[col] = float(importance)  # Convert from numpy to Python native type
        
        # Generate enhanced explanation using OpenRouter
        prediction_data = {
            'prediction_class': prediction_class,
            'success_probability': success_probability,
            'risk_probability': risk_probability,
            'input_data': input_data
        }
        
        enhanced_explanation = generate_enhanced_explanation(prediction_data, feature_importance)
        
        # Save prediction to database
        prediction = AIPrediction.objects.create(
            model=model_obj,
            input_data=json.dumps(input_data),
            prediction_class=int(prediction_class),
            success_probability=float(success_probability),
            risk_probability=float(risk_probability),
            feature_importances=json.dumps(feature_importance),
            explanation=enhanced_explanation
        )
        
        # Prepare the result
        result = {
            'prediction_id': prediction.id,
            'model_name': model_obj.name,
            'model_accuracy': model_obj.accuracy,
            'last_trained': model_obj.last_trained,
            'prediction_class': int(prediction_class),
            'success_probability': float(success_probability),
            'risk_probability': float(risk_probability),
            'feature_importance': feature_importance,
            'input_data': input_data,
            'explanation': enhanced_explanation
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in make_prediction: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        } 