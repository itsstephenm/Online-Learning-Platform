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
from .models import AIAdoptionData, AIPrediction, InsightTopic, AIInsight, AIModel, AIPredictionData
import logging
from datetime import datetime
from django.db.models import Count
import pickle

logger = logging.getLogger(__name__)

# Path to save and load models
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ai_models')
os.makedirs(MODEL_DIR, exist_ok=True)
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, 'adoption_model_latest.joblib')
MODEL_VERSION = datetime.now().strftime("%Y%m%d")

# Model file paths
MODEL_PATH = os.path.join(MODEL_DIR, 'ai_adoption_model.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'target_encoder.joblib')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.joblib')

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

def generate_insights(data):
    """
    Generate insights from survey data
    
    Args:
        data (pd.DataFrame): Survey data
        
    Returns:
        list: Generated insights
    """
    insights = []
    
    # Insight 1: AI adoption rate
    adoption_rate = data['adoption_positive'].mean() * 100
    insights.append({
        'topic': 'adoption_rate',
        'content': f"The overall AI adoption rate is {adoption_rate:.1f}%.",
        'data_points': len(data),
        'confidence': 0.95
    })
    
    # Insight 2: Faculty with highest adoption
    faculty_adoption = data.groupby('faculty')['adoption_positive'].agg(['mean', 'count'])
    faculty_adoption = faculty_adoption[faculty_adoption['count'] >= 5]  # At least 5 responses
    if not faculty_adoption.empty:
        top_faculty = faculty_adoption.sort_values('mean', ascending=False).index[0]
        top_rate = faculty_adoption.loc[top_faculty, 'mean'] * 100
        insights.append({
            'topic': 'faculty_adoption',
            'content': f"The faculty with the highest AI adoption rate is {top_faculty} with {top_rate:.1f}%.",
            'data_points': faculty_adoption.loc[top_faculty, 'count'],
            'confidence': 0.9
        })
    
    # Insight 3: Most common tools
    if 'tools_used' in data.columns:
        all_tools = []
        for tools in data['tools_used'].dropna():
            tools_list = [t.strip() for t in tools.split(',')]
            all_tools.extend(tools_list)
        
        tool_counts = pd.Series(all_tools).value_counts()
        if not tool_counts.empty:
            top_tool = tool_counts.index[0]
            top_count = tool_counts[0]
            insights.append({
                'topic': 'common_tools',
                'content': f"The most commonly used AI tool is {top_tool}, mentioned {top_count} times.",
                'data_points': len(all_tools),
                'confidence': 0.85
            })
    
    # Insight 4: Learning improvement
    if 'improves_learning' in data.columns:
        yes_count = (data['improves_learning'] == 'yes').sum()
        total = len(data)
        yes_percent = (yes_count / total) * 100
        insights.append({
            'topic': 'learning_improvement',
            'content': f"{yes_percent:.1f}% of respondents report that AI tools improve their learning.",
            'data_points': total,
            'confidence': 0.9
        })
    
    # Insight 5: Most common challenge
    if 'challenges' in data.columns:
        challenges_topics = analyze_text_fields(data, 'challenges')
        if challenges_topics:
            # Get the most common keywords across all documents
            all_keywords = []
            for topic in challenges_topics.values():
                all_keywords.extend(topic['keywords'])
            
            keyword_counts = pd.Series(all_keywords).value_counts()
            if not keyword_counts.empty:
                top_challenge = keyword_counts.index[0]
                insights.append({
                    'topic': 'challenges',
                    'content': f"The most common challenge with AI tools is related to '{top_challenge}'.",
                    'data_points': len(challenges_topics),
                    'confidence': 0.8
                })
    
    return insights

def import_from_csv(csv_file, save_to_db=True):
    """
    Import and process data from a CSV file
    
    Args:
        csv_file: File-like object or path to CSV
        save_to_db (bool): Whether to save to database
        
    Returns:
        tuple: Processed data and trained model
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Clean data
    clean_data = clean_dataframe(df)
    
    # Train model
    model, preprocessor, accuracy, report = train_model(clean_data, hyperparameter_tuning=True)
    
    # Generate insights
    insights = generate_insights(clean_data)
    
    if save_to_db:
        # Save data to database
        for _, row in clean_data.iterrows():
            # Create AIAdoptionData instance
            adoption_data = AIAdoptionData(
                email_domain=row.get('email_domain', 'unknown'),
                faculty=row.get('faculty', 'Unknown'),
                level_of_study=row.get('level_of_study', 'Unknown'),
                ai_familiarity=row.get('ai_familiarity', 3),
                uses_ai_tools=row.get('uses_ai_tools', 'no'),
                tools_used=row.get('tools_used', ''),
                usage_frequency=row.get('usage_frequency', 'never'),
                challenges=row.get('challenges', ''),
                suggestions=row.get('suggestions', ''),
                improves_learning=row.get('improves_learning', 'no')
            )
            adoption_data.save()
            
            # Make prediction
            prediction, confidence, features_used = predict_adoption_level(row)
            
            # Create AIPrediction instance
            ai_prediction = AIPrediction(
                adoption_data=adoption_data,
                prediction=prediction,
                confidence=confidence,
                model_version='1.0',
                features_used=json.dumps(features_used)
            )
            ai_prediction.save()
        
        # Save insights to database
        for insight in insights:
            # Get or create topic
            topic, created = InsightTopic.objects.get_or_create(
                name=insight['topic'],
                defaults={
                    'description': insight['topic'].replace('_', ' ').title(),
                    'keywords': json.dumps([])
                }
            )
            
            # Create AIInsight instance
            ai_insight = AIInsight(
                topic=topic,
                content=insight['content'],
                data_points=insight['data_points'],
                confidence=insight['confidence'],
                generated_by='AI Import Process'
            )
            ai_insight.save()
    
    return clean_data, model, accuracy, insights

def get_chart_data(data_type='adoption_by_faculty'):
    """
    Get data for charts
    
    Args:
        data_type (str): Type of chart data to generate
        
    Returns:
        dict: Chart data
    """
    # Get adoption data from database
    adoption_data = AIAdoptionData.objects.all()
    
    if not adoption_data:
        return {"error": "No data available"}
    
    # Convert to DataFrame
    data_list = []
    for item in adoption_data:
        data_list.append({
            'id': item.id,
            'email_domain': item.email_domain,
            'faculty': item.faculty,
            'level_of_study': item.level_of_study,
            'ai_familiarity': item.ai_familiarity,
            'uses_ai_tools': item.uses_ai_tools,
            'tools_used': item.tools_used,
            'usage_frequency': item.usage_frequency,
            'challenges': item.challenges,
            'suggestions': item.suggestions,
            'improves_learning': item.improves_learning
        })
    
    df = pd.DataFrame(data_list)
    
    # Get predictions
    predictions = []
    for item in adoption_data:
        try:
            pred = AIPrediction.objects.filter(adoption_data=item).latest('prediction_date')
            predictions.append({
                'id': item.id,
                'prediction': pred.prediction,
                'confidence': pred.confidence
            })
        except AIPrediction.DoesNotExist:
            pass
    
    pred_df = pd.DataFrame(predictions)
    
    # Merge dataframes
    if not pred_df.empty:
        df = df.merge(pred_df, on='id', how='left')
    
    # Create adoption target based on database values
    df['adoption_positive'] = (
        (df['uses_ai_tools'] == 'yes') & 
        (df['usage_frequency'].isin(['weekly', 'daily'])) & 
        (df['ai_familiarity'] >= 3)
    ).astype(int)
    
    # Generate chart data based on type
    if data_type == 'adoption_by_faculty':
        # Group by faculty and count adoption
        faculty_data = df.groupby('faculty')['adoption_positive'].agg(['mean', 'count']).reset_index()
        faculty_data['adoption_rate'] = faculty_data['mean'] * 100
        faculty_data = faculty_data.sort_values('adoption_rate', ascending=False)
        
        # Prepare data for chart
        labels = faculty_data['faculty'].tolist()
        data = faculty_data['adoption_rate'].tolist()
        counts = faculty_data['count'].tolist()
        
        return {
            'type': 'bar',
            'labels': labels,
            'datasets': [{
                'label': 'Adoption Rate (%)',
                'data': data,
                'backgroundColor': 'rgba(54, 162, 235, 0.8)'
            }],
            'metadata': {
                'counts': counts
            }
        }
    
    elif data_type == 'tools_usage':
        # Extract all tools
        all_tools = []
        for tools in df['tools_used'].dropna():
            tools_list = [t.strip() for t in tools.split(',')]
            all_tools.extend(tools_list)
        
        # Count usage
        tool_counts = pd.Series(all_tools).value_counts().reset_index()
        tool_counts.columns = ['tool', 'count']
        tool_counts = tool_counts.sort_values('count', ascending=False).head(10)
        
        # Prepare data for chart
        labels = tool_counts['tool'].tolist()
        data = tool_counts['count'].tolist()
        
        return {
            'type': 'bar',
            'labels': labels,
            'datasets': [{
                'label': 'Usage Count',
                'data': data,
                'backgroundColor': 'rgba(75, 192, 192, 0.8)'
            }]
        }
    
    elif data_type == 'prediction_distribution':
        # Count predictions
        if 'prediction' in df.columns:
            pred_counts = df['prediction'].value_counts().reset_index()
            pred_counts.columns = ['prediction', 'count']
            
            # Order by adoption level
            order = ['very_low', 'low', 'medium', 'high', 'very_high']
            pred_counts['prediction'] = pd.Categorical(
                pred_counts['prediction'], categories=order, ordered=True
            )
            pred_counts = pred_counts.sort_values('prediction')
            
            # Prepare data for chart
            labels = [p.replace('_', ' ').title() for p in pred_counts['prediction']]
            data = pred_counts['count'].tolist()
            
            # Define colors
            colors = [
                'rgba(220, 53, 69, 0.8)',   # very_low - red
                'rgba(255, 193, 7, 0.8)',   # low - yellow
                'rgba(23, 162, 184, 0.8)',  # medium - teal
                'rgba(0, 123, 255, 0.8)',   # high - blue
                'rgba(40, 167, 69, 0.8)'    # very_high - green
            ]
            
            return {
                'type': 'doughnut',
                'labels': labels,
                'datasets': [{
                    'data': data,
                    'backgroundColor': colors
                }]
            }
        else:
            return {"error": "No prediction data available"}
    
    elif data_type == 'adoption_by_study_level':
        # Group by level of study and count adoption
        level_data = df.groupby('level_of_study')['adoption_positive'].agg(['mean', 'count']).reset_index()
        level_data['adoption_rate'] = level_data['mean'] * 100
        
        # Prepare data for chart
        labels = level_data['level_of_study'].tolist()
        data = level_data['adoption_rate'].tolist()
        counts = level_data['count'].tolist()
        
        return {
            'type': 'radar',
            'labels': labels,
            'datasets': [{
                'label': 'Adoption Rate (%)',
                'data': data,
                'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                'borderColor': 'rgba(54, 162, 235, 0.8)',
                'pointBackgroundColor': 'rgba(54, 162, 235, 1)',
                'pointBorderColor': '#fff'
            }],
            'metadata': {
                'counts': counts
            }
        }
    
    elif data_type == 'familiarity_distribution':
        # Count familiarity levels
        fam_counts = df['ai_familiarity'].value_counts().reset_index()
        fam_counts.columns = ['familiarity', 'count']
        
        # Map numeric levels to labels
        fam_map = {
            1: 'Very Low',
            2: 'Low',
            3: 'Medium',
            4: 'High',
            5: 'Very High'
        }
        fam_counts['familiarity_label'] = fam_counts['familiarity'].map(fam_map)
        
        # Order by familiarity level
        fam_counts = fam_counts.sort_values('familiarity')
        
        # Prepare data for chart
        labels = fam_counts['familiarity_label'].tolist()
        data = fam_counts['count'].tolist()
        
        return {
            'type': 'polarArea',
            'labels': labels,
            'datasets': [{
                'data': data,
                'backgroundColor': [
                    'rgba(220, 53, 69, 0.7)',   # Very Low - red
                    'rgba(255, 193, 7, 0.7)',   # Low - yellow
                    'rgba(23, 162, 184, 0.7)',  # Medium - teal
                    'rgba(0, 123, 255, 0.7)',   # High - blue
                    'rgba(40, 167, 69, 0.7)'    # Very High - green
                ]
            }]
        }
    
    else:
        return {"error": f"Unknown chart type: {data_type}"}

def process_nl_query(query, user):
    """
    Process a natural language query about the AI adoption data
    
    Args:
        query (str): The natural language query
        user (User): The user making the query
        
    Returns:
        dict: The processed response
    """
    # This is a simplified implementation - in a real system you might use
    # more advanced NLP techniques or an LLM for natural language understanding
    
    query_lower = query.lower()
    
    # Define some simple patterns to match
    patterns = {
        'faculty_adoption': ['faculty', 'department', 'field of study'],
        'study_level': ['undergraduate', 'postgraduate', 'doctorate', 'level of study'],
        'familiarity': ['familiarity', 'knowledge', 'experience'],
        'tools': ['tools', 'applications', 'software'],
        'challenges': ['challenges', 'difficulties', 'problems', 'issues'],
        'learning': ['learning', 'education', 'study', 'understanding']
    }
    
    # Check which patterns match the query
    matched_patterns = []
    for pattern, keywords in patterns.items():
        if any(keyword in query_lower for keyword in keywords):
            matched_patterns.append(pattern)
    
    # Default response if no patterns match
    if not matched_patterns:
        return {
            'response_type': 'text',
            'response': "I couldn't understand your query. Please try asking about faculty adoption, study level, familiarity, tools, challenges, or learning impact."
        }
    
    # Generate a response based on the matched patterns
    response = {
        'response_type': 'text',
        'response': ''
    }
    
    # Get data from database
    adoption_data = AIAdoptionData.objects.all()
    
    if 'faculty_adoption' in matched_patterns:
        faculty_counts = {}
        for record in adoption_data:
            if record.faculty not in faculty_counts:
                faculty_counts[record.faculty] = {'total': 0}
            
            faculty_counts[record.faculty]['total'] += 1
            
            adoption_level = record.adoption_level
            if adoption_level not in faculty_counts[record.faculty]:
                faculty_counts[record.faculty][adoption_level] = 0
            
            faculty_counts[record.faculty][adoption_level] += 1
        
        response['response'] = "Here's the breakdown of AI adoption by faculty:\n\n"
        for faculty, counts in faculty_counts.items():
            response['response'] += f"{faculty.capitalize()}: {counts['total']} students\n"
        
        # Add chart data
        chart_data = []
        for faculty, counts in faculty_counts.items():
            for level, count in counts.items():
                if level != 'total':
                    chart_data.append({
                        'faculty': faculty,
                        'adoption_level': level,
                        'count': count
                    })
        
        response['response_type'] = 'chart'
        response['chart_data'] = chart_data
        response['chart_type'] = 'bar'
        response['chart_title'] = 'AI Adoption by Faculty'
    
    if 'study_level' in matched_patterns:
        level_counts = {}
        for record in adoption_data:
            if record.level_of_study not in level_counts:
                level_counts[record.level_of_study] = {'total': 0}
            
            level_counts[record.level_of_study]['total'] += 1
            
            adoption_level = record.adoption_level
            if adoption_level not in level_counts[record.level_of_study]:
                level_counts[record.level_of_study][adoption_level] = 0
            
            level_counts[record.level_of_study][adoption_level] += 1
        
        response['response'] = "Here's the breakdown of AI adoption by study level:\n\n"
        for level, counts in level_counts.items():
            response['response'] += f"{level.capitalize()}: {counts['total']} students\n"
        
        # Add chart data
        chart_data = []
        for level, counts in level_counts.items():
            for adoption, count in counts.items():
                if adoption != 'total':
                    chart_data.append({
                        'level_of_study': level,
                        'adoption_level': adoption,
                        'count': count
                    })
        
        response['response_type'] = 'chart'
        response['chart_data'] = chart_data
        response['chart_type'] = 'bar'
        response['chart_title'] = 'AI Adoption by Study Level'
    
    return response

def process_csv_data(file_path):
    """
    Process a CSV file containing AI adoption data and save to database
    
    Args:
        file_path (str): Path to the uploaded CSV file
        
    Returns:
        dict: Results of the processing including counts and errors
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Validate the required columns
        required_columns = [
            'level_of_study', 'faculty', 'ai_familiarity', 
            'uses_ai_tools', 'tools_used', 'usage_frequency',
            'challenges', 'suggestions', 'improves_learning',
            'adoption_level'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {
                'success': False,
                'error': f"Missing required columns: {', '.join(missing_columns)}"
            }
        
        # Process each row and save to database
        records_added = 0
        errors = []
        
        for _, row in df.iterrows():
            try:
                # Calculate derived fields
                tools_count = len(row['tools_used'].split(',')) if pd.notnull(row['tools_used']) and row['tools_used'] else 0
                challenges_count = len(row['challenges'].split(',')) if pd.notnull(row['challenges']) and row['challenges'] else 0
                
                # Create and save the record
                adoption_data = AIAdoptionData(
                    level_of_study=row['level_of_study'],
                    faculty=row['faculty'],
                    ai_familiarity=int(row['ai_familiarity']),
                    uses_ai_tools=row['uses_ai_tools'],
                    tools_used=row['tools_used'] if pd.notnull(row['tools_used']) else None,
                    usage_frequency=row['usage_frequency'],
                    challenges=row['challenges'] if pd.notnull(row['challenges']) else None,
                    suggestions=row['suggestions'] if pd.notnull(row['suggestions']) else None,
                    improves_learning=row['improves_learning'],
                    tools_count=tools_count,
                    challenges_count=challenges_count,
                    adoption_level=row['adoption_level'],
                    source='csv'
                )
                adoption_data.save()
                records_added += 1
            except Exception as e:
                errors.append(f"Error processing row {_+1}: {str(e)}")
        
        # Generate insights from the newly added data
        if records_added > 0:
            generate_insights_from_data()
        
        return {
            'success': True,
            'records_added': records_added,
            'errors': errors if errors else None
        }
    
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def prepare_features(data):
    """
    Prepare features for model training or prediction
    
    Args:
        data (pd.DataFrame): DataFrame containing the adoption data
        
    Returns:
        tuple: X (features) and y (target) if target column is present, otherwise just X
    """
    # Create derived features
    if 'tools_used' in data.columns:
        data['tools_count'] = data['tools_used'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) and x else 0
        )
    
    if 'challenges' in data.columns:
        data['challenges_count'] = data['challenges'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) and x else 0
        )
    
    # Basic encoding for categorical variables
    data['uses_ai_tools_binary'] = data['uses_ai_tools'].apply(lambda x: 1 if x == 'yes' else 0)
    data['improves_learning_binary'] = data['improves_learning'].apply(
        lambda x: 1 if x == 'yes' else (0.5 if x == 'maybe' else 0)
    )
    
    # Encode usage frequency
    frequency_mapping = {
        'never': 0,
        'rarely': 1,
        'sometimes': 2,
        'often': 3,
        'daily': 4
    }
    data['usage_frequency_value'] = data['usage_frequency'].map(frequency_mapping)
    
    # Select features for model
    feature_columns = [
        'level_of_study', 'faculty', 'ai_familiarity',
        'uses_ai_tools_binary', 'tools_count', 'challenges_count',
        'usage_frequency_value', 'improves_learning_binary'
    ]
    
    X = data[feature_columns]
    
    # Return target if available
    if 'adoption_level' in data.columns:
        y = data['adoption_level']
        return X, y
    else:
        return X

def train_model():
    """
    Train a model using the data in the database
    
    Returns:
        dict: Results of the training process
    """
    try:
        # Get all data from the database
        adoption_data = AIAdoptionData.objects.all()
        
        if len(adoption_data) < 10:
            return {
                'success': False,
                'error': "Not enough data for training. Need at least 10 records."
            }
        
        # Convert to DataFrame
        data_list = []
        for record in adoption_data:
            data_list.append({
                'level_of_study': record.level_of_study,
                'faculty': record.faculty,
                'ai_familiarity': record.ai_familiarity,
                'uses_ai_tools': record.uses_ai_tools,
                'tools_used': record.tools_used,
                'usage_frequency': record.usage_frequency,
                'challenges': record.challenges,
                'improves_learning': record.improves_learning,
                'tools_count': record.tools_count,
                'challenges_count': record.challenges_count,
                'adoption_level': record.adoption_level
            })
        
        df = pd.DataFrame(data_list)
        
        # Prepare features and target
        X, y = prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define preprocessing for numeric and categorical features
        numeric_features = ['ai_familiarity', 'tools_count', 'challenges_count', 
                            'usage_frequency_value', 'improves_learning_binary', 'uses_ai_tools_binary']
        categorical_features = ['level_of_study', 'faculty']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create and train the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Define hyperparameter grid
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
        
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save the model
        model_path = CURRENT_MODEL_PATH
        joblib.dump(best_model, model_path)
        
        return {
            'success': True,
            'accuracy': accuracy,
            'best_params': grid_search.best_params_,
            'classification_report': class_report,
            'model_path': model_path,
            'model_version': MODEL_VERSION
        }
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def make_prediction(input_data):
    """
    Make a prediction using the trained model
    
    Args:
        input_data (dict): Dictionary containing input features
        
    Returns:
        dict: Prediction results
    """
    try:
        # Load the model
        if not os.path.exists(CURRENT_MODEL_PATH):
            return {
                'success': False,
                'error': "No trained model found. Please train a model first."
            }
        
        model = joblib.load(CURRENT_MODEL_PATH)
        
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Prepare features
        X = prepare_features(df)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(X)[0]
        max_prob_index = np.argmax(probabilities)
        confidence = probabilities[max_prob_index]
        
        # Collect feature importances if available
        feature_importances = {}
        if hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
            # For models like RandomForest that have feature_importances_
            classifier = model.steps[-1][1]
            importances = classifier.feature_importances_
            
            # Get feature names
            preprocessor = model.steps[0][1]
            feature_names = numeric_features = ['ai_familiarity', 'tools_count', 'challenges_count', 
                                               'usage_frequency_value', 'improves_learning_binary', 
                                               'uses_ai_tools_binary', 'level_of_study', 'faculty']
            
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    feature_importances[feature_names[i]] = float(importance)
        
        return {
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'feature_importances': feature_importances,
            'model_version': MODEL_VERSION
        }
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def generate_insights_from_data():
    """
    Generate insights from the AI adoption data
    
    Returns:
        dict: Results of insight generation
    """
    try:
        # Get all data from the database
        adoption_data = AIAdoptionData.objects.all()
        
        if len(adoption_data) < 5:
            return {
                'success': False,
                'error': "Not enough data for insight generation. Need at least 5 records."
            }
        
        # Convert to DataFrame
        data_list = []
        for record in adoption_data:
            data_list.append({
                'level_of_study': record.level_of_study,
                'faculty': record.faculty,
                'ai_familiarity': record.ai_familiarity,
                'uses_ai_tools': record.uses_ai_tools,
                'tools_used': record.tools_used,
                'usage_frequency': record.usage_frequency,
                'challenges': record.challenges,
                'improves_learning': record.improves_learning,
                'tools_count': record.tools_count,
                'challenges_count': record.challenges_count,
                'adoption_level': record.adoption_level
            })
        
        df = pd.DataFrame(data_list)
        
        # Generate insights by topic
        insights_generated = 0
        
        # 1. Adoption Patterns by Faculty
        faculty_adoption = df.groupby(['faculty', 'adoption_level']).size().reset_index()
        faculty_adoption.columns = ['faculty', 'adoption_level', 'count']
        
        if len(faculty_adoption) > 0:
            topic, created = InsightTopic.objects.get_or_create(
                name="Adoption by Faculty",
                defaults={'description': "Analysis of AI adoption patterns across different faculties"}
            )
            
            chart_data = faculty_adoption.to_dict('records')
            
            insight = AIInsight(
                topic=topic,
                title="AI Adoption Patterns Across Faculties",
                content="Analysis of adoption levels across different faculties shows variations in how students from different academic backgrounds are adopting AI technologies.",
                source_data={"table": "faculty_adoption", "record_count": len(faculty_adoption)},
                chart_data=chart_data,
                chart_type="bar",
                relevance_score=0.9
            )
            insight.save()
            insights_generated += 1
        
        # 2. Familiarity vs Adoption
        familiarity_adoption = df.groupby(['ai_familiarity', 'adoption_level']).size().reset_index()
        familiarity_adoption.columns = ['ai_familiarity', 'adoption_level', 'count']
        
        if len(familiarity_adoption) > 0:
            topic, created = InsightTopic.objects.get_or_create(
                name="Familiarity vs Adoption",
                defaults={'description': "Analysis of how AI familiarity correlates with adoption levels"}
            )
            
            chart_data = familiarity_adoption.to_dict('records')
            
            insight = AIInsight(
                topic=topic,
                title="Correlation Between AI Familiarity and Adoption Levels",
                content="There appears to be a correlation between students' familiarity with AI and their adoption levels. Higher familiarity tends to lead to more advanced adoption patterns.",
                source_data={"table": "familiarity_adoption", "record_count": len(familiarity_adoption)},
                chart_data=chart_data,
                chart_type="heatmap",
                relevance_score=0.85
            )
            insight.save()
            insights_generated += 1
        
        # 3. Study Level Impact
        study_level_adoption = df.groupby(['level_of_study', 'adoption_level']).size().reset_index()
        study_level_adoption.columns = ['level_of_study', 'adoption_level', 'count']
        
        if len(study_level_adoption) > 0:
            topic, created = InsightTopic.objects.get_or_create(
                name="Study Level Impact",
                defaults={'description': "Analysis of how study level affects AI adoption"}
            )
            
            chart_data = study_level_adoption.to_dict('records')
            
            insight = AIInsight(
                topic=topic,
                title="Impact of Study Level on AI Adoption",
                content="Analysis shows differences in AI adoption patterns across different levels of study, with postgraduate students generally showing higher adoption rates.",
                source_data={"table": "study_level_adoption", "record_count": len(study_level_adoption)},
                chart_data=chart_data,
                chart_type="bar",
                relevance_score=0.8
            )
            insight.save()
            insights_generated += 1
        
        # 4. Learning Improvement Perception
        learning_impact = df.groupby(['improves_learning', 'adoption_level']).size().reset_index()
        learning_impact.columns = ['improves_learning', 'adoption_level', 'count']
        
        if len(learning_impact) > 0:
            topic, created = InsightTopic.objects.get_or_create(
                name="Learning Impact",
                defaults={'description': "Analysis of perceived learning improvements from AI use"}
            )
            
            chart_data = learning_impact.to_dict('records')
            
            insight = AIInsight(
                topic=topic,
                title="Perception of AI's Impact on Learning",
                content="Students who believe AI improves their learning tend to have higher adoption levels, suggesting a correlation between perceived usefulness and adoption.",
                source_data={"table": "learning_impact", "record_count": len(learning_impact)},
                chart_data=chart_data,
                chart_type="bar",
                relevance_score=0.75
            )
            insight.save()
            insights_generated += 1
        
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

def process_training_data(file_path, training_data_obj):
    """Process the uploaded CSV file and prepare it for training"""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Store data info in the training data object
        training_data_obj.row_count = len(df)
        training_data_obj.column_names = ','.join(df.columns.tolist())
        
        # Basic data profiling
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Calculate basic stats
        stats = {
            'numerical_features': len(numerical_cols),
            'categorical_features': len(categorical_cols),
            'missing_values': df.isnull().sum().sum(),
            'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else {}
        }
        
        # Save the stats
        training_data_obj.data_profile = json.dumps(stats)
        training_data_obj.save()
        
        return {
            'success': True,
            'message': 'Data processed successfully',
            'training_data_id': training_data_obj.id
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Error processing data: {str(e)}'
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
            model = pickle.load(f)
        
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        max_prob = max(probabilities)
        
        # Create prediction data object
        prediction_data = AIPredictionData.objects.create(
            model=model_obj,
            input_data=json.dumps(input_data),
            prediction_result=prediction,
            confidence=float(max_prob)
        )
        
        return {
            'success': True,
            'prediction': prediction,
            'confidence': float(max_prob),
            'prediction_id': prediction_data.id
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Error making prediction: {str(e)}'
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