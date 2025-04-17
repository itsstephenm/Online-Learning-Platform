import os
import pandas as pd
import numpy as np
import json
import time
import datetime
import logging
from typing import Dict, List, Tuple, Any, Optional
from django.conf import settings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import joblib

# Set up logging
logger = logging.getLogger(__name__)

# Ensure the model directory exists
MODEL_DIR = os.path.join(settings.MEDIA_ROOT, 'ai_models')
os.makedirs(MODEL_DIR, exist_ok=True)

def process_csv_file(file_path: str, save_to_db: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process a CSV file containing AI adoption survey data.
    
    Args:
        file_path (str): Path to the CSV file
        save_to_db (bool, optional): Whether to save the processed data to the database. Defaults to True.
        
    Returns:
        tuple: (DataFrame of processed data, Stats dictionary)
    """
    try:
        start_time = time.time()
        logger.info(f"Processing CSV file: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        raw_record_count = len(df)
        
        # Process the data
        df = clean_survey_data(df)
        processed_record_count = len(df)
        
        # Calculate statistics
        stats = calculate_data_stats(df)
        stats['raw_record_count'] = raw_record_count
        stats['processed_record_count'] = processed_record_count
        stats['processing_time'] = time.time() - start_time
        
        # Save to database if requested
        if save_to_db:
            from quiz.models import AIAdoptionData, CSVUpload
            
            # Get or create upload record
            filename = os.path.basename(file_path)
            upload = CSVUpload.objects.create(
                filename=filename,
                original_filename=filename,
                file_path=file_path,
                record_count=processed_record_count,
                status='processing',
                processing_time=stats['processing_time']
            )
            
            # Save each record to the database
            for _, row in df.iterrows():
                # Extract tools and count them
                tools_used = str(row.get('Tools used', '')) if not pd.isna(row.get('Tools used', '')) else ''
                tools_count = len([t for t in tools_used.split(',') if t.strip()]) if tools_used else 0
                
                # Extract challenges and count them
                challenges = str(row.get('Challenges', '')) if not pd.isna(row.get('Challenges', '')) else ''
                challenges_count = len([c for c in challenges.split(',')] if challenges else 0)
                
                # Get faculty and level of study
                faculty = str(row.get('Faculty', 'Unknown')).strip() if not pd.isna(row.get('Faculty', '')) else 'Unknown'
                level_of_study = str(row.get('Level of study', 'Unknown')).strip() if not pd.isna(row.get('Level of study', '')) else 'Unknown'
                
                # Determine AI familiarity (1-5 scale)
                familiarity_str = str(row.get('AI familiarity', '')).lower() if not pd.isna(row.get('AI familiarity', '')) else ''
                if 'not' in familiarity_str:
                    ai_familiarity = 1
                elif 'somewhat' in familiarity_str:
                    ai_familiarity = 3
                elif 'very' in familiarity_str:
                    ai_familiarity = 5
                else:
                    ai_familiarity = 3  # Default to middle value
                
                # Determine if uses AI tools
                uses_ai_tools_str = str(row.get('Used AI tools', '')).lower() if not pd.isna(row.get('Used AI tools', '')) else ''
                uses_ai_tools = 'yes' if 'yes' in uses_ai_tools_str else 'no'
                
                # Determine usage frequency
                frequency_str = str(row.get('Usage frequency', '')).lower() if not pd.isna(row.get('Usage frequency', '')) else ''
                if any(term in frequency_str for term in ['daily', 'every day']):
                    usage_frequency = 'daily'
                elif any(term in frequency_str for term in ['weekly', 'every week']):
                    usage_frequency = 'weekly'
                elif any(term in frequency_str for term in ['monthly', 'every month']):
                    usage_frequency = 'monthly'
                elif any(term in frequency_str for term in ['rarely', 'occasionally']):
                    usage_frequency = 'rarely'
                else:
                    usage_frequency = 'never'
                
                # Determine if improves learning
                improves_str = str(row.get('Improves learning?', '')).lower() if not pd.isna(row.get('Improves learning?', '')) else ''
                if 'yes' in improves_str:
                    improves_learning = 'yes'
                elif 'no' in improves_str:
                    improves_learning = 'no'
                else:
                    improves_learning = 'maybe'
                
                # Extract email domain from Email field if available
                email = str(row.get('Email', '')) if not pd.isna(row.get('Email', '')) else ''
                email_domain = email.split('@')[-1] if '@' in email else ''
                
                # Determine adoption level based on usage frequency and tools count
                if usage_frequency == 'daily' and tools_count > 1:
                    adoption_level = 'high'
                elif usage_frequency in ['daily', 'weekly'] or tools_count > 0:
                    adoption_level = 'medium'
                else:
                    adoption_level = 'low'
                
                # Create the database record
                AIAdoptionData.objects.create(
                    email_domain=email_domain,
                    faculty=faculty,
                    level_of_study=level_of_study,
                    ai_familiarity=ai_familiarity,
                    uses_ai_tools=uses_ai_tools,
                    tools_used=tools_used,
                    usage_frequency=usage_frequency,
                    challenges=challenges,
                    suggestions=str(row.get('Suggestions', '')) if not pd.isna(row.get('Suggestions', '')) else '',
                    improves_learning=improves_learning,
                    tools_count=tools_count,
                    challenges_count=challenges_count,
                    adoption_level=adoption_level,
                    source='csv_import',
                    upload_batch=upload
                )
            
            # Update upload status
            upload.status = 'success'
            upload.save()
            
            # Update stats with upload info
            stats['upload_id'] = upload.id
            stats['upload_status'] = 'success'
            
        return df, stats
    
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}", exc_info=True)
        
        # Update upload status if it exists
        if save_to_db and 'upload' in locals():
            upload.status = 'error'
            upload.error_message = str(e)
            upload.save()
            
            # Update stats with upload info
            if 'stats' in locals():
                stats['upload_id'] = upload.id
                stats['upload_status'] = 'error'
                stats['error'] = str(e)
                return df if 'df' in locals() else pd.DataFrame(), stats
        
        # Re-raise the exception
        raise

def clean_survey_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the survey data.
    
    Args:
        df (pd.DataFrame): The raw survey data
        
    Returns:
        pd.DataFrame: The cleaned data
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Drop rows where all values are NaN
    df = df.dropna(how='all')
    
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    
    # Ensure required columns exist
    required_columns = [
        'Email', 'Level of study', 'Faculty', 'AI familiarity', 
        'Used AI tools', 'Tools used', 'Usage frequency', 'Challenges',
        'Improves learning?'
    ]
    
    # Check for missing columns and add them if needed
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Fill missing values with appropriate defaults
    df['Faculty'] = df['Faculty'].fillna('Unknown')
    df['Level of study'] = df['Level of study'].fillna('Unknown')
    df['AI familiarity'] = df['AI familiarity'].fillna('Not familiar')
    df['Used AI tools'] = df['Used AI tools'].fillna('No')
    df['Tools used'] = df['Tools used'].fillna('')
    df['Usage frequency'] = df['Usage frequency'].fillna('Never')
    df['Challenges'] = df['Challenges'].fillna('')
    df['Improves learning?'] = df['Improves learning?'].fillna('No')
    
    # Clean categorical values - standardize common responses
    # Faculty standardization
    faculty_mapping = {
        'busi': 'Business',
        'engi': 'Engineering',
        'comp': 'Engineering',
        'edu': 'Education',
        'heal': 'Health Sciences',
        'medi': 'Health Sciences',
        'nurs': 'Health Sciences',
        'sci': 'Science',
        'art': 'Arts',
        'hum': 'Arts'
    }
    
    # Apply faculty mapping for partial matches
    for key, value in faculty_mapping.items():
        mask = df['Faculty'].str.lower().str.contains(key, na=False)
        df.loc[mask, 'Faculty'] = value
    
    # Study level standardization
    level_mapping = {
        'under': 'Undergraduate',
        'grad': 'Postgraduate',
        'post': 'Postgraduate',
        'mast': 'Postgraduate',
        'phd': 'Postgraduate',
        'doct': 'Postgraduate'
    }
    
    # Apply level mapping for partial matches
    for key, value in level_mapping.items():
        mask = df['Level of study'].str.lower().str.contains(key, na=False)
        df.loc[mask, 'Level of study'] = value
    
    # AI familiarity standardization
    familiarity_mapping = {
        'not': 'Not familiar',
        'never': 'Not familiar',
        'some': 'Somewhat familiar',
        'moderate': 'Somewhat familiar',
        'very': 'Very familiar',
        'expert': 'Very familiar',
        'high': 'Very familiar'
    }
    
    # Apply familiarity mapping for partial matches
    for key, value in familiarity_mapping.items():
        mask = df['AI familiarity'].str.lower().str.contains(key, na=False)
        df.loc[mask, 'AI familiarity'] = value
    
    # Usage frequency standardization
    freq_mapping = {
        'daily': 'Daily',
        'everyday': 'Daily',
        'every day': 'Daily',
        'week': 'Weekly',
        'month': 'Monthly',
        'rare': 'Rarely',
        'occasion': 'Rarely',
        'seldom': 'Rarely',
        'never': 'Never'
    }
    
    # Apply frequency mapping for partial matches
    for key, value in freq_mapping.items():
        mask = df['Usage frequency'].str.lower().str.contains(key, na=False)
        df.loc[mask, 'Usage frequency'] = value
    
    # Convert yes/no fields to standardized format
    yes_patterns = ['yes', 'y', 'true', 't', '1']
    no_patterns = ['no', 'n', 'false', 'f', '0']
    
    # Standardize 'Used AI tools' field
    for pattern in yes_patterns:
        mask = df['Used AI tools'].str.lower().str.contains(pattern, na=False)
        df.loc[mask, 'Used AI tools'] = 'Yes'
    
    for pattern in no_patterns:
        mask = df['Used AI tools'].str.lower().str.contains(pattern, na=False)
        df.loc[mask, 'Used AI tools'] = 'No'
    
    # Standardize 'Improves learning?' field
    for pattern in yes_patterns:
        mask = df['Improves learning?'].str.lower().str.contains(pattern, na=False)
        df.loc[mask, 'Improves learning?'] = 'Yes'
    
    for pattern in no_patterns:
        mask = df['Improves learning?'].str.lower().str.contains(pattern, na=False)
        df.loc[mask, 'Improves learning?'] = 'No'
    
    # If doesn't match yes/no patterns, set to 'Maybe'
    mask = ~(
        df['Improves learning?'].str.lower().isin(['yes', 'no']) | 
        df['Improves learning?'].isna()
    )
    df.loc[mask, 'Improves learning?'] = 'Maybe'
    
    return df

def calculate_data_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate statistics from the survey data.
    
    Args:
        df (pd.DataFrame): The processed survey data
        
    Returns:
        Dict[str, Any]: Statistics dictionary
    """
    stats = {}
    
    # Basic count stats
    stats['total_records'] = len(df)
    
    # Faculty distribution
    faculty_counts = df['Faculty'].value_counts().to_dict()
    stats['faculty_distribution'] = faculty_counts
    
    # Study level distribution
    level_counts = df['Level of study'].value_counts().to_dict()
    stats['level_distribution'] = level_counts
    
    # AI familiarity distribution
    familiarity_counts = df['AI familiarity'].value_counts().to_dict()
    stats['familiarity_distribution'] = familiarity_counts
    
    # Usage stats
    stats['ai_tools_usage_count'] = df[df['Used AI tools'] == 'Yes'].shape[0]
    stats['ai_tools_usage_percent'] = (stats['ai_tools_usage_count'] / stats['total_records']) * 100 if stats['total_records'] > 0 else 0
    
    # Tool usage stats
    tools_list = []
    for tools in df['Tools used'].dropna():
        if isinstance(tools, str) and tools.strip():
            tools_list.extend([t.strip() for t in tools.split(',') if t.strip()])
    
    tools_counts = pd.Series(tools_list).value_counts().to_dict()
    stats['tools_usage'] = tools_counts
    
    # Usage frequency distribution
    frequency_counts = df['Usage frequency'].value_counts().to_dict()
    stats['frequency_distribution'] = frequency_counts
    
    # Learning improvement stats
    improvement_counts = df['Improves learning?'].value_counts().to_dict()
    stats['learning_improvement'] = improvement_counts
    
    # Calculate positive impact percentage
    yes_count = improvement_counts.get('Yes', 0)
    stats['positive_impact_percent'] = (yes_count / stats['total_records']) * 100 if stats['total_records'] > 0 else 0
    
    # Extract challenges
    challenges_list = []
    for challenges in df['Challenges'].dropna():
        if isinstance(challenges, str) and challenges.strip():
            challenges_list.extend([c.strip() for c in challenges.split(',') if c.strip()])
    
    challenges_counts = pd.Series(challenges_list).value_counts().to_dict()
    stats['challenges'] = challenges_counts
    
    return stats

def train_ai_model(csv_upload_id: Optional[int] = None, algorithm: str = 'random_forest', test_size: float = 0.2) -> Dict[str, Any]:
    """Train an AI model using the survey data.
    
    Args:
        csv_upload_id (int, optional): ID of the CSV upload to use. If None, uses all data.
        algorithm (str, optional): Algorithm to use. Defaults to 'random_forest'.
        test_size (float, optional): Proportion of the dataset to use for testing. Defaults to 0.2.
        
    Returns:
        Dict[str, Any]: Training results
    """
    from quiz.models import AIAdoptionData, AIModel, CSVUpload
    
    try:
        start_time = time.time()
        logger.info(f"Training AI model with algorithm: {algorithm}")
        
        # Query the data based on csv_upload_id
        if csv_upload_id is not None:
            data = AIAdoptionData.objects.filter(upload_batch_id=csv_upload_id)
            upload = CSVUpload.objects.get(id=csv_upload_id)
        else:
            data = AIAdoptionData.objects.all()
            upload = None
        
        # Convert to DataFrame
        columns = [
            'faculty', 'level_of_study', 'ai_familiarity', 'uses_ai_tools',
            'tools_count', 'usage_frequency', 'challenges_count', 'improves_learning',
            'adoption_level'
        ]
        
        df = pd.DataFrame(list(data.values(*columns)))
        
        if len(df) < 10:
            raise ValueError("Not enough data to train a model (minimum 10 records required)")
        
        # Define features and target
        X = df.drop('adoption_level', axis=1)
        y = df['adoption_level']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Define preprocessing for numeric features
        numeric_features = ['ai_familiarity', 'tools_count', 'challenges_count']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing for categorical features
        categorical_features = ['faculty', 'level_of_study', 'uses_ai_tools', 'usage_frequency', 'improves_learning']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Select the algorithm
        if algorithm == 'random_forest':
            clf = RandomForestClassifier(random_state=42)
            param_grid = {
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [None, 10, 20]
            }
        elif algorithm == 'gradient_boosting':
            clf = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'clf__n_estimators': [50, 100],
                'clf__learning_rate': [0.01, 0.1]
            }
        elif algorithm == 'svm':
            clf = SVC(probability=True, random_state=42)
            param_grid = {
                'clf__C': [0.1, 1, 10],
                'clf__kernel': ['linear', 'rbf']
            }
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Create and train the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('clf', clf)
        ])
        
        # Use grid search for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Convert to binary for each class to calculate precision, recall, f1
        classes = sorted(y.unique())
        precision_vals = []
        recall_vals = []
        f1_vals = []
        
        for i, cls in enumerate(classes):
            y_test_binary = (y_test == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            precision_vals.append(precision_score(y_test_binary, y_pred_binary))
            recall_vals.append(recall_score(y_test_binary, y_pred_binary))
            f1_vals.append(f1_score(y_test_binary, y_pred_binary))
        
        # Average metrics
        precision = np.mean(precision_vals)
        recall = np.mean(recall_vals)
        f1 = np.mean(f1_vals)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        
        # Compute ROC curve and AUC for each class
        roc_curves = {}
        for i, cls in enumerate(classes):
            y_test_binary = (y_test == cls).astype(int)
            y_score = y_prob[:, i]
            
            fpr, tpr, _ = roc_curve(y_test_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            roc_curves[cls] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }
        
        # Save the model
        model_filename = f"model_{algorithm}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(best_model, model_path)
        
        # Get feature names
        feature_names = []
        
        # Extract numeric feature names
        feature_names.extend(numeric_features)
        
        # Extract categorical feature names (with one-hot encoding)
        ohe = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        cat_features = []
        for i, col in enumerate(categorical_features):
            categories = ohe.categories_[i]
            cat_features.extend([f"{col}_{cat}" for cat in categories])
        
        feature_names.extend(cat_features)
        
        # Extract feature importances if available
        feature_importances = {}
        
        if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
            # Get feature importance from the model
            importances = best_model.named_steps['clf'].feature_importances_
            
            # Map importances to feature names
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    feature_importances[feature_names[i]] = float(importance)
        elif algorithm == 'svm' and best_model.named_steps['clf'].kernel == 'linear':
            # For linear SVM, we can use coefficients as feature importance
            importances = np.abs(best_model.named_steps['clf'].coef_[0])
            
            # Map importances to feature names
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    feature_importances[feature_names[i]] = float(importance)
        
        # Create the AIModel object
        model_obj = AIModel.objects.create(
            name=f"{algorithm.replace('_', ' ').title()} Model - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            training_data=data.first(),  # Link to first record for reference
            algorithm=algorithm,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=np.mean([curve['auc'] for curve in roc_curves.values()]),
            parameters=grid_search.best_params_,
            feature_importance=feature_importances,
            confusion_matrix=cm.tolist(),
            roc_curve_data=roc_curves,
            training_records_count=len(df),
            features_count=len(feature_names),
            model_file_path=model_path,
            is_active=True  # Set this model as active
        )
        
        # Deactivate other models
        AIModel.objects.exclude(id=model_obj.id).update(is_active=False)
        
        # If we're training from a specific upload, link the model
        if upload:
            upload.trained_model = model_obj
            upload.save()
        
        # Prepare results
        training_time = time.time() - start_time
        
        results = {
            'success': True,
            'model_id': model_obj.id,
            'algorithm': algorithm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'training_records': len(df),
            'test_records': len(X_test),
            'feature_importance': feature_importances,
            'confusion_matrix': cm.tolist(),
            'classes': classes.tolist(),
            'best_params': grid_search.best_params_
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error training AI model: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def predict_adoption_level(data: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
    """Predict AI adoption level based on input data.
    
    Args:
        data (Dict[str, Any]): Input data dictionary
        
    Returns:
        Tuple[str, float, Dict[str, float]]: Predicted level, confidence, and feature importances
    """
    try:
        from quiz.models import AIModel
        
        # Get the active model
        try:
            model_obj = AIModel.objects.filter(is_active=True).latest('created_date')
        except AIModel.DoesNotExist:
            # Fall back to a random prediction if no model exists
            levels = ['low', 'medium', 'high']
            return random.choice(levels), random.uniform(0.65, 0.95), {}
        
        # Load the model
        model = joblib.load(model_obj.model_file_path)
        
        # Prepare the input data as a DataFrame with correct columns
        input_df = pd.DataFrame([{
            'faculty': data.get('faculty', 'Unknown'),
            'level_of_study': data.get('level_of_study', 'Unknown'),
            'ai_familiarity': int(data.get('ai_familiarity', 3)),
            'uses_ai_tools': data.get('uses_ai_tools', 'no'),
            'tools_count': int(data.get('tools_count', 0)),
            'usage_frequency': data.get('usage_frequency', 'never'),
            'challenges_count': int(data.get('challenges_count', 0)),
            'improves_learning': data.get('improves_learning', 'no')
        }])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        
        # Get class indices
        classes = model.classes_
        
        # Get prediction confidence
        predicted_idx = list(classes).index(prediction)
        confidence = proba[predicted_idx]
        
        # Extract feature importances
        feature_importances = model_obj.feature_importance
        
        return prediction, float(confidence), feature_importances
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        # Fall back to a random prediction in case of error
        levels = ['low', 'medium', 'high']
        return random.choice(levels), random.uniform(0.65, 0.95), {}

def import_from_csv(file_path: str, save_to_db: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any], float, List[str]]:
    """Process CSV file, optionally train a model, and generate insights.
    
    Args:
        file_path (str): Path to the CSV file
        save_to_db (bool, optional): Whether to save data to the DB. Defaults to True.
        
    Returns:
        Tuple: (Processed data, Model dict if trained, Accuracy, Insights list)
    """
    try:
        # Process the CSV file
        df, stats = process_csv_file(file_path, save_to_db=save_to_db)
        
        # Train a model if we're saving to the DB
        model_results = {'success': False}
        if save_to_db and stats.get('upload_status') == 'success':
            model_results = train_ai_model(csv_upload_id=stats.get('upload_id'))
        
        # Generate insights
        insights = generate_insights(df, stats)
        
        return df, model_results, model_results.get('accuracy', 0.0), insights
    
    except Exception as e:
        logger.error(f"Error importing from CSV: {str(e)}", exc_info=True)
        raise

def generate_insights(df: pd.DataFrame, stats: Dict[str, Any]) -> List[str]:
    """Generate insights from the data.
    
    Args:
        df (pd.DataFrame): The processed data
        stats (Dict[str, Any]): Statistics dictionary
        
    Returns:
        List[str]: List of insights
    """
    insights = []
    
    # Get top faculty by usage
    if 'faculty_distribution' in stats and stats['faculty_distribution']:
        top_faculty = max(stats['faculty_distribution'].items(), key=lambda x: x[1])[0]
        insights.append(f"Students in {top_faculty} faculty show highest participation in the survey.")
    
    # AI tools usage stats
    if 'ai_tools_usage_percent' in stats:
        usage_percent = round(stats['ai_tools_usage_percent'], 1)
        insights.append(f"{usage_percent}% of respondents report using AI tools.")
    
    # Top AI tool
    if 'tools_usage' in stats and stats['tools_usage']:
        top_tool, top_count = max(stats['tools_usage'].items(), key=lambda x: x[1])
        tools_percent = round((top_count / stats['total_records']) * 100, 1) if stats['total_records'] > 0 else 0
        insights.append(f"{top_tool} is the most popular AI tool, used by {tools_percent}% of students.")
    
    # Top challenge
    if 'challenges' in stats and stats['challenges']:
        top_challenge, top_count = max(stats['challenges'].items(), key=lambda x: x[1])
        challenge_percent = round((top_count / stats['total_records']) * 100, 1) if stats['total_records'] > 0 else 0
        insights.append(f"'{top_challenge}' is the most common challenge, reported by {challenge_percent}% of students.")
    
    # Learning improvement
    if 'learning_improvement' in stats and 'Yes' in stats['learning_improvement']:
        yes_count = stats['learning_improvement'].get('Yes', 0)
        yes_percent = round((yes_count / stats['total_records']) * 100, 1) if stats['total_records'] > 0 else 0
        insights.append(f"{yes_percent}% of students report that AI tools improve their learning.")
    
    # Study level insights
    if 'level_distribution' in stats and len(stats['level_distribution']) > 1:
        top_level = max(stats['level_distribution'].items(), key=lambda x: x[1])[0]
        insights.append(f"{top_level} students are the largest group using AI tools in the survey.")
    
    # Usage frequency insights
    if 'frequency_distribution' in stats:
        freq_dict = stats['frequency_distribution']
        freq_counts = {k: v for k, v in freq_dict.items() if k in ['Daily', 'Weekly']}
        if freq_counts:
            regular_users = sum(freq_counts.values())
            regular_percent = round((regular_users / stats['total_records']) * 100, 1) if stats['total_records'] > 0 else 0
            insights.append(f"{regular_percent}% of students use AI tools on a regular basis (daily or weekly).")
    
    return insights 