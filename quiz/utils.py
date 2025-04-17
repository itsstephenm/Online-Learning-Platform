def train_ai_model(csv_upload_id=None):
    """Train AI model on the uploaded data."""
    try:
        # Get the data
        if csv_upload_id:
            data = AIAdoptionData.objects.filter(upload_batch_id=csv_upload_id)
        else:
            data = AIAdoptionData.objects.all()
        
        if not data.exists():
            return {
                'success': False,
                'error': 'No data available for training'
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(list(data.values()))
        
        # Prepare features
        features = [
            'ai_familiarity',
            'faculty',
            'level_of_study',
            'usage_frequency',
            'uses_ai_tools'
        ]
        
        # Encode categorical variables
        categorical_features = ['faculty', 'level_of_study', 'usage_frequency', 'uses_ai_tools']
        df_encoded = pd.get_dummies(df[features], columns=categorical_features)
        
        # Target variable (improves_learning)
        y = df['improves_learning'].map({'yes': 1, 'no': 0, 'maybe': 0.5})
        X = df_encoded
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model (Random Forest)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions and accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_data = AIModel.objects.create(
            name=f'RF_Model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            algorithm='random_forest',
            accuracy=accuracy,
            features=features,
            upload_batch_id=csv_upload_id if csv_upload_id else None
        )
        
        # Save feature importances
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        model_data.feature_importance = feature_importance.to_json()
        model_data.save()
        
        return {
            'success': True,
            'accuracy': accuracy,
            'model_id': model_data.id
        }
        
    except Exception as e:
        logger.error(f"Error in train_ai_model: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        } 