"""
Student Performance Prediction CLI
CRISP-DM Phase 6: Deployment
Loads the trained Multiple Linear Regression model and outputs predicted Exam_Score
along with 95% Confidence Interval (CI) and 95% Prediction Interval (PI).
"""

import os
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_linear_model.joblib')

def load_deployed_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Please run run_analysis.py first.")
    artifact = joblib.load(MODEL_PATH)
    return artifact['model'], artifact['features'], artifact['full_columns']

def predict_student(student_data: dict, alpha=0.05):
    """
    Predict exam score for a student dictionary.
    student_data example:
    {
        'Hours_Studied': 25,
        'Attendance': 90,
        'Parental_Involvement': 'High',
        'Access_to_Resources': 'High',
        'Extracurricular_Activities': 'Yes',
        'Sleep_Hours': 7,
        'Previous_Scores': 85,
        'Motivation_Level': 'High',
        'Internet_Access': 'Yes',
        'Tutoring_Sessions': 2,
        'Family_Income': 'High',
        'Teacher_Quality': 'High',
        'School_Type': 'Public',
        'Peer_Influence': 'Positive',
        'Physical_Activity': 3,
        'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'Postgraduate',
        'Distance_from_Home': 'Near',
        'Gender': 'Female'
    }
    """
    model, selected_features, full_columns = load_deployed_model()
    
    # Create DataFrame and apply one-hot dummy encoding matching training schema
    raw_df = pd.DataFrame([student_data])
    encoded_df = pd.get_dummies(raw_df)
    
    # Align columns with full dummy features from training set
    aligned_df = pd.DataFrame(0.0, index=[0], columns=full_columns)
    for col in encoded_df.columns:
        if col in aligned_df.columns:
            aligned_df[col] = float(encoded_df[col].iloc[0])
            
    # Extract only the selected features used in the final model
    X_input = aligned_df[selected_features].copy()
    X_input_const = sm.add_constant(X_input, has_constant='add')
    
    pred_res = model.get_prediction(X_input_const)
    summary_frame = pred_res.summary_frame(alpha=alpha)
    
    pred_mean = summary_frame['mean'].iloc[0]
    ci_lower = summary_frame['mean_ci_lower'].iloc[0]
    ci_upper = summary_frame['mean_ci_upper'].iloc[0]
    pi_lower = summary_frame['obs_ci_lower'].iloc[0]
    pi_upper = summary_frame['obs_ci_upper'].iloc[0]
    se = summary_frame['mean_se'].iloc[0]
    
    return {
        'predicted_score': round(float(pred_mean), 2),
        'standard_error': round(float(se), 3),
        'confidence_interval_95': (round(float(ci_lower), 2), round(float(ci_upper), 2)),
        'prediction_interval_95': (round(float(pi_lower), 2), round(float(pi_upper), 2))
    }

if __name__ == '__main__':
    print("=" * 60)
    print("  學生考試成績線性回歸預測系統 (Student Performance Predictor)")
    print("=" * 60)
    
    # Demo Case 1: High engagement student
    student_a = {
        'Hours_Studied': 28,
        'Attendance': 95,
        'Parental_Involvement': 'High',
        'Access_to_Resources': 'High',
        'Extracurricular_Activities': 'Yes',
        'Sleep_Hours': 8,
        'Previous_Scores': 90,
        'Motivation_Level': 'High',
        'Internet_Access': 'Yes',
        'Tutoring_Sessions': 3,
        'Family_Income': 'High',
        'Teacher_Quality': 'High',
        'School_Type': 'Public',
        'Peer_Influence': 'Positive',
        'Physical_Activity': 4,
        'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'Postgraduate',
        'Distance_from_Home': 'Near',
        'Gender': 'Female'
    }
    
    res_a = predict_student(student_a)
    print("\n[範例一：高學習投入學生]")
    print(f"預測分數: {res_a['predicted_score']}")
    print(f"95% 信賴區間 (期望值 CI): {res_a['confidence_interval_95']}")
    print(f"95% 預測區間 (個別值 PI): {res_a['prediction_interval_95']}")
    
    # Demo Case 2: At-risk student
    student_b = {
        'Hours_Studied': 10,
        'Attendance': 65,
        'Parental_Involvement': 'Low',
        'Access_to_Resources': 'Low',
        'Extracurricular_Activities': 'No',
        'Sleep_Hours': 6,
        'Previous_Scores': 55,
        'Motivation_Level': 'Low',
        'Internet_Access': 'No',
        'Tutoring_Sessions': 0,
        'Family_Income': 'Low',
        'Teacher_Quality': 'Low',
        'School_Type': 'Public',
        'Peer_Influence': 'Negative',
        'Physical_Activity': 1,
        'Learning_Disabilities': 'Yes',
        'Parental_Education_Level': 'High School',
        'Distance_from_Home': 'Far',
        'Gender': 'Male'
    }
    
    res_b = predict_student(student_b)
    print("\n[範例二：潛在高風險/需輔導學生]")
    print(f"預測分數: {res_b['predicted_score']}")
    print(f"95% 信賴區間 (期望值 CI): {res_b['confidence_interval_95']}")
    print(f"95% 預測區間 (個別值 PI): {res_b['prediction_interval_95']}")
    print("=" * 60)

