
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gradio as gr
import os

# Load the model
model_path = 'career_prediction_model.pkl'
with open(model_path, 'rb') as f:
    saved_data = pickle.load(f)

model = saved_data['model']
label_encoders = saved_data['label_encoders']
target_encoder = saved_data['target_encoder']
features = saved_data['features']
target = 'What would you like to become when you grow up'

# Function for individual prediction
def predict_career(work_env, academic_perf, motivation, leadership, tech_savvy, preferred_subjects, gender, risk_taking=5, financial_stability=5, work_exp="No Experience"):
    # Prepare input data
    input_data = pd.DataFrame({
        'Preferred Work Environment': [work_env],
        'Academic Performance (CGPA/Percentage)': [float(academic_perf)],
        'Motivation for Career Choice ': [motivation],  # Note the space at the end
        'Leadership Experience': [leadership],
        'Tech-Savviness': [tech_savvy],
        'Preferred Subjects in Highschool/College': [preferred_subjects],  # New feature
        'Gender': [gender],  # New feature
        'Risk-Taking Ability ': [float(risk_taking)],  # Note the space at the end
        'Financial Stability - self/family (1 is low income and 10 is high income)': [float(financial_stability)],
        'Previous Work Experience (If Any)': [work_exp]
    })
    
    # Encode categorical features
    for feature in features:
        if feature in input_data.columns:
            if feature in label_encoders and input_data[feature].dtype == 'object':
                try:
                    input_data[feature] = label_encoders[feature].transform(input_data[feature])
                except ValueError:
                    # Handle unknown categories
                    print(f"Warning: Unknown category in {feature}. Using most frequent category.")
                    input_data[feature] = 0  # Default to first category
        else:
            print(f"Warning: Feature {feature} not found in input data.")
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    predicted_career = target_encoder.inverse_transform([int(prediction)])[0]
    
    # Get probabilities for all classes
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data)[0]
        class_probs = {target_encoder.inverse_transform([i])[0]: prob 
                      for i, prob in enumerate(probabilities)}
        sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))
        
        result = f"Predicted career: {predicted_career}\n\nProbabilities:\n"
        for career, prob in sorted_probs.items():
            result += f"{career}: {prob:.2f}\n"
        return result
    else:
        return f"Predicted career: {predicted_career}"

# Get unique values for dropdowns
work_env_options = list(label_encoders['Preferred Work Environment'].classes_)
motivation_options = list(label_encoders['Motivation for Career Choice '].classes_)
leadership_options = list(label_encoders['Leadership Experience'].classes_)
tech_savvy_options = list(label_encoders['Tech-Savviness'].classes_)

# Get options for new features with error handling
subject_options = []
if 'Preferred Subjects in Highschool/College' in label_encoders:
    subject_options = list(label_encoders['Preferred Subjects in Highschool/College'].classes_)
else:
    # Default options if not in the model
    subject_options = ["Science", "Commerce", "Arts", "Unknown"]

gender_options = []
if 'Gender' in label_encoders:
    gender_options = list(label_encoders['Gender'].classes_)
else:
    # Default options if not in the model
    gender_options = ["Male", "Female", "Other"]

# Get work experience options if available
work_exp_options = []
if 'Previous Work Experience (If Any)' in label_encoders:
    work_exp_options = list(label_encoders['Previous Work Experience (If Any)'].classes_)
else:
    work_exp_options = ["No Experience", "Internship", "Part Time", "Full Time"]

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_career,
    inputs=[
        gr.Dropdown(work_env_options, label="Preferred Work Environment"),
        gr.Number(label="Academic Performance (CGPA/Percentage)", minimum=0, maximum=10),
        gr.Dropdown(motivation_options, label="Motivation for Career Choice"),
        gr.Dropdown(leadership_options, label="Leadership Experience"),
        gr.Dropdown(tech_savvy_options, label="Tech-Savviness"),
        gr.Dropdown(subject_options, label="Preferred Subjects"),
        gr.Dropdown(gender_options, label="Gender"),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Risk-Taking Ability"),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Financial Stability"),
        gr.Dropdown(work_exp_options, label="Previous Work Experience")
    ],
    outputs="text",
    title="Career Prediction Model",
    description="Enter your details to predict your future career path",
    theme="huggingface"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
