import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = r'c:\Users\bibek\Desktop\syntax squad\deployment\career_prediction_model.pkl'
with open(model_path, 'rb') as f:
    saved_data = pickle.load(f)

# Extract model components
model = saved_data['model']
label_encoders = saved_data['label_encoders']
target_encoder = saved_data['target_encoder']
features = saved_data['features']
primary_features = saved_data['primary_features']
secondary_features = saved_data['secondary_features']

# Load the dataset
df = pd.read_csv(r'c:\Users\bibek\Desktop\syntax squad\final_train.csv')
target = 'What would you like to become when you grow up'

# Data preprocessing (same as in training)
df[features] = df[features].fillna('Unknown')
df[target] = df[target].fillna('Corporate Employee')

# Convert numeric features
df['Academic Performance (CGPA/Percentage)'] = pd.to_numeric(
    df['Academic Performance (CGPA/Percentage)'], errors='coerce')
df['Academic Performance (CGPA/Percentage)'].fillna(
    df['Academic Performance (CGPA/Percentage)'].mean(), inplace=True)

df['Risk-Taking Ability '] = pd.to_numeric(
    df['Risk-Taking Ability '], errors='coerce')
df['Risk-Taking Ability '].fillna(
    df['Risk-Taking Ability '].median(), inplace=True)

df['Financial Stability - self/family (1 is low income and 10 is high income)'] = pd.to_numeric(
    df['Financial Stability - self/family (1 is low income and 10 is high income)'], errors='coerce')
df['Financial Stability - self/family (1 is low income and 10 is high income)'].fillna(
    df['Financial Stability - self/family (1 is low income and 10 is high income)'].median(), inplace=True)

# Encode categorical features
for feature in features:
    if df[feature].dtype == 'object':
        if feature in label_encoders:
            df[feature] = label_encoders[feature].transform(df[feature])

# Encode target
df[target] = target_encoder.transform(df[target])

# Function to test model with samples
def test_model_with_samples(model, df, features, target, n_samples=10):
    # Get random samples from the dataset
    test_samples = df.sample(n_samples, random_state=42)
    
    # Extract features and expected target
    X_test_samples = test_samples[features]
    y_test_samples = test_samples[target]
    
    # Get predictions
    y_pred_samples = model.predict(X_test_samples)
    
    # Convert to original labels
    expected_careers = target_encoder.inverse_transform(y_test_samples)
    predicted_careers = target_encoder.inverse_transform(y_pred_samples)
    
    # Create a comparison dataframe
    results = pd.DataFrame({
        'Expected': expected_careers,
        'Predicted': predicted_careers,
        'Match': expected_careers == predicted_careers
    })
    
    # Add some key features for context
    for feature in primary_features:
        if feature in X_test_samples.columns:
            if feature in label_encoders and feature in df.columns and df[feature].dtype != 'float64':
                # Convert encoded values back to original labels for categorical features
                original_values = []
                for val in X_test_samples[feature]:
                    try:
                        original_values.append(label_encoders[feature].inverse_transform([int(val)])[0])
                    except:
                        original_values.append("Unknown")
                results[feature] = original_values
            else:
                # Keep numeric features as is
                results[feature] = X_test_samples[feature].values
    
    # Calculate accuracy
    accuracy = results['Match'].mean()
    print(f"Test accuracy on {n_samples} samples: {accuracy:.2f}")
    
    return results, test_samples.index

# Run the test
test_results, sample_indices = test_model_with_samples(model, df, features, target, n_samples=10)

# Display the results
print("\nTest Results:")
print(test_results)

# Create a more detailed output with probabilities
print("\nDetailed Predictions:")
for i, (idx, row) in enumerate(zip(sample_indices, test_results.iterrows())):
    print(f"\nSample {i+1}:")
    print(f"Expected: {row[1]['Expected']}")
    print(f"Predicted: {row[1]['Predicted']}")
    print(f"Match: {'✓' if row[1]['Match'] else '✗'}")
    
    # Get input features for this sample
    input_data = df.loc[[idx]][features]
    
    # Get probabilities for all classes
    probabilities = model.predict_proba(input_data)[0]
    class_probs = {target_encoder.inverse_transform([i])[0]: prob 
                  for i, prob in enumerate(probabilities)}
    sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:3])  # Top 3 probabilities
    
    print("Top 3 Probabilities:")
    for career, prob in sorted_probs.items():
        print(f"  {career}: {prob:.2f}")
    
    # Print key features
    print("Key Features:")
    for feature in primary_features:
        if feature in row[1]:
            print(f"  {feature}: {row[1][feature]}")

# Save the results to a CSV file
test_results.to_csv(r'c:\Users\bibek\Desktop\syntax squad\model_test_results.csv', index=False)
print("\nResults saved to model_test_results.csv")