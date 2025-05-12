import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'c:\Users\bibek\Desktop\syntax squad\final_train.csv')

# Define target variable
target = 'What would you like to become when you grow up'

# Select categorical features for analysis
categorical_features = [
    'Preferred Work Environment', 
    'Motivation for Career Choice ',
    'Leadership Experience',
    'Tech-Savviness',
    'Previous Work Experience (If Any)',
    'Participation in Extracurricular Activities',
    'Gender',
    'Highest Education Level',
    'Preferred Subjects in Highschool/College'
]

# Handle missing values
df[categorical_features] = df[categorical_features].fillna('Unknown')
df[target] = df[target].fillna('Corporate Employee')

# Encode categorical features and target
label_encoders = {}
X = pd.DataFrame()

for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le
    
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[target])

# Apply chi-square test
chi2_selector = SelectKBest(chi2, k='all')
chi2_selector.fit(X, y)

# Get chi-square scores and p-values
chi2_scores = chi2_selector.scores_
p_values = chi2_selector.pvalues_

# Create a DataFrame to display results
feature_scores = pd.DataFrame({
    'Feature': categorical_features,
    'Chi-Square Score': chi2_scores,
    'P-Value': p_values
})

# Sort by chi-square score in descending order
feature_scores = feature_scores.sort_values('Chi-Square Score', ascending=False)

print("Chi-Square Test Results:")
print(feature_scores)

# Visualize the results
plt.figure(figsize=(12, 8))
sns.barplot(x='Chi-Square Score', y='Feature', data=feature_scores)
plt.title('Feature Importance based on Chi-Square Test')
plt.tight_layout()
plt.savefig(r'c:\Users\bibek\Desktop\syntax squad\chi_square_feature_importance.png')
plt.show()

# Identify statistically significant features (p-value < 0.05)
significant_features = feature_scores[feature_scores['P-Value'] < 0.05]
print("\nStatistically Significant Features (p < 0.05):")
print(significant_features)

# You can also perform the test on numerical features after binning
numerical_features = [
    'Academic Performance (CGPA/Percentage)',
    'Risk-Taking Ability ',
    'Financial Stability - self/family (1 is low income and 10 is high income)'
]

# Bin numerical features
X_numerical = pd.DataFrame()
for feature in numerical_features:
    # Convert to numeric, coerce errors to NaN
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    # Fill NaN with mean
    df[feature].fillna(df[feature].mean(), inplace=True)
    # Bin the data into 5 categories
    X_numerical[feature] = pd.qcut(df[feature], q=5, labels=False, duplicates='drop')

# Apply chi-square test on numerical features
chi2_selector_num = SelectKBest(chi2, k='all')
chi2_selector_num.fit(X_numerical, y)

# Get chi-square scores and p-values for numerical features
chi2_scores_num = chi2_selector_num.scores_
p_values_num = chi2_selector_num.pvalues_

# Create a DataFrame to display results for numerical features
numerical_feature_scores = pd.DataFrame({
    'Feature': numerical_features,
    'Chi-Square Score': chi2_scores_num,
    'P-Value': p_values_num
})

# Sort by chi-square score in descending order
numerical_feature_scores = numerical_feature_scores.sort_values('Chi-Square Score', ascending=False)

print("\nChi-Square Test Results for Numerical Features:")
print(numerical_feature_scores)

# Combine all significant features
all_significant_features = pd.concat([
    significant_features,
    numerical_feature_scores[numerical_feature_scores['P-Value'] < 0.05]
])

print("\nAll Significant Features (Recommended Feature Vector):")
print(all_significant_features)