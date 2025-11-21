import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. Load Real Data from Kaggle
# ==========================================
# Replace the previous 'create_dummy_data' function with this:
df = pd.read_csv(r"C:\Users\chara\PycharmProjects\Employee_Attrition\Dataset\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# IMPORTANT: The Kaggle dataset uses "Yes" and "No" for Attrition.
# We must convert this to 1 and 0 for the machine learning model.
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print(f"Dataset Shape: {df.shape}")
print(df.head())

# ==========================================
# 2. Data Preprocessing (Updated)
# ==========================================

# Separate Features (X) and Target (y)
# We drop 'Attrition' because it's the target.
# We also drop 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'
# as they are useless identifiers or have the same value for everyone.
X = df.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
y = df['Attrition']

# Identify Categorical Columns automatically
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Apply Label Encoding to Binary columns and One-Hot to others
le = LabelEncoder()

for col in categorical_cols:
    # If a column has only 2 unique values (like OverTime: Yes/No), use Label Encoding
    if len(X[col].unique()) <= 2:
        X[col] = le.fit_transform(X[col])
    else:
        # Otherwise use One-Hot Encoding (like Department: Sales/HR/R&D)
        X = pd.get_dummies(X, columns=[col], drop_first=True)

# Split data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 3. Model Training (Random Forest)
# ==========================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. Evaluation
# ==========================================
y_pred = model.predict(X_test)

print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualizing the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (0=Stayed, 1=Left)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==========================================
# 5. Feature Importance (Corrected for Kaggle Data)
# ==========================================

# 1. Get the correct feature names directly from the X DataFrame
# (We must do this BEFORE scaling converted X into a numpy array,
# but since X is still available in memory, we use it here)
feature_names = X.columns

# 2. Get importances from the model
importances = model.feature_importances_

# 3. Sort them to find the most important ones
indices = np.argsort(importances)[::-1]

print("\n--- Top 10 Factors Driving Attrition ---")
for i in range(10):
    # We use a try-except block just in case of any lingering index mismatches
    try:
        print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
    except IndexError:
        continue

# 4. Plotting
plt.figure(figsize=(12, 6)) # Made the chart slightly wider for readability
plt.title("Top 15 Features in Predicting Attrition")

# Only plot the top 15 features to avoid overcrowding the chart
top_n = 15
plt.bar(range(top_n), importances[indices[:top_n]], align="center")

# Use the correct names for the x-axis labels
plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
plt.tight_layout()
plt.show()