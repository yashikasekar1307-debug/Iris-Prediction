# -------------------------------
# Iris Prediction 
# 1️Upload CSV
from google.colab import files
uploaded = files.upload()  # Upload your IRIS.csv

# 2️Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_style("whitegrid")

# 3️Load dataset
df = pd.read_csv("IRIS.csv")

# Display full DataFrame
pd.set_option('display.max_rows', None)
print(df)

# Column names
print("\nColumns:", df.columns.tolist())

# Dataset info
df.info()

# Statistical summary
print("\nStatistical Summary:\n", df.describe())

# Class distribution
print("\nSpecies distribution:\n", df['species'].value_counts())

# 4️Data Visualization
sns.pairplot(df, hue='species', palette='Set1')
plt.show()

plt.figure(figsize=(8,6))
numeric_cols = df.select_dtypes(include='number')
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# 5️Split data into features and target
X = df.drop('species', axis=1)
y = df['species']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Decision Tree Classifier
# -------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

# Test results
test_results = X_test.copy()
test_results['Actual'] = y_test.values
test_results['Predicted'] = y_pred
print("\nDecision Tree Test Results:\n", test_results)

# Accuracy
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred, labels=df['species'].unique())
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=df['species'].unique(), yticklabels=df['species'].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Decision Tree visualization
plt.figure(figsize=(12,8))
plot_tree(dt_model, feature_names=X.columns, class_names=df['species'].unique(), filled=True)
plt.show()

# -------------------------------
# Logistic Regression
# -------------------------------
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Accuracy
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Classification report
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))

# Confusion matrix heatmap
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=df['species'].unique())
plt.figure(figsize=(6,5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens',
            xticklabels=df['species'].unique(), yticklabels=df['species'].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# -------------------------------
# Bar graph: Actual vs Predicted counts
# -------------------------------
actual_counts = y_test.value_counts()
pred_counts = pd.Series(y_pred).value_counts()
df_counts = pd.DataFrame({'Actual': actual_counts, 'Predicted': pred_counts})
df_counts.plot(kind='bar', figsize=(8,6))
plt.title("Actual vs Predicted Species Count")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.show()

# -------------------------------
# Save test results to CSV
# -------------------------------
test_results.to_csv("iris_test_predictions.csv", index=False)
print("\nTest results saved successfully as 'iris_test_predictions.csv'")

# Download CSV in Colab
files.download("iris_test_predictions.csv")
