import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart.csv')

# Separate features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE for oversampling the minority class
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Hyperparameter tuning for RandomForestClassifier
param_grid_rf = {
    'n_estimators': [50, 100, 150], 
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train_res, y_train_res)

# Evaluate RandomForestClassifier
rf_best = grid_rf.best_estimator_
rf_accuracy = accuracy_score(y_test, rf_best.predict(X_test_scaled))
rf_f1 = f1_score(y_test, rf_best.predict(X_test_scaled))
rf_precision = precision_score(y_test, rf_best.predict(X_test_scaled))
rf_roc_auc = roc_auc_score(y_test, rf_best.predict(X_test_scaled))

# Cross-validation for RandomForest
rf_cv = cross_val_score(rf_best, X_train_res, y_train_res, cv=5, scoring='accuracy')
print("Random Forest Cross-validation Accuracy:", rf_cv)

# Hyperparameter tuning for DecisionTreeClassifier
param_grid_dt = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

dt = DecisionTreeClassifier(random_state=42)
grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='accuracy')
grid_dt.fit(X_train_res, y_train_res)

# Evaluate DecisionTreeClassifier
dt_best = grid_dt.best_estimator_
dt_accuracy = accuracy_score(y_test, dt_best.predict(X_test_scaled))
dt_f1 = f1_score(y_test, dt_best.predict(X_test_scaled))
dt_precision = precision_score(y_test, dt_best.predict(X_test_scaled))
dt_roc_auc = roc_auc_score(y_test, dt_best.predict(X_test_scaled))

# Cross-validation for DecisionTree
dt_cv = cross_val_score(dt_best, X_train_res, y_train_res, cv=5, scoring='accuracy')
print("Decision Tree Cross-validation Accuracy:", dt_cv)

# Hyperparameter tuning for LogisticRegression
param_grid_logreg = {
    'C': [0.1, 1, 10], # Regularization strength
    'solver': ['liblinear', 'saga'] # Algorithm used for optimization
}

logreg = LogisticRegression(max_iter=1000)
grid_logreg = GridSearchCV(logreg, param_grid_logreg, cv=5, scoring='accuracy')
grid_logreg.fit(X_train_res, y_train_res)

# Evaluate LogisticRegression
logreg_best = grid_logreg.best_estimator_
logreg_accuracy = accuracy_score(y_test, logreg_best.predict(X_test_scaled))
logreg_f1 = f1_score(y_test, logreg_best.predict(X_test_scaled))
logreg_precision = precision_score(y_test, logreg_best.predict(X_test_scaled))
logreg_roc_auc = roc_auc_score(y_test, logreg_best.predict(X_test_scaled))

# Cross-validation for LogisticRegression
logreg_cv = cross_val_score(logreg_best, X_train_res, y_train_res, cv=5, scoring='accuracy')
print("Logistic Regression Cross-validation Accuracy:", logreg_cv)

# Hyperparameter tuning for KNeighborsClassifier
param_grid_knn = {
    'n_neighbors': [5, 10, 15],
    'weights': ['uniform'],# voting strategy
    'metric': ['manhattan'] # distance calculation method
}

knn = KNeighborsClassifier()
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(X_train_res, y_train_res)

# Evaluate KNeighborsClassifier
knn_best = grid_knn.best_estimator_
knn_accuracy = accuracy_score(y_test, knn_best.predict(X_test_scaled))
knn_f1 = f1_score(y_test, knn_best.predict(X_test_scaled))
knn_precision = precision_score(y_test, knn_best.predict(X_test_scaled))
knn_roc_auc = roc_auc_score(y_test, knn_best.predict(X_test_scaled))

# Cross-validation for KNeighborsClassifier
knn_cv = cross_val_score(knn_best, X_train_res, y_train_res, cv=5, scoring='accuracy')
print("KNN Cross-validation Accuracy:", knn_cv)

# Print the evaluation results
print("\nRandom Forest:")
print(f"Accuracy: {rf_accuracy}")
print(f"F1 Score: {rf_f1}")
print(f"Precision: {rf_precision}")
print(f"ROC AUC: {rf_roc_auc}")

print("\nDecision Tree:")
print(f"Accuracy: {dt_accuracy}")
print(f"F1 Score: {dt_f1}")
print(f"Precision: {dt_precision}")
print(f"ROC AUC: {dt_roc_auc}")

print("\nLogistic Regression:")
print(f"Accuracy: {logreg_accuracy}")
print(f"F1 Score: {logreg_f1}")
print(f"Precision: {logreg_precision}")
print(f"ROC AUC: {logreg_roc_auc}")

print("\nKNN:")
print(f"Accuracy: {knn_accuracy}")
print(f"F1 Score: {knn_f1}")
print(f"Precision: {knn_precision}")
print(f"ROC AUC: {knn_roc_auc}")

# Confusion Matrix Visualization
models = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'KNN']
models_best = [rf_best, dt_best, logreg_best, knn_best]

for model, model_name in zip(models_best, models):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
# After calculating accuracy, F1 score, precision, and ROC AUC for each model, 
# add this code to generate the comparison graph:

# Store the evaluation metrics for each model in lists
models = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'KNN']
accuracies = [rf_accuracy, dt_accuracy, logreg_accuracy, knn_accuracy]
f1_scores = [rf_f1, dt_f1, logreg_f1, knn_f1]
precisions = [rf_precision, dt_precision, logreg_precision, knn_precision]
roc_auc_scores = [rf_roc_auc, dt_roc_auc, logreg_roc_auc, knn_roc_auc]

# Create a DataFrame for the metrics
metrics_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'F1 Score': f1_scores,
    'Precision': precisions,
    'ROC AUC': roc_auc_scores
})

# Plotting the comparison graph
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each metric
metrics_df.plot(x='Model', y=['Accuracy', 'F1 Score', 'Precision', 'ROC AUC'], kind='bar', ax=ax, 
                color=['skyblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow'], width=0.8)

# Set the title and labels
ax.set_title('Model Comparison: Accuracy, F1 Score, Precision, ROC AUC', fontsize=14)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1)  # Set the y-axis limit to [0, 1] for better visualization
plt.xticks(rotation=45)  # Rotate model names for better readability
plt.tight_layout()

# Show the plot
plt.show()
