""" This script implements and compares three different regression models:
1. Decision Tree Regressor
2. XGBoost Regressor
3. Neural Network
for predicting restaurant ratings based on various features. """

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold

# Load dataset
data = pd.read_csv("/content/Dataset .csv")

# Exploratory Data Analysis
# Check dimension of dataset
data.shape  # Shows number of rows and columns

# View the first 10 rows of a dataset
data.head(10)

# Get statistical summary of numerical columns
data.describe()

# Check for duplicate rows
data.duplicated().sum()

# View information about the dataset (data types, null values)
data.info()

# Preprocess Data
# Drop columns that are not useful for modeling
data = data.drop(columns=[
    'Restaurant ID', 'Restaurant Name', 'Address', 'Locality Verbose',
    'Longitude', 'Latitude', 'Rating color', 'Rating text'
])

# Check for null values
data.isnull().sum()

# Fill null values with 'Unknown'
data.fillna('Unknown', inplace=True)

# Feature engineering - count number of cuisines for each restaurant
data['Number of Cuisines'] = data['Cuisines'].apply(lambda x: len(x.split(',')) if 'Cuisines' in data.columns else 0

# Check new shape after feature engineering
data.shape
data.head(10)

# Plot distribution of target variable (Aggregate rating)
plt.figure(figsize=(8, 6))
sns.histplot(data['Aggregate rating'], bins=30, kde=True)
plt.title("Distribution of Aggregate Rating")
plt.xlabel("Aggregate Rating")
plt.ylabel("Frequency")
plt.show()

# Check for null values again
data.isnull().sum()
data.info()

# Data Encoding Section

# Binary encoding for binary columns (Yes/No -> 1/0)
binary_columns = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
data[binary_columns] = data[binary_columns].apply(lambda x: x.map({'Yes': 1, 'No': 0}))

# One-hot encoding for nominal columns (City, Currency)
data = pd.get_dummies(data, columns=['City', 'Currency'], drop_first=True)

# Frequency encoding for high-cardinality columns (Locality, Cuisines)
data['Locality_Frequency'] = data['Locality'].map(data['Locality'].value_counts(normalize=True))
data['Cuisines_Frequency'] = data['Cuisines'].map(data['Cuisines'].value_counts(normalize=True))

# Drop original high-cardinality columns after encoding
data = data.drop(columns=['Locality', 'Cuisines'])

data.head(10)

# Feature selection
X = data.drop(columns=['Aggregate rating'])
y = data['Aggregate rating']

# Remove low-variance features (features that don't change much)
selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)

# Split data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Decision Tree Regressor

# Define the model
dtr_model = DecisionTreeRegressor(random_state=42)

# Define hyperparameters to tune
parametre_grid = {
    'max_depth': [None, 10, 20, 30],  # Maximum depth of tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]     # Minimum samples required at a leaf node
}

# Perform GridSearchCV to find best hyperparameters
grid_search = GridSearchCV(estimator=dtr_model, param_grid=parametre_grid, 
                          cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters found
print("Best Parameters for Decision Tree:", grid_search.best_params_)

# Evaluate the best model
best_dtr_model = grid_search.best_estimator_
y_pred_dtr = best_dtr_model.predict(X_test)
print("Decision Tree MSE:", mean_squared_error(y_test, y_pred_dtr))
print("Decision Tree R²:", r2_score(y_test, y_pred_dtr))

# Get feature importances from the decision tree
dtr_importance = best_dtr_model.feature_importances_
feature_names = data.drop(columns=['Aggregate rating']).columns
selected_feature_names = feature_names[selector.get_support(indices=True)]
dtr_feature_importance_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': dtr_importance})
print("Decision Tree Feature Importance:")
print(dtr_feature_importance_df.sort_values(by='Importance', ascending=False))

# Model 2: XGBoost Regressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define the model
xgb_model = XGBRegressor(random_state=42)

# Define hyperparameters to tune (using random distributions)
param_dist = {
    'n_estimators': randint(100, 500),        # Number of trees
    'max_depth': randint(3, 10),              # Maximum tree depth
    'learning_rate': uniform(0.01, 0.3),      # Learning rate
    'subsample': uniform(0.6, 0.4),           # Subsample ratio
    'colsample_bytree': uniform(0.6, 0.4)     # Column subsample ratio
}

# Perform RandomizedSearchCV (faster than GridSearch for large parameter spaces)
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, 
                                 n_iter=10, cv=5, scoring='neg_mean_squared_error', 
                                 random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Print best parameters found
print("Best Parameters for XGBoost:", random_search.best_params_)

# Evaluate the best model
best_xgb_model = random_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)
print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))
print("XGBoost R²:", r2_score(y_test, y_pred_xgb))

# Get feature importances from XGBoost
xgb_importance = best_xgb_model.feature_importances_
xgb_feature_importance_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': xgb_importance})
print("XGBoost Feature Importance:")
print(xgb_feature_importance_df.sort_values(by='Importance', ascending=False))

# Model 3: Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network architecture
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Input layer
    Dense(32, activation='relu'),                                         # Hidden layer
    Dense(1)                                                              # Output layer
])

# Compile the model with Adam optimizer and MSE loss
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
nn_model.fit(X_train_scaled, y_train, epochs=20, batch_size=20, validation_split=0.2, verbose=1)

# Evaluate the model
y_pred_nn = nn_model.predict(X_test_scaled)
print("Neural Network MSE:", mean_squared_error(y_test, y_pred_nn))
print("Neural Network R²:", r2_score(y_test, y_pred_nn))

# Visualization Section

# Create subplots for actual vs predicted ratings
plt.figure(figsize=(15, 5))

# Decision Tree
plt.subplot(1, 3, 1)
sns.scatterplot(x=y_test, y=y_pred_dtr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect prediction line
plt.title("Decision Tree: Actual vs Predicted")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")

# XGBoost
plt.subplot(1, 3, 2)
sns.scatterplot(x=y_test, y=y_pred_xgb)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("XGBoost: Actual vs Predicted")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")

# Neural Network
plt.subplot(1, 3, 3)
sns.scatterplot(x=y_test, y=y_pred_nn.flatten())
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Neural Network: Actual vs Predicted")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")

plt.tight_layout()
plt.show()

# Residual Plots
plt.figure(figsize=(15, 5))

# Decision Tree residuals
plt.subplot(1, 3, 1)
sns.residplot(x=y_test, y=y_pred_dtr, lowess=True, line_kws={'color': 'red'})
plt.title("Decision Tree: Residual Plot")
plt.xlabel("Actual Rating")
plt.ylabel("Residuals")

# XGBoost residuals
plt.subplot(1, 3, 2)
sns.residplot(x=y_test, y=y_pred_xgb, lowess=True, line_kws={'color': 'red'})
plt.title("XGBoost: Residual Plot")
plt.xlabel("Actual Rating")
plt.ylabel("Residuals")

# Neural Network residuals
plt.subplot(1, 3, 3)
sns.residplot(x=y_test, y=y_pred_nn, lowess=True, line_kws={'color': 'red'})
plt.title("Neural Network: Residual Plot")
plt.xlabel("Actual Rating")
plt.ylabel("Residuals")

# Feature Importance Visualization
# Decision Tree feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=dtr_feature_importance_df['Importance'], y=dtr_feature_importance_df['Feature'], palette='viridis')
plt.title("Decision Tree: Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# XGBoost feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_feature_importance_df['Importance'], y=xgb_feature_importance_df['Feature'], palette='viridis')
plt.title("XGBoost: Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Model Performance Comparison
# Create a DataFrame for model performance metrics
performance_df = pd.DataFrame({
    'Model': ['Decision Tree', 'XGBoost', 'Neural Network'],
    'MSE': [mean_squared_error(y_test, y_pred_dtr),
             mean_squared_error(y_test, y_pred_xgb),
             mean_squared_error(y_test, y_pred_nn)],
    'R²': [r2_score(y_test, y_pred_dtr),
            r2_score(y_test, y_pred_xgb),
            r2_score(y_test, y_pred_nn)]
})

# Plot comparison of MSE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=performance_df, palette='Blues')
plt.title("Model Comparison: MSE")
plt.xlabel("Model")
plt.ylabel("MSE")

# Plot comparison of R²
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R²', data=performance_df, palette='Greens')
plt.title("Model Comparison: R²")
plt.xlabel("Model")
plt.ylabel("R²")

plt.tight_layout()
plt.show()

# Logging Results
import logging
import os

# Verify directory and configure logging
print("Current directory:", os.getcwd())
logging.basicConfig(
    filename='result.log',
    filemode='w',  # Overwrite existing log
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Important for notebooks
)

# Test logging
logging.info("This should appear in result.log")

# Verify file creation
if os.path.exists('result.log'):
    print("Log file created successfully!")
else:
    print("Still missing - check permissions.")

# Log model performance results
# Decision Tree
mse_dtr = mean_squared_error(y_test, y_pred_dtr)
r2_dtr = r2_score(y_test, y_pred_dtr)
logging.info(f"Decision Tree MSE: {mse_dtr}, R²: {r2_dtr}")

# XGBoost
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
logging.info(f"XGBoost MSE: {mse_xgb}, R²: {r2_xgb}")

# Neural Network
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)
logging.info(f"Neural Network MSE: {mse_nn}, R²: {r2_nn}")
