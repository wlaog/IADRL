import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load your data (assuming it’s saved as 'data.csv' in the same directory)
data = pd.read_csv('env\SVRBelief\exp_data.csv')

# Define feature matrix X and target vector y
X = data[['xcar', 'vcar', 'acar', 'ttccar', 'xped', 'vped', 'aped', 'ttcped', 'roadtype']]
y = data['belief']

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize traditional models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVR': SVR(kernel='rbf')
}

# Add a deep learning model (MLP) to the models dictionary
def build_mlp():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Single output for regression
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Initialize an empty dictionary to store results
results = {}

# Train and evaluate each traditional model
for name, model in models.items():
    # Cross-validation to get an estimate of model performance
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2, 'CV MSE': -cv_scores.mean()}
    print(f"{name}: MSE={mse:.4f}, R2={r2:.4f}, CV MSE={-cv_scores.mean():.4f}")

# Train and evaluate the MLP (Deep Learning) model
mlp_model = build_mlp()
mlp_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
y_pred_mlp = mlp_model.predict(X_test).flatten()

# Calculate performance metrics for the deep learning model
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)
results['Deep Learning (MLP)'] = {'MSE': mse_mlp, 'R2': r2_mlp}
print(f"Deep Learning (MLP): MSE={mse_mlp:.4f}, R2={r2_mlp:.4f}")

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results).T

# Plotting the results
plt.figure(figsize=(14, 6))

# MSE and R2 comparison
plt.subplot(1, 2, 1)
sns.barplot(x=results_df.index, y="MSE", data=results_df)
plt.title("Mean Squared Error (MSE) Comparison")
plt.ylabel("MSE")
plt.xlabel("Model")

plt.subplot(1, 2, 2)
sns.barplot(x=results_df.index, y="R2", data=results_df)
plt.title("R-Squared (R2) Comparison")
plt.ylabel("R2")
plt.xlabel("Model")

plt.tight_layout()
plt.show()

# Plot true vs predicted values for the best model (choose the one with the highest R2 score)
best_model_name = results_df['R2'].idxmax()

if best_model_name == 'Deep Learning (MLP)':
    best_model = build_mlp()
    best_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
    y_pred_best = best_model.predict(X_test).flatten()
else:
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
plt.xlabel("True Belief Values")
plt.ylabel("Predicted Belief Values")
plt.title(f"True vs Predicted Belief Values - {best_model_name}")
plt.show()