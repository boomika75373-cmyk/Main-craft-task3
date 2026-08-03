import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# Step 2: Load and Prepare Dataset
# ==========================================
data = fetch_california_housing(as_frame=True)
df = pd.concat([data.data, data.target.rename("HousePrice")], axis=1)

# Separate features and target
X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]

# ==========================================
# Step 3: Feature Scaling
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# Step 4: Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================================
# Step 5: Detect Overfitting (Train vs Test Performance)
# ==========================================
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)

train_pred = tree.predict(X_train)
test_pred = tree.predict(X_test)

train_rmse = mean_squared_error(y_train, train_pred, squared=False)
test_rmse = mean_squared_error(y_test, test_pred, squared=False)

print(f"--- Step 5: Overfitting Detection ---")
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}\n")

# ==========================================
# Step 6: Cross-Validation (Reliable Evaluation)
# ==========================================
cv_scores = cross_val_score(
    tree, X_scaled, y,
    scoring="neg_root_mean_squared_error",
    cv=5
)

cv_rmse = -cv_scores.mean()
print(f"--- Step 6: Cross-Validation ---")
print(f"Cross-Validation RMSE: {cv_rmse}\n")

# ==========================================
# Step 7: Hyperparameter Tuning Using GridSearchCV
# ==========================================
param_grid = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5
)

grid.fit(X_train, y_train)
print(f"--- Step 7: Hyperparameter Tuning ---")
print("Best Parameters:", grid.best_params_, "\n")

# ==========================================
# Step 8: Evaluate Optimized Model
# ==========================================
best_tree = grid.best_estimator_
y_pred_tree = best_tree.predict(X_test)

opt_rmse = mean_squared_error(y_test, y_pred_tree, squared=False)
opt_r2 = r2_score(y_test, y_pred_tree)

print(f"--- Step 8: Optimized Model Evaluation ---")
print(f"Optimized Tree RMSE: {opt_rmse}")
print(f"Optimized Tree R2 Score: {opt_r2}\n")

# ==========================================
# Optional Baseline Models Evaluation (For Comparison)
# ==========================================
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_pred, squared=False)
lr_r2 = r2_score(y_test, lr_pred)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_rmse = mean_squared_error(y_test, ridge_pred, squared=False)
ridge_r2 = r2_score(y_test, ridge_pred)

# ==========================================
# Step 9: Model Comparison Summary Table
# ==========================================
results = {
    "Model": ["Linear Regression", "Ridge Regression", "Tuned Decision Tree"],
    "RMSE": [lr_rmse, ridge_rmse, opt_rmse],
    "R2 Score": [lr_r2, ridge_r2, opt_r2]
}

comparison_df = pd.DataFrame(results)
print(f"--- Step 9: Model Comparison Summary ---")
print(comparison_df)

