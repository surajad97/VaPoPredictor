# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2
from sklearn.feature_selection import SelectKBest, f_regression

# Import the  Dataset
data = pd.read_csv('/content/Vapor_pressures_rdkit_reduced_trainall.csv')
data.head()

K_data = data.loc[data['Name'] == 'ETHYL CYANOACETATE']
plt.figure(figsize=(15, 7))
plt.plot(K_data['T'], K_data['Pvap'], 'kx', markersize=3)
plt.xlabel('Temperature [K]')
plt.ylabel('Vapour Pressure [kPa]')

"""#### 1b) Perform polynomial regression to predict the vapor pressure from temperature for _ethyl cyanoacetate_. Evaluate the prediction accuracy for varying degree of polynomial and plot the results. (1P)"""

# Extract the relevant columns for regression
X = K_data['T'].values.reshape(-1, 1)
y = K_data['Pvap'].values

# Define the degree of polynomial
degree = 5

# Create polynomial features
polynomial_features = PolynomialFeatures(degree=degree)
X_poly = polynomial_features.fit_transform(X)

# Perform linear regression
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions
y_pred = model.predict(X_poly)

# Calculate R-squared score
r2 = r2_score(y, y_pred)

# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(K_data['T'], K_data['Pvap'], 'kx', markersize=3, label='Actual')
plt.plot(K_data['T'], y_pred, 'r-', label='Predicted')
plt.xlabel('Temperature [K]')
plt.ylabel('Vapour Pressure [kPa]')
plt.title('Polynomial Regression - Actual vs. Predicted')
plt.legend()
plt.grid(True)
plt.show()

# Print the R-squared score
print("R-squared score:", r2)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Extract the relevant columns for regression
X = K_data['T'].values.reshape(-1, 1)
y = K_data['Pvap'].values

# Initialize lists to store the degree and corresponding R-squared values
degrees = []
r2_scores = []

# Perform polynomial regression for degrees ranging from 1 to 10
for degree in range(1, 11):
    # Create polynomial features
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X)

    # Perform linear regression
    model = LinearRegression()
    model.fit(X_poly, y)

    # Make predictions
    y_pred = model.predict(X_poly)

    # Calculate R-squared score
    r2 = r2_score(y, y_pred)

    # Append degree and R-squared values to lists
    degrees.append(degree)
    r2_scores.append(r2)

# Plot the R-squared values against degree of polynomial
plt.figure(figsize=(10, 6))
plt.plot(degrees, r2_scores, 'bo-')
plt.xlabel('Degree of Polynomial')
plt.ylabel('R-squared Score')
plt.title('Polynomial Regression - Degree vs. R-squared')
plt.grid(True)
plt.show()

"""#### 1c) Predict the vapor pressure of _ethyl cyanoacetate_ with Gaussian Processes trying at least two different kernels. Evaluate the prediction accuracy. Plot the mean (prediction) and standard deviation (uncertainty) of the final posterior. (1P)"""

# Extract the relevant columns for regression
X = K_data['T'].values.reshape(-1, 1)
y = K_data['Pvap'].values

# Define the kernels to try
kernels = [RBF(), Matern()]

# Perform Gaussian Process regression with different kernels
for kernel in kernels:
    # Create Gaussian Process regressor with the specified kernel
    model = GaussianProcessRegressor(kernel=kernel)

    # Fit the model to the data
    model.fit(X, y)

    # Generate test data for evaluation
    X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    # Make predictions on the test data
    y_pred, y_std = model.predict(X_test, return_std=True)

    # Plot the mean (prediction) and standard deviation (uncertainty)
    plt.figure(figsize=(10, 6))
    plt.plot(K_data['T'], K_data['Pvap'], 'kx', markersize=3, label='Actual')
    plt.plot(X_test, y_pred, 'r-', label='Mean')
    plt.fill_between(X_test.flatten(), y_pred - 2 * y_std, y_pred + 2 * y_std, color='gray', alpha=0.3, label='Uncertainty')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Vapour Pressure [kPa]')
    plt.legend()
    plt.show()

"""#### 2a) Now use the entire data set. Add _ln(P)_, the natural log of the vapor pressure _Pvap_, to the data set. Perform validation-testing-splitting based on the chemical species. Predict _ln(P)_ from temperature and the various molecular descriptors in the data set using KNN regression. (1P) Evaluate the prediction accuracy for varying number of neighbors and plot the results. (1P)"""

# Add ln(Pvap) as a new column in the dataset
data['ln_Pvap'] = np.log(data['Pvap'])

X = np.array(data['T']).reshape(-1, 1)
y = np.array(data['ln_Pvap']).reshape(-1, 1)
X_hat = ss.fit_transform(X)
y_hat = ss.fit_transform(y)

# Split the data based on chemical species
X_train, X_test, y_train, y_test = train_test_split(X_hat, y_hat, test_size=0.2, stratify=data['Name'])

# Initialize lists to store results
neighbors = []
mse_scores = []
r2_scores = []

# Perform KNN regression for different numbers of neighbors
for k in range(1, 21):
    # Create KNN regressor with the specified number of neighbors
    model = KNeighborsRegressor(n_neighbors=k)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate mean squared error and R-squared score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Append results to lists
    neighbors.append(k)
    mse_scores.append(mse)
    r2_scores.append(r2)

# Plot the mean squared error and R-squared score for different numbers of neighbors
plt.figure(figsize=(10, 6))
plt.plot(neighbors, mse_scores, 'bo-', label='MSE')
plt.plot(neighbors, r2_scores, 'ro-', label='R-squared')
plt.xlabel('Number of Neighbors')
plt.ylabel('Score')
plt.title('KNN Regression - Number of Neighbors vs. Score')
plt.legend()
plt.grid(True)
plt.show()

"""#### 2b) Perform feature selection with one filter method and one wrapper method with a model of your choice for the prediction of _ln(P)_ from temperature and molecular descriptors. Determine the most important features with both methods. (1P)"""

#Filter method

numerical_columns = data.select_dtypes(include=np.number).columns
X = data[numerical_columns].drop(['ln_Pvap'], axis=1)  # Select the numerical columns except the target variable
y = data['ln_Pvap']

VSmodel_Correlation = SelectKBest(f_regression, k=10).fit(X, y)
input_scores = VSmodel_Correlation.scores_

# find the top ranked inputs
top_k_inputs_Correlation = np.argsort(input_scores)[::-1][:10] + 1#  [::-1] reverses the array returned by argsort() and [:n] gives that last n elements
print(top_k_inputs_Correlation)

# reduce X to only top relevant inputs
X_relevant = VSmodel_Correlation.transform(X)

#Wrapper methods


numerical_columns = data.select_dtypes(include=np.number).columns
X = data[numerical_columns].drop(['ln_Pvap'], axis=1)  # Select the numerical columns except the target variable
y = data['ln_Pvap']

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

model = KNeighborsRegressor(n_neighbors=10)

model.fit(X_train_std, y_train)

sfs_forward = SFS(model,                                #the no of neighnours are fixed before doing SFS
          n_features_to_select=6,       #no of features is hyperparameter
          direction='forward',
          scoring='neg_mean_squared_error',
          n_jobs=-1,
          cv=5)
sfs_forward = sfs_forward.fit(X_train_std, y_train)                             #is used in the part of the project
X_train_std_transformed = sfs_forward.transform(X_train_std)
X_test_std_transformed = sfs_forward.transform(X_test_std)
model_transformed = KNeighborsRegressor(n_neighbors=10)
model_transformed.fit(X_train_std_transformed, y_train)
# print('MSE_test_transformed:', i, MSE(y_test,model_transformed.predict(X_test_std_transformed)))
print('R2_test_transformed:', R2(y_test,model_transformed.predict(X_test_std_transformed)))
selected_features = sfs_forward.get_support()
selected_features.show

"""#### 3a) Predict _ln(P)_ using an ANN. Show training and validation error throughout training (1P)."""

# Select the relevant features for the ANN
X = data[['T']].values
y = data['ln_Pvap'].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

# Scale the features using StandardScaler
scaler = ss()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Create the ANN regressor
model = MLPRegressor()

# Define the hyperparameters grid to search
param_grid = {
    'hidden_layer_sizes': [(2,), (10, 10)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5)

# Train the ANN with grid search
grid_search.fit(X_train_scaled, y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Make predictions with the best model
y_train_pred = best_model.predict(X_train_scaled)
y_val_pred = best_model.predict(X_val_scaled)

# Calculate the mean squared error for training and validation sets
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

# Calculate the R-squared score for training and validation sets
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

# Print the mean squared error and R-squared score
print("Training MSE:", train_mse)
print("Validation MSE:", val_mse)
print("Training R2:", train_r2)
print("Validation R2:", val_r2)

"""#### 3b) Perform hyperparameter selection for the ANN and optimize its performance. (1P) Indicate clearly your final model. (1P for highest performing models)"""



"""#### 4) Perform PCA for the input features; select the PCs which explain 90% of the variance and state the number of components required. Then perform linear regression using the PCs. Evaluate the performance for varying number of PCs. Compare with linear regression on the original data set with and without feature selection, explain and interpret the results. (1P)"""

# Select the input features and target variable
X = data.drop(['ln_Pvap'], axis=1).values
y = data['ln_Pvap'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)

# Determine the number of components that explain 90% of the variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
n_components = np.argmax(cumulative_variance_ratio >= 0.9) + 1

# Select the top n_components principal components
X_train_pca_selected = X_train_pca[:, :n_components]

# Perform linear regression on the selected principal components
model_pca = LinearRegression()
model_pca.fit(X_train_pca_selected, y_train)

# Transform the test data using PCA and select the same number of components
X_test_scaled_pca = pca.transform(X_test_scaled)
X_test_pca_selected = X_test_scaled_pca[:, :n_components]

# Make predictions using the selected principal components
y_pred_pca = model_pca.predict(X_test_pca_selected)

# Evaluate the performance of linear regression using PCA
mse_pca = mean_squared_error(y_test, y_pred_pca)
r2_pca = r2_score(y_test, y_pred_pca)

# Perform linear regression on the original data set
model_orig = LinearRegression()
model_orig.fit(X_train_scaled, y_train)

# Make predictions using the original data set
y_pred_orig = model_orig.predict(X_test_scaled)

# Evaluate the performance of linear regression on the original data set
mse_orig = mean_squared_error(y_test, y_pred_orig)
r2_orig = r2_score(y_test, y_pred_orig)

# Perform linear regression on the original data set with feature selection
model_fs = LinearRegression()
model_fs.fit(X_train_scaled[:, fs_indices], y_train)

# Make predictions using the original data set with feature selection
y_pred_fs = model_fs.predict(X_test_scaled[:, fs_indices])

# Evaluate the performance of linear regression on the original data set with feature selection
mse_fs = mean_squared_error(y_test, y_pred_fs)
r2_fs = r2_score(y_test, y_pred_fs)

# Print the results
print("Performance using PCA:")
print("Number of Principal Components:", n_components)
print("MSE (PCA):", mse_pca)
print("R-squared (PCA):", r2_pca)

print("\nPerformance on the original data set:")
print("MSE (Original):", mse_orig)
print("R-squared (Original):", r2_orig)

print("\nPerformance on the original data set with feature selection:")
print("MSE (Feature Selection):", mse_fs)
print("R-squared (Feature Selection):", r2_fs)