#  House Price Prediction Using ML
 ## Step 1: Data Preparation
 ### Data Loading
  
 *import pandas as pd *

*#Load the dataset into a Pandas DataFrame *
*data = { *
    *'Id': [0, 1, 2, 3, 4, 5] *
    *MSSubClass: [60, 20, 60, 70, 60, 50] *
    *'MSZoning': ['RL', 'RL', 'RL', 'RL', 'RL', 'RL'], *
    *'LotArea': [8450, 9600, 11250, 9550, 14260, 14115], *
    *'LotConfig': ['Inside', 'FR2', 'Inside', 'Corner', *'FR2', 'Inside'], *
    *'BldgType': ['1Fam', '1Fam', '1Fam', '1Fam', '1Fam', '1Fam'], *
    *'OverallCond': [5, 8, 5, 5, 5, 5], *
    *'YearBuilt': [2003, 1976, 2001, 1915, 2000, 1993], *
    *YearRemodAdd': [2003, 1976, 2002, 1970, 2000, 1995], *
    *'Exterior1st': ['VinylSd', 'MetalSd', 'VinylSd', 'Wd Sdng', 'VinylSd', 'VinylSd'], *
    *'BsmtFinSF2': [0, 0, 0, 0, 0, 0], *
    *'TotalBsmtSF': [856, 1262, 920, 756, 1145, 796], *
    *'SalePrice': [208500, 181500, 223500, 140000, 250000, 143000] *
*} *
*df = pd.DataFrame(data) *



 ## Step 2: Data Visualization

*import matplotlib.pyplot as plt *

*#Visualize features using histograms *
*df.hist(figsize=(10, 10)) *
*plt.show() *

*#Visualize correlation between features *
*import seaborn as sns *

*correlation_matrix = df.corr() *
*sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm') *
*plt.title('Correlation Matrix') *
*plt.show() *



## Step 3:  Feature Analysis
### Correlation Analysis

# Correlation with target variable (SalePrice)
*correlation_with_sale_price = correlation_matrix['SalePrice'].sort_values(ascending=False) *
*print(correlation_with_sale_price) *


## Step 4: Building the Machine Learning Model
 
*from sklearn.model_selection import train_test_split *
*from sklearn.linear_model import LinearRegression *
*from sklearn.metrics import mean_squared_error, r2_score *

*#Splitting the data into train and test sets *
*X = df.drop(['Id', 'SalePrice'], axis=1) *
*y = df['SalePrice'] *
*X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) *

*#Building the linear regression model *
*model = LinearRegression() *
*model.fit(X_train, y_train) *

*#Making predictions *
*y_pred = model.predict(X_test) *

*#Model evaluation *
*mse = mean_squared_error(y_test, y_pred) *
*r2 = r2_score(y_test, y_pred) *
*print("Mean Squared Error (MSE):", mse) *
*print("R-squared (R2):", r2) *

## Step 5: Model Evaluation and Fine-Tuning

*Calculate metrics like Mean Squared Error (MSE) and R-squared: *

*from sklearn.metrics import mean_squared_error, r2_score *

*mse = mean_squared_error(y_test, y_pred) *
*r2 = r2_score(y_test, y_pred) *

*print("Mean Squared Error:", mse) *
*print("R-squared:", r2) *

### Model Evaluation:

Mean Squared Error (MSE):

a. The Mean Squared Error (MSE) measures the average squared difference between the actual values (SalePrice) and the predicted values by the model.
b. MSE is calculated by taking the average of the squared differences between the actual and predicted values.
c. A lower MSE indicates that the model's predictions are closer to the actual values, implying better performance.

R-squared (R2):

a. R-squared (R2) is a statistical measure that represents the proportion of the variance in the dependent variable (SalePrice) that is predictable from the independent variables.
b. R2 ranges from 0 to 1, where 1 indicates a perfect fit, and 0 indicates that the model does not explain any of the variability of the dependent variable around its mean.
c. A higher R2 value suggests that the model fits the data better, explaining more of the variance.

     *mse_nn = mean_squared_error(y_test, predictions_nn)*
     *print(f'Mean Squared Error (Neural Network): {mse_nn}')*

## Fine-Tuning:

Hyperparameter Tuning:

a. Hyperparameters are parameters that are not directly learned by the model but affect the learning process.
b. In linear regression, there are no hyperparameters to tune explicitly, but in more complex models like ensemble methods or neural networks, tuning hyperparameters such as learning rate, regularization strength, or number of estimators can significantly impact model performance.

     *from sklearn.model_selection import GridSearchCV*

     *param_grid = {*
     *'n_estimators': [50, 100, 200],*
     *max_depth': [None, 10, 20],*
     *'min_samples_split': [2, 5, 10],*
     }*

     *grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')*
     *grid_search.fit(X_train, y_train)*

     *best_rf_model = grid_search.best_estimator_*
     *best_predictions_rf = best_rf_model.predict(X_test)*
     *best_mse_rf = mean_squared_error(y_test, best_predictions_rf)*
     *print(f"Best Random Forest MSE: {best_mse_rf}")*

## Feature Engineering:

a. Creation and modification of the feature matrix data,Feature Engineering is quite important and quite a cyclic process, we want to input a feature matrix that will help teach a model something useful.
b. We want to make sure we feed the model data that is most relevant to the prediction of a target variable, perhaps as less overlapping as possible as well.
c. Features with very high correlation teach a model similar things, multiple times, maybe consider combing them and dropping the others.

## Model Selection:

a. While linear regression is a simple and interpretable model, it may not capture complex relationships in the data effectively.
b. Exploring other regression algorithms such as Ridge Regression, Lasso Regression, Decision Trees, or Random Forests could lead to better performance, depending on the characteristics of the dataset.

## Cross-Validation:

a. Cross-validation is a technique used to assess the generalization performance of a model by splitting the data into multiple subsets.
b. Techniques like k-fold cross-validation or leave-one-out cross-validation can provide more reliable estimates of the model's performance and help prevent overfitting.

## Conclusion:

Model evaluation and fine-tuning are iterative processes aimed at improving the performance of machine learning models.
By evaluating metrics like MSE and R-squared, and exploring different techniques for fine-tuning, we can develop more accurate models that better capture the underlying patterns in the data.

