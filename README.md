# Credit-Scoring-Project-using-Machine-Learning
This document explains the Python code for building a credit scoring model using Logistic Regression. The model predicts whether a loan applicant is a good credit risk (will repay the loan) or a bad credit risk (will default on the loan).
1. Libraries and Data Loading
The code imports necessary libraries:
• pandas: Data manipulation
• numpy: Numerical operations
• sklearn.model_selection: Train-test split
• sklearn.preprocessing: Data scaling
• sklearn.metrics: Model evaluation metrics
• sklearn.linear_model: Logistic regression model
• joblib: Saves models for later use
It then reads the credit scoring dataset from an Excel file named
"a_Dataset_CreditScoring.xlsx" and displays its shape (number of rows and columns) and
the first few rows (using head()) for initial exploration.
2. Data Preprocessing
• The code drops the 'ID' column as it likely doesn't contribute to predicting credit
risk.
• It checks for missing values using isna().sum().
• Missing values are filled with the mean value of each feature in the dataset using
fillna(dataset.mean()).
• The target variable (denoting good or bad credit risk) is separated from the
features using .iloc.
o y: Represents the target variable.
o X: Represents the features used for prediction.
3. Train-Test Split
The code splits the data into training and testing sets using train_test_split:
• X_train: Training data features for model training.
• X_test: Testing data features for model evaluation.
• y_train: Training data target labels for model training.
• y_test: Testing data target labels for model evaluation.
The test size is set to 20% (test_size=0.2) using a random seed (random_state=0) for
reproducibility.
4. Feature Scaling
The code applies StandardScaler to standardize the features in the training set
(X_train). Standardization scales features to have a mean of 0 and a standard deviation of
1. This can improve the performance of some machine learning models. The scaler is then
used to transform the features in the testing set (X_test) using the same parameters
learned from the training data.
5. Model Training and Saving
• A Logistic Regression classifier is created (classifier).
• The model is trained on the training data (classifier.fit(X_train,
y_train))
• The trained model is saved using joblib.dump for later use.
6. Prediction and Evaluation
• The model makes predictions on the testing data (y_pred =
classifier.predict(X_test))
• The confusion matrix and accuracy score are calculated to evaluate the model's
performance on the testing data.
o The confusion matrix shows how many data points were correctly or
incorrectly classified.
o The accuracy score is the proportion of predictions that were correct.
7. Generating Prediction Probabilities
• The model predicts probabilities of belonging to each class
(classifier.predict_proba(X_test)) for the test data.
• This results in a NumPy array with two columns, likely representing probabilities
for good and bad credit risk.
• These probabilities are stored in a DataFrame named df_prediction_prob.
8. Combining Results and Saving
• Three DataFrames are created:
o df_test_dataset: Contains the actual target values (ground truth) for the
test data.
o df_prediction_prob: Contains the predicted probabilities for each data
point.
o df_prediction_target: Contains the predicted target labels.
• These DataFrames are combined using pd.concat to create a single DataFrame
dfx that includes the actual outcomes, predicted probabilities, and predicted target
labels.
• Finally, dfx is saved as a CSV file named "Copy of a_Dataset_CreditScoring.xlsx".
The .head() method displays the first few rows of this combined DataFrame.
