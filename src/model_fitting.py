

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


    
def evaluate_model(y_true, probabilities, predictions):
    """
    Evaluate the model's performance and return metrics.
    """
    accuracy = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    roc_auc = roc_auc_score(y_true, probabilities)
    # confusion = confusion_matrix(y_true, predictions)
    metrics = {
        "accuracy": np.round(accuracy,decimals=3),
        "f1_score": np.round(f1,decimals=3),
        "roc_auc_score": np.round(roc_auc,decimals=3)
        # "confusion_matrix": confusion
    }
    return metrics

def logistic_model(df, get_summary=False, split_data=True):
    """
    Fits a logistic regression model, makes predictions, and evaluates performance.

    Parameters:
    - df: DataFrame containing the data. The last column is assumed to be the target variable.
    - get_summary: If True, prints the summary of the logistic regression model.
    - split_data: If True, splits the data into training and testing sets.

    Returns:
    - predictions: Binary predictions.
    - probabilities: Predicted probabilities.
    - metrics: Evaluation metrics.
    """
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1] 

    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=111)
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
        logit_model = sm.Logit(y_train, X_train)
        logit_results = logit_model.fit()
        probabilities = logit_results.predict(X_test)  
        predictions = (probabilities >= 0.5).astype(int)
        metrics = evaluate_model(y_test, probabilities, predictions)
        
        result_df = X_test.copy()
        result_df['y_true'] = y_test
        result_df['y_hat'] = predictions

    else:
        X_with_constant = sm.add_constant(X)
        logit_model = sm.Logit(y, X_with_constant)
        logit_results = logit_model.fit()
        probabilities = logit_results.predict(X_with_constant)  
        predictions = (probabilities >= 0.5).astype(int) 
        metrics = evaluate_model(y, probabilities, predictions)
        result_df = X_with_constant.copy()
        result_df['y_true'] = y
        result_df['y_hat'] = predictions

    if get_summary:
        print(logit_results.summary())
        
    return result_df, metrics

    # return predictions, probabilities, metrics, result_df

    
    





# Note for myself, I had these fittings when I was estimating the statisical significance. I will come back to them later.
def function_ols_fitting(data_name, get_model_summary = False, scenario=None):
    """
    Fit OLS models for continous dependent variables, make predictions, and evaluate performance.
    """
    features_independet = data_name['S_and_X'] #X independent features
    target_z = data_name['Z_data'] #y
    features_independet = sm.add_constant(features_independet)
    models = {}
    performance = {}
    preds = {}
    for feature in target_z.columns:
        print(f"Fitting OLS model for target for {scenario}: {feature}")
        ols_model = sm.OLS(target_z[feature], features_independet)
        ols_results = ols_model.fit()
        models[feature] = ols_results
        if get_model_summary:
            print(ols_results.summary())
        preds[feature] = ols_results.predict(features_independet)
        mse = mean_squared_error(target_z[feature], preds[feature])
        r2 = r2_score(target_z[feature], preds[feature])
        performance[feature] = {'MSE': mse, 'R2': r2}
        
    print(f"The summary results are: {performance}")
    return models, preds

def function_logistic_fitting(data_name, get_model_summary = False, scenario=None):
    """
    Fit a logistic regression model for a single binary dependent variable, make predictions, evaluate performance, and present the model summary.
    """
    features_independet = data_name['S_and_X']
    # features_independet = data_name['SXZ'] # fitting to all S, X, and Z
    target_y = data_name['Y_data'] 
    features_independet = sm.add_constant(features_independet)
    print(f"Fitting Logistic Regression model for {scenario}")
    
    # Fit the logistic regression model using statsmodels
    logit_model = sm.Logit(target_y, features_independet)
    logit_results = logit_model.fit()
    if get_model_summary:
        print(logit_results.summary())
    predictions = logit_results.predict(features_independet) >= 0.5  # Convert probabilities to binary predictions
    probabilities = logit_results.predict(features_independet)  # Probabilities for ROC AUC
    
    # Evaluate performance
    accuracy = accuracy_score(target_y, predictions)
    f1 = f1_score(target_y, predictions)
    roc_auc = roc_auc_score(target_y, probabilities)
    cm = confusion_matrix(target_y, predictions)
    
    performance = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Confusion Matrix': cm.tolist()  
}
    print(f"Performance: {performance}")
    return logit_results, probabilities
    