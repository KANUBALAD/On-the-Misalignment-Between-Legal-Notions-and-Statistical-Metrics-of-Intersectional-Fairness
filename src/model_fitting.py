
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix



    
def evaluate_model(y_true, probabilities, predictions):
    """
    Evaluate the model's performance and return metrics.
    """
    accuracy = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    roc_auc = roc_auc_score(y_true, probabilities)
    confusion = confusion_matrix(y_true, predictions)
    metrics = {
        "accuracy": np.round(accuracy,decimals=3),
        "f1_score": np.round(f1,decimals=3),
        "roc_auc_score": np.round(roc_auc,decimals=3),
        "confusion_matrix": confusion
    }
    return metrics

def logistic_model(df, get_summary=False, split_data=True, random_seed=None):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed)
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


    
    



