import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def log_transform(x):
    """Applies the natural logarithm to the input value"""
    return np.log(x)


def preprocess_X(covariates_X):
    """
    Preprocesses the covariates data by standardizing numerical columns and one-hot encoding categorical columns.

    Parameters:
    covariates_X (pd.DataFrame): The covariates data.

    Returns:
    pd.DataFrame: The preprocessed covariates X.
    """
    numerical_features = covariates_X.select_dtypes(include=[float]).columns
    categorical_features = covariates_X.select_dtypes(include=[int, object]).columns
    
    
    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)])
    
    X_transformed = preprocessor.fit_transform(covariates_X)
    feature_names = numerical_features.tolist() + preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist()
    X_transformed_df = pd.DataFrame(X_transformed, index=covariates_X.index, columns=feature_names)
    return X_transformed_df, preprocessor

def preprocess_Z(treatment_Z, data_name=None):
    """
    Preprocesses the treatment data by standardizing numerical columns and applying log transformation if necessary.

    Parameters:
    treatment_Z (pd.DataFrame): The treatment data.
    data_name (str): The name of the dataset for specific processing.

    Returns:
    dict: The standardized treatment data and its means and standard deviations.
    """
    if data_name != 'german':
        treatment_Z = treatment_Z.apply(log_transform)
        
    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, treatment_Z.columns)])
    
    Z_scaled = preprocessor.fit_transform(treatment_Z)
    Z_scaled_df = pd.DataFrame(Z_scaled, index=treatment_Z.index, columns=treatment_Z.columns)
    Zstandardized = {
        'Zmeans': treatment_Z.mean(),
        'Z_sigmas': treatment_Z.std(),
        'Z_scaled': Z_scaled_df}
    
    return Zstandardized

def preprocess_XZ(covariates_X=None, treatment_Z=None, data_name=None):
    """
    Preprocesses the covariates and treatment data by calling the appropriate preprocessing functions.

    Parameters:
    covariates_X (pd.DataFrame): The covariates data.
    treatment_Z (pd.DataFrame): The treatment data.
    data_name (str): The name of the dataset for specific processing.

    Returns:
    pd.DataFrame or dict: The preprocessed covariates X or treatment data Z.
    """
    if covariates_X is not None:
        return preprocess_X(covariates_X)
    elif treatment_Z is not None:
        return preprocess_Z(treatment_Z, data_name)
    
    

def inverse_transform_X(X_transformed, Xpreprocessor):
    """
    Inverse transforms the preprocessed covariates data back to their original values.

    Parameters:
    X_transformed (pd.DataFrame): The preprocessed covariates data.
    preprocessor (ColumnTransformer): The preprocessor object used for transformation.

    Returns:
    pd.DataFrame: The original covariates data.
    """
    # Extract the individual transformers
    numerical_transformer = Xpreprocessor.named_transformers_['num']
    categorical_transformer = Xpreprocessor.named_transformers_['cat']
    
    # Check if numerical transformer has feature_names_in_
    if hasattr(numerical_transformer.named_steps['scaler'], 'feature_names_in_'):
        numerical_feature_names = numerical_transformer.named_steps['scaler'].feature_names_in_
        X_numerical = numerical_transformer.inverse_transform(X_transformed.iloc[:, :len(numerical_feature_names)])
    else:
        numerical_feature_names = []
        X_numerical = np.empty((X_transformed.shape[0], 0))
    
    # Perform inverse transformation for categorical features
    X_categorical = categorical_transformer.inverse_transform(X_transformed.iloc[:, len(numerical_feature_names):])
    categorical_feature_names = Xpreprocessor.transformers_[1][2]
    
    # Combine numerical and categorical features
    X_original = np.concatenate([X_numerical, X_categorical], axis=1)
    feature_names = list(numerical_feature_names) + list(categorical_feature_names)
    X_original_df = pd.DataFrame(X_original, index=X_transformed.index, columns=feature_names)
    
    return X_original_df

    


def preprocess_sensitive_data(data):
    """
    Preprocess data to ensure it's in the correct format.
    Handles both pd.Series (reshaped to DataFrame) and pd.DataFrame.

    Parameters:
    data (pd.Series or pd.DataFrame): The input data to preprocess.
    
    Returns:
    np.ndarray: Processed data as a 2D numpy array.
    """
    if isinstance(data, pd.Series):
        return data.values.reshape(-1, 1)
    elif isinstance(data, pd.DataFrame):
        # Ensure all column names are strings
        data.columns = data.columns.astype(str)
        return data
    else:
        raise ValueError("Input data must be a pandas Series or DataFrame")  

  
