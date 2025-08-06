import numpy as np
import pandas as pd
import yaml
import itertools
from preprocess import preprocess_sensitive_data
from src.data_loader import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_datasets(config_path, interactions_enabled):
    """
    Load datasets based on the configuration and whether interactions are enabled or not.
    
    Parameters:
    config_path (str): Path to the configuration file.
    interactions_enabled (bool): Whether interactions are enabled or not.
    
    Returns:
    dict: Loaded datasets.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config['model_interactions'] = interactions_enabled
    data_loader = DataLoader(config, seed=42)
    
    if config['data_name'] == 'german':
        datasets_ , _, _ = data_loader.load()
    else:
        datasets_ = data_loader.load()
    
    partition = config['use_dataset']
    S_data = datasets_[f"{partition}_S"]
    X_data = datasets_[f"{partition}_X"]
    Z_data = datasets_[f"{partition}_Z"]
    Y_data = datasets_[f"{partition}_Y"]
    
    S_reshaped = preprocess_sensitive_data(S_data)
    S_and_X = pd.concat([pd.DataFrame(S_reshaped), pd.DataFrame(X_data)], axis=1)
    SX_Z = pd.concat([S_and_X, pd.DataFrame(Z_data)], axis=1)
    
    return {
        'S_data': S_data,
        'X_data': X_data,
        'Z_data': Z_data,
        'S_and_X': S_and_X,
        'SXZ':   SX_Z,
        'Y_data':Y_data
    }
    



def get_subgroups(df, sensitive_attributes):
    """
    Returns list of (label, subgroup_df) for every combination of sensitive_attributes.
    """
    unique_vals = {attr: df[attr].dropna().unique() for attr in sensitive_attributes}
    subgroups = []
    for combo in itertools.product(*(unique_vals[attr] for attr in sensitive_attributes)):
        mask = np.ones(len(df), dtype=bool)
        for attr, val in zip(sensitive_attributes, combo):
            mask &= (df[attr] == val)
        sub_df = df[mask]
        subgroups.append((combo, sub_df))
    return subgroups




def standardize_metrics(metrics, data_type = 'dict', scalar_type=None, columns_to_scale=None):
    """
    Standardizes or normalizes metrics using the specified scaler.

    Parameters:
    - metrics: dict or pd.DataFrame
        Dictionary or DataFrame of metrics to scale.
    - data_type: str
        Type of input data ('dict' for dictionary, 'df' for DataFrame).
    - scalar_type: str
        Type of scaler to use ('minmax', 'zscore').
    - columns_to_scale: list of str or None
        List of column names to scale (only applicable if data_type='df').

    Returns:
    - scaled_results: pd.DataFrame
        Scaled metrics as a DataFrame.
    """
    if data_type == 'dict':
        df = pd.DataFrame.from_dict(metrics, orient='index')
    elif data_type == 'df':
        if columns_to_scale is None:
            raise ValueError("When data_type='df', you must specify columns_to_scale.")
        df = metrics[columns_to_scale]
        
    else:
        raise ValueError("data_type must be 'dict' or 'df'.")
        
    # choose scalar 
    if scalar_type == 'MinMax':
        scaler = MinMaxScaler()
    elif scalar_type == 'Zscore':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scalar_type: {scalar_type}")

    scaled_array = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)
    
    if data_type == 'df':
        metrics = metrics.copy()
        metrics[columns_to_scale] = scaled_df
        return metrics

    return scaled_df
    
    # return scaled_df
    



