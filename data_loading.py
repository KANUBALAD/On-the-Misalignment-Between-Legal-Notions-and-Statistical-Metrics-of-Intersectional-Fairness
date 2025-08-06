import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_synthetic(path, model_interactions = None, plot_cov = None):
    df = pd.read_csv(path)
    if model_interactions:
        df.insert(2, 'interaction_sr', df['Gender'] * df['Race'])  # Insert after S and R
    if plot_cov:
        corr_matrix = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap - Synthetic Data")
        plt.savefig('../output_data/plots/synthetic_data.pdf')
    return df



def load_german_data(path, log_transform=None,model_interactions = None, plot_cov = None):
    """
    Loads the german data.
    Parameters:
    Datapath (str): this is the path to where the data is
    Returns:
    pd.DataFrame of the german credit
    """
    df = pd.read_csv(path)
    df['TARGET'] = df['class']
    del df['class']
    gender_dict = {
        "'male single'": "male",
        "'female div/dep/mar'": "female",
        "'male div/sep'": "male",
        "'male mar/wid'": "male"}
    df['personal_status'] = df['personal_status'].map(gender_dict)
    df.rename(columns={'personal_status': 'gender'}, inplace=True)
    dict_class = {"good": 1, "bad": 0} 
    dict_gender = {"female": 0, "male": 1} 
    df["gender"] = df["gender"].map(dict_gender)
    df["TARGET"] = df["TARGET"].map(dict_class)
    df['age'] = df['age'].apply(lambda x: np.where(x > 25, 1, 0))
    if model_interactions:
        df.insert(2, 'interaction_sr', df['gender'] * df['age'])  # Insert after S and R
    if plot_cov:
        corr_matrix = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap - German Data")
        plt.savefig('../output_data/plots/german_correlation_heatmap.pdf')
        
    if log_transform:
        df['credit_amount'] = np.log(df['credit_amount'])
        df['duration'] = np.log(df['duration'])
        return df
    else:
        return df
