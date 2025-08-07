   
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import cm

import numpy as np
import pandas as pd
import seaborn as sns   
import os

from tueplots import bundles
bundles.aaai2024()
plt.rcParams.update(bundles.aaai2024())

# Increase the resolution of all the plots below
plt.rcParams.update({"figure.dpi": 800})

sns.set_style("whitegrid")  
sns.set_context("paper", font_scale=1.6)

pd.set_option('display.max_columns', 10)
pd.set_option("display.precision", 5)
 
# sns.set_style("whitegrid")  
# sns.set_context("paper", font_scale=1.3)

 
# def plot_features(data):     
#     rows = 2
#     cols = 4
#     names_cols = ['Gender','Race','Education','Income','Savings','LoanAmount','Duration', 'Y']

#     fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
#     axes = axes.flatten()
#     for i, col in enumerate(names_cols):
#         axes[i].hist(data[col])  
#         axes[i].set_xlabel(col)
#         axes[i].set_title(f"Distribution of {col}")

#     for j in range(len(names_cols), len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     plt.show()
    
def correlation_matrix(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    
    
def get_color_mapping(unique_groups, cmap_name="viridis"):
    """
    Given a list of group names, return a consistent color mapping (dict).
    """
    unique_groups = sorted(unique_groups)  # ensure consistent order
    cmap = get_cmap(cmap_name)
    colors = [cmap(i / len(unique_groups)) for i in range(len(unique_groups))]
    return dict(zip(unique_groups, colors))
    

def thresholding(results_summary, group_by=None, data_name=None):
    """
    Plots and returns average thresholds by either intersectional groups (e.g., Gender_Race)
    or a single sensitive attribute.

    Parameters:
    - results_summary: dict with 'data_df' and 'thresholds'
    - plot_intersection: bool, if True uses intersection of Gender and Race
    - group_by: str, optional column name to group by (e.g., 'Gender', 'Race')

    Returns:
    - DataFrame with average threshold per group
    """
    data_csv = results_summary['data_df'].copy()
    thresholds = results_summary['thresholds']
    data_csv['threshold'] = thresholds
    if group_by is None:
        data_csv['IntersectionalGroup'] = data_csv['Gender'].astype(str) + "_" + data_csv['Race'].astype(str)
        group_col = 'IntersectionalGroup'
    else:
        group_col = group_by
        
    grouped = data_csv.groupby(group_col)['threshold'].mean().reset_index()

    group_labels = sorted(data_csv[group_col].unique())
    color_map = get_color_mapping(group_labels)
    fontsize = 24
    # Match bar colors by group order
    bar_colors = [color_map[g] for g in grouped[group_col]]

    plt.figure(figsize=(8, 4))
    plt.bar(grouped[group_col], grouped['threshold'], color=bar_colors, edgecolor="black")

    
    plt.xlabel(group_col, fontsize=fontsize)
    plt.ylabel(f" Base on Outcome.", fontsize=fontsize)
    # plt.title(f" Thresholds for Outcome.", fontsize=fontsize)
    # plt.title(f"Average Threshold by {group_col}")
    plt.xticks(rotation=45, fontsize=fontsize)
    plt.tight_layout()
    
    save_path = os.path.join('../images', f"{data_name}.pdf")
    plt.savefig(save_path, format="pdf", dpi=800)
    plt.show()


    


def visualize_histograms(data, data_name = None,  measure_bias_col=None, plot_intersectional=True):
    """
    Visualize the distribution of covariates across intersectional groups of Gender and Race.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing at least 'Gender' and 'Race'.
    """
    data = data.copy()  # Avoid modifying original DataFrame
    data['IntersectionalGroup'] = data['Gender'].astype(str) + "_" + data['Race'].astype(str)
    
    # measure_bias_col = 'IntersectionalGroup'
    
    if plot_intersectional :
        measure_bias_col = 'IntersectionalGroup'
    else:
        measure_bias_col = measure_bias_col

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    sns.histplot(data=data, x='Education', kde=True, hue=measure_bias_col, ax=axes[0], palette="coolwarm", alpha=0.7)
    axes[0].set_title(f"Distribution of Education by {measure_bias_col}")
    
    sns.histplot(data=data, x='Income', kde=True, hue=measure_bias_col, ax=axes[1], palette="viridis", alpha=0.7)
    axes[1].set_title(f"Distribution of Income by {measure_bias_col}")
    
    sns.histplot(data=data, x='Savings', kde=True, hue=measure_bias_col, ax=axes[2], palette="coolwarm", alpha=0.7)
    axes[2].set_title(f"Distribution of Savings by {measure_bias_col}")
    
    sns.histplot(data=data, x='LoanAmount', kde=True, hue=measure_bias_col, ax=axes[3], palette="viridis", alpha=0.7)
    axes[3].set_title(f"Distribution of Loan Amount by {measure_bias_col}")
    
    sns.histplot(data=data, x='Duration', kde=True, hue=measure_bias_col, ax=axes[4], palette="coolwarm", alpha=0.7)
    axes[4].set_title(f"Distribution of Duration by {measure_bias_col}")
    
    axes[5].axis('off')
    
    fig.tight_layout()
    # fig.suptitle(f"Histogram of {data_name} dataset", fontsize=18, y=1.02)
    # plt.show()
    fontsize = 24
    group_labels = sorted(data[measure_bias_col].unique())
    color_map = get_color_mapping(group_labels)
    # colors = [color_map[g] for g in group_labels]
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Y', hue=measure_bias_col, data=data, palette=color_map)
    # plt.title(f'Target Variable Distribution by {data_name}', fontsize=16)
    plt.xlabel('Target Variable', fontsize=fontsize)
    plt.ylabel('Count', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    
    save_path = os.path.join('../images', f"ycounts_{data_name}.pdf")
    plt.savefig(save_path, format="pdf", dpi=800)
    plt.show()



def plot_metrics(results, output_file='intersectional_fairness_metrics.pdf', data_type='Generated Data', dpi=800, scaling_method=None):
    """
    Plots metrics as horizontal bar plots with high resolution and clear labels for Overleaf.
    """
    # Convert results to DataFrame
    if scaling_method:
        df = pd.DataFrame(results)
    else:
        df = pd.DataFrame(results).T
    metrics = df.columns  # Metric names
    datasets = df.index  # Dataset names
    y = np.arange(len(datasets))  # Y-axis positions for datasets
    height = 0.10  # Height of each bar
    cmap = cm.get_cmap('viridis', len(metrics))  # Colormap
    colors = [cmap(i) for i in range(len(metrics))]
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Larger figure size for clarity

    for i, metric in enumerate(metrics):
        ax.barh(y + i * height, df[metric], height, label=metric, color=colors[i], edgecolor='black')
    label_fontsize = 26
    # Add labels, title, and legend
    ax.set_ylabel('Datasets', fontsize=label_fontsize, labelpad=15)  # Larger font size and padding
    if scaling_method:
        ax.set_xlabel(f'{scaling_method} Scaled Values', fontsize=26, labelpad=15)
    else:
        ax.set_xlabel('Unscaled Values', fontsize=24, labelpad=15)
        
    ax.set_yticks(y + height * (len(metrics) - 1) / 2)
    ax.set_yticklabels(datasets, fontsize=label_fontsize)
    
    xtick_positions = np.arange(0, 1.1, 0.2)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f'{x:.1f}' for x in xtick_positions], fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    
    ax.legend(fontsize=20, title_fontsize=22, loc='lower right', frameon=True)
    
    # ax.legend(title="Intersectional Metrics", fontsize=20, title_fontsize=22, loc='lower right', frameon=True)

    ax.grid(True, linestyle='--', axis='x', alpha=0.6)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')  # Ensure no clipping
    plt.show()
    # plt.close(fig)  # Close the figure to free memory





def plot_grouped_metrics(results, title='Intersectional Unfairness Metrics for ', output_file='intersectional_fairness_classifier_on_classifier.pdf',
                         data_type='Classifier Predictions', dpi=800, scaling_method=None):
    """
    Plots grouped metrics as horizontal bar plots with high resolution and clear labels for Overleaf.
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    metrics = df.columns.tolist()
    datasets = df.index.tolist()
    n_metrics = len(metrics)
    n_datasets = len(datasets)

    # Bar settings
    height = 0.08 # height of each bar
    group_spacing = 0.3  # extra spacing between dataset groups
    total_height = n_metrics * height + group_spacing
    y = np.arange(n_datasets) * total_height

    # Color mapping
    cmap = cm.get_cmap('viridis', n_metrics)
    colors = [cmap(i) for i in range(n_metrics)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(25, 17))  # Larger figure size for clarity

    for i, metric in enumerate(metrics):
        offsets = y + i * height
        ax.barh(offsets, df[metric], height=height, label=metric.replace("_", " ").title(), color=colors[i], edgecolor='black')
    label_size = 55
    # Set y-axis ticks and labels
    ax.set_yticks(y + (n_metrics - 1) * height / 2)
    ax.set_yticklabels(datasets, fontsize=label_size)

    # Set axis labels and title
    ax.set_ylabel('Datasets', fontsize=label_size, labelpad=15)
    
    if scaling_method:
        ax.set_xlabel(f'{scaling_method} Metric Values', fontsize=label_size, labelpad=15)
    else:
        ax.set_xlabel('Datasets', fontsize=label_size, labelpad=15)

    # Add title
    # ax.set_title(f'{title} {data_type}', fontsize=28, pad=20)

    # Customize tick parameters
    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)

    # Add legend
    ax.legend(fontsize=38, title_fontsize=40, loc='lower right', frameon=True)
    # ax.legend(title="Intersectional Metrics", fontsize=14, title_fontsize=18, loc='lower right', frameon=True)

    # Add gridlines
    ax.grid(True, linestyle='--', axis='x', alpha=0.6)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')  # Ensure no clipping
    plt.show()


def plot_idd_decomposition(results):
    """
    Plots the IDD decomposition (subgroup, additive, intersectional) for each dataset.

    Parameters
    ----------
    results : dict
        Dictionary where keys are dataset names and values are dictionaries with keys
        'idd_subgroups', 'idd_additive', 'idd_intersectional'.
    """
    labels = list(results.keys())
    D_worst_disparity = [results[k]['idd_worst_disparity'] for k in labels]
    D_additive = [results[k]['idd_additive'] for k in labels]
    D_intersectional = [results[k]['idd_intersectional'] for k in labels]

    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar([p - width for p in x], D_worst_disparity, width, label='D_worst_disparity')
    plt.bar(x, D_additive, width, label='D_additive')
    plt.bar([p + width for p in x], D_intersectional, width, label='D_intersectional')

    plt.xticks(x, labels, rotation=30, fontsize=20)
    plt.ylabel('Disparity Value', fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('IDD Decomposition Values', fontsize=25)
    plt.legend()
    plt.tight_layout()
    plt.savefig('idd_decomposition_values.pdf', dpi=800, bbox_inches='tight')  # Save with high resolution
    plt.show()








# def plot_grouped_metrics(results, title='Intersectional Unfairness Metrics for ', output_file='intersectional_fairness_classifier_on_classifier.pdf',
#                          data_type='Classifier Predictions', dpi=800, scaling_method=None):
#     """
#     Plots grouped metrics with high resolution and clear labels for Overleaf.
#     """
#     # Convert results to DataFrame
#     df = pd.DataFrame(results)
#     metrics = df.columns.tolist()
#     datasets = df.index.tolist()
#     n_metrics = len(metrics)
#     n_datasets = len(datasets)
#     width = 0.08  # width of each bar
#     group_spacing = 0.3  # extra spacing between dataset groups
#     total_width = n_metrics * width + group_spacing
#     x = np.arange(n_datasets) * total_width

#     # Color mapping
#     cmap = cm.get_cmap('viridis', n_metrics)
#     colors = [cmap(i) for i in range(n_metrics)]

#     # Create the plot
#     fig, ax = plt.subplots(figsize=(12, 8))  # Larger figure size for clarity
#     # fig, ax = plt.subplots(figsize=(16, 16))

#     for i, metric in enumerate(metrics):
#         offsets = x + i * width
#         ax.bar(offsets, df[metric], width=width, label=metric.replace("_", " ").title(), color=colors[i], edgecolor='black')

#     # Set x-axis ticks and labels
#     ax.set_xticks(x + (n_metrics - 1) * width / 2)
#     ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=20)

#     # Set axis labels and title
#     ax.set_xlabel('Datasets', fontsize=26, labelpad=15)
#     if scaling_method:
#         ax.set_ylabel(f'{scaling_method} Scaled Values', fontsize=26, labelpad=15)
#     else:
#         ax.set_ylabel('Unscaled Values', fontsize=28, labelpad=15)
#     # Set y-axis tick labels
#     ax.tick_params(axis='y', labelsize=30)
#     ax.tick_params(axis='x', labelsize=30)

#     # Add legend
    
#     ax.legend(title="Intersectional Metrics", ncols=2,  fontsize=26, title_fontsize=26, loc='upper left', frameon=True)

#     # Add gridlines
#     ax.grid(True, linestyle='--', axis='y', alpha=0.6)

#     # Adjust layout and save the plot
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=dpi, bbox_inches='tight')  # Ensure no clipping
#     plt.show()
#     # plt.close(fig)  # Close the figure to free memory
