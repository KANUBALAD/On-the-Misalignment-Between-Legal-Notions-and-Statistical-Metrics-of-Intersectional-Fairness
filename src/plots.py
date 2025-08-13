import os
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import get_cmap
sys.path.append(os.path.join('..', 'src'))
from helper import standardize_across_metrics, standardize_metrics_adaptive,standardize_within_metrics, discrimination_y, discrimination_treatment




def plot_measure_discrimination(data, measure_bias_col, data_name = None, outcome_col='Y', plot=False, plot_intersectional=True):
    if plot:
        # plots.plot_features(data)
        visualize_histograms(data, data_name = data_name, measure_bias_col=measure_bias_col, plot_intersectional=plot_intersectional)
        # plots.correlation_matrix(data)
    discrimination_y(data, outcome_col)
    discrimination_treatment(data, measure_bias_col)
    

def get_color_mapping(unique_groups, cmap_name="viridis"):
    """
    Given a list of group names, return a consistent color mapping (dict).
    """
    unique_groups = sorted(unique_groups)  # ensure consistent order
    cmap = get_cmap(cmap_name)
    colors = [cmap(i / len(unique_groups)) for i in range(len(unique_groups))]
    return dict(zip(unique_groups, colors))
    


def visualize_histograms(data, data_name = None,  measure_bias_col=None, plot_intersectional=True):
    """
    Visualize the distribution of covariates across intersectional groups of Gender and Race.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing at least 'Gender' and 'Race'.
    """
    data = data.copy()  # Avoid modifying original DataFrame
    data['IntersectionalGroup'] = data['Gender'].astype(str) + "_" + data['Race'].astype(str)
        
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



def plot_metrics_from_summary(results_summary, output_file, 
                             data_type='Generated Data', dpi=800, 
                             scalar_type=None, metrics_to_scale=None, 
                             apply_scaling=True, force_scaling_type=None):
    """
    Plots metrics with adaptive standardization based on single/multi-seed detection.
    
    Parameters:
    - results_summary: Summary DataFrame
    - output_file: Path to save plot
    - data_type: Type of data for labels
    - dpi: Plot resolution
    - scalar_type: 'MinMax', 'Zscore', 'Robust'
    - metrics_to_scale: List of metrics to scale (None = all)
    - apply_scaling: Whether to apply scaling
    - force_scaling_type: 'across' or 'within' to override automatic detection
    """
    
    # Apply scaling if requested
    if apply_scaling and scalar_type:
        if force_scaling_type == 'across':
            plot_data = standardize_across_metrics(results_summary, scalar_type, metrics_to_scale)
            scaling_info = f"{scalar_type} (Cross-Metric)"
        elif force_scaling_type == 'within':
            plot_data = standardize_within_metrics(results_summary, scalar_type, metrics_to_scale)
            scaling_info = f"{scalar_type} (Within-Metric)"
        else:
            # Adaptive scaling
            plot_data = standardize_metrics_adaptive(results_summary, scalar_type, metrics_to_scale)
            is_single_seed = all(results_summary['count'] == 1)
            scaling_info = f"{scalar_type} ({'Cross-Metric' if is_single_seed else 'Within-Metric'})"
    else:
        plot_data = results_summary.copy()
        scaling_info = None
    
    # Pivot the summary DataFrame
    df = plot_data.pivot(index='scenario', columns='metric', values='mean')
    
    desired_order = ['no_bias', 'single', 'multiple', 'intersectional', 'compounded']
    available_scenarios = df.index.tolist()
    ordered_scenarios = [scenario for scenario in desired_order if scenario in available_scenarios]
    df = df.reindex(ordered_scenarios)

    # Get metrics and scenarios
    metrics = df.columns
    datasets = df.index
    
    # Set up plot parameters
    y = np.arange(len(datasets))
    height = 0.10
    cmap = cm.get_cmap('viridis', len(metrics))
    colors = [cmap(i) for i in range(len(metrics))]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bars
    for i, metric in enumerate(metrics):
        ax.barh(y + i * height, df[metric], height, label=metric, color=colors[i], edgecolor='black')
    
    # Formatting
    label_fontsize = 26
    ax.set_ylabel('Datasets', fontsize=label_fontsize, labelpad=15)
    
    if scaling_info:
        ax.set_xlabel(f'{scaling_info} Values', fontsize=24, labelpad=15)
    else:
        ax.set_xlabel('Unscaled Values', fontsize=24, labelpad=15)
        
    ax.set_yticks(y + height * (len(metrics) - 1) / 2)
    ax.set_yticklabels(datasets, fontsize=label_fontsize)
    
    # Set x-axis ticks
    max_val = df.max().max()
    min_val = df.min().min()
    
    if scaling_info and scalar_type == 'MinMax':
        xtick_positions = np.arange(0, 1.1, 0.2)
    else:
        range_val = max_val - min_val
        tick_spacing = max(0.1, range_val / 5)
        xtick_positions = np.arange(min_val, max_val + tick_spacing, tick_spacing)
    
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f'{x:.1f}' for x in xtick_positions], fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    
    ax.legend(fontsize=20, title_fontsize=22, loc='lower right', frameon=True)
    ax.grid(True, linestyle='--', axis='x', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.show()
    


