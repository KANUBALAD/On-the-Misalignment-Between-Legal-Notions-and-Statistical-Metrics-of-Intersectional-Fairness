# import plots
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler




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


def discrimination_y(data, measure_bias_col, outcome_col='Y'):
    group_means = data.groupby(measure_bias_col)[outcome_col].mean()
    diff = group_means.max() - group_means.min()
    print(f"🔍 Outcome disparity by {measure_bias_col}:")
    print(group_means)
    print(f"Disparity (max - min): {diff:.4f}\n")
    

def discrimination_treatment(data, measure_bias_col):
    treatment_cols = ['LoanAmount', 'Duration']
    for col in treatment_cols:
        group_means_treatment = data.groupby(measure_bias_col)[col].mean()
        diff_treatment = group_means_treatment.max() - group_means_treatment.min()
        print(f"🔍 Treatment disparity in {col} by {measure_bias_col}:")
        print(group_means_treatment)
        print(f"Disparity (max - min) in {col}: {diff_treatment:.4f}\n")


def extract_metric_across_seeds(all_results, scenario, metric_name):
    """Extract a specific metric for a specific scenario across all seeds"""
    values = []
    seeds_with_data = []
    for seed, seed_results in all_results.items():
        if scenario in seed_results and metric_name in seed_results[scenario]:
            value = seed_results[scenario][metric_name]
            if value is not None:
                values.append(value)
                seeds_with_data.append(seed)
    
    return values, seeds_with_data


def extract_scenario_data(all_results, scenario):
    """Extract all metrics for a specific scenario across all seeds"""
    scenario_data = {}
    
    # Get all unique metrics for this scenario
    all_metrics = set()
    for seed_results in all_results.values():
        if scenario in seed_results:
            all_metrics.update(seed_results[scenario].keys())
    
    # Extract each metric across seeds
    for metric in all_metrics:
        values, seeds_with_data = extract_metric_across_seeds(all_results, scenario, metric)
        scenario_data[metric] = {
            'values': values,
            'seeds': seeds_with_data,
            'mean': np.mean(values) if values else None,
            'std': np.std(values, ddof=1) if len(values) > 1 else 0,
            'median': np.median(values) if values else None,
            'min': np.min(values) if values else None,
            'max': np.max(values) if values else None
        }
    
    return scenario_data


def create_summary_dataframe(all_results):
    """Create a summary DataFrame with statistics for each scenario-metric combination"""
    scenarios = set()
    for seed_results in all_results.values():
        scenarios.update(seed_results.keys())
    
    summary_rows = []
    for scenario in scenarios:
        scenario_data = extract_scenario_data(all_results, scenario)
        
        for metric_name, metric_info in scenario_data.items():
            summary_rows.append({
                'scenario': scenario,
                'metric': metric_name,
                'mean': metric_info['mean'],
                'std': metric_info['std'],
                'median': metric_info['median'],
                'min': metric_info['min'],
                'max': metric_info['max'],
                'count': len(metric_info['values'])
            })
    
    return pd.DataFrame(summary_rows)





def _get_scaler(scalar_type):
    """Helper function to get the appropriate scaler."""
    scaler_map = {
        'MinMax': MinMaxScaler(),
        'Zscore': StandardScaler(), 
        'Robust': RobustScaler()
    }
    
    if scalar_type not in scaler_map:
        raise ValueError(f"Unknown scalar_type: {scalar_type}")
    
    return scaler_map[scalar_type]


def _pivot_to_summary(df_pivot, original_summary):
    """Convert pivot DataFrame back to summary format."""
    # Convert to summary format
    summary_scaled = df_pivot.reset_index().melt(
        id_vars='scenario', 
        var_name='metric', 
        value_name='mean'
    )
    # Add back other columns from original
    other_cols = [col for col in original_summary.columns if col not in ['scenario', 'metric', 'mean']]
    for col in other_cols:
        temp_df = original_summary[['scenario', 'metric', col]]
        summary_scaled = summary_scaled.merge(temp_df, on=['scenario', 'metric'], how='left')
    
    return summary_scaled

def standardize_metrics_adaptive(results_summary, scalar_type='MinMax', metrics_to_scale=None):
    """
    Adaptively standardizes metrics based on whether data is single-seed or multi-seed.
    Parameters:
    - results_summary: Summary DataFrame with scenario, metric, mean, std, count columns
    - scalar_type: 'MinMax', 'Zscore', 'Robust'
    - metrics_to_scale: List of specific metrics to scale (None = all)
    
    Returns:
    - Scaled results_summary DataFrame
    
    """
    
    # Determine if this is single-seed or multi-seed data
    is_single_seed = all(results_summary['count'] == 1)
    
    if is_single_seed:
        print("🎯 Detected single-seed data: Scaling ACROSS metrics for comparability")
        return standardize_across_metrics(results_summary, scalar_type, metrics_to_scale)
    else:
        print("🎯 Detected multi-seed data: Scaling WITHIN metrics across scenarios")
        return standardize_within_metrics(results_summary, scalar_type, metrics_to_scale)


def standardize_across_metrics(results_summary, scalar_type='MinMax', metrics_to_scale=None):
    """
    Single-seed standardization: Scale across ALL metric values to make them comparable.
    
    Example: If demographic_disparity=0.5, elift=0.3, slift=0.8
    After MinMax scaling: All values will be on 0-1 scale relative to each other.
    """
    
    # Pivot to get scenario x metric structure
    df_pivot = results_summary.pivot(index='scenario', columns='metric', values='mean')
    
    # Filter metrics if specified
    if metrics_to_scale is not None:
        available_metrics = [m for m in metrics_to_scale if m in df_pivot.columns]
        if not available_metrics:
            raise ValueError(f"None of specified metrics {metrics_to_scale} found in data")
        df_to_scale = df_pivot[available_metrics]
        df_unchanged = df_pivot.drop(columns=available_metrics)
    else:
        df_to_scale = df_pivot
        df_unchanged = pd.DataFrame(index=df_pivot.index)
    
    # Choose scaler
    scaler = _get_scaler(scalar_type)
    
    # Scale ALL values together (flattened then reshaped)
    original_shape = df_to_scale.shape
    flattened_values = df_to_scale.values.flatten().reshape(-1, 1)
    scaled_flattened = scaler.fit_transform(flattened_values)
    scaled_values = scaled_flattened.reshape(original_shape)
    
    # Create scaled DataFrame
    df_scaled = pd.DataFrame(scaled_values, index=df_to_scale.index, columns=df_to_scale.columns)
    
    # Combine scaled and unchanged metrics
    if not df_unchanged.empty:
        df_final = pd.concat([df_scaled, df_unchanged], axis=1)
    else:
        df_final = df_scaled
    
    # Convert back to summary format
    return _pivot_to_summary(df_final, results_summary)


def standardize_within_metrics(results_summary, scalar_type='MinMax', metrics_to_scale=None):
    """
    Multi-seed standardization: Scale each metric independently across scenarios.
    
    Example: demographic_disparity values [0.1, 0.3, 0.5] across scenarios
    After MinMax scaling: [0.0, 0.5, 1.0] - preserves relative ordering within metric
    """
    
    scaled_summary = results_summary.copy()
    
    # Get metrics to scale
    all_metrics = scaled_summary['metric'].unique()
    if metrics_to_scale is None:
        metrics_to_scale = all_metrics
    
    # Choose scaler
    scaler = _get_scaler(scalar_type)
    # Scale each metric independently
    for metric in metrics_to_scale:
        if metric in all_metrics:
            metric_mask = scaled_summary['metric'] == metric
            metric_values = scaled_summary.loc[metric_mask, 'mean'].values.reshape(-1, 1)
            # Apply scaling
            scaled_values = scaler.fit_transform(metric_values)
            scaled_summary.loc[metric_mask, 'mean'] = scaled_values.flatten()
    return scaled_summary


