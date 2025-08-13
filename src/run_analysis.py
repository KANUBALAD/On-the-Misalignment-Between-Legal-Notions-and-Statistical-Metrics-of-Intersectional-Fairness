import os
import pandas as pd
from pathlib import Path
import metrics
import sys

import numpy as np
sys.path.append(os.path.join('..', 'src'))



class FairnessAnalyzer:
    def __init__(self,  sensitive_attrs=["Gender", "Race"], 
                 outcome_col="Y", base_data_path="generated_data", prediction_col=None):
        self.base_data_path = Path(base_data_path)
        self.sensitive_attrs = sensitive_attrs
        self.outcome_col = outcome_col
        self.prediction_col = prediction_col
        
        # Scenario mapping
        self.scenario_mapping = {
            'no_discrimination': 'no_bias',
            'single_discrimination': 'single',
            'multiple_discrimination': 'multiple', 
            'intersectional_discrimination': 'intersectional',
            'compounded_discrimination': 'compounded'
        }
        
    def get_available_seeds(self):
        """Get all available seeds from the data directory"""
        seed_folders = [f.name for f in self.base_data_path.iterdir() if f.is_dir() and f.name.startswith('seed_')]
        seeds = [int(folder.split('_')[1]) for folder in seed_folders]
        return sorted(seeds)
    
    def analyzer_seed(self, seed, verbose=True):
        """Analyze fairness metrics for a single seed across all scenarios"""
        seed_folder = self.base_data_path / f"seed_{seed}"
        
        if not seed_folder.exists():
            raise ValueError(f"Seed folder not found: {seed_folder}")
        
        results = {}
        if verbose:
            print(f"🎯 Analyzing seed {seed}...")
        
        # Find all CSV files in the seed folder
        csv_files = list(seed_folder.rglob("*.csv"))
        
        for csv_file in csv_files:
            # Extract scenario name from filename
            filename = csv_file.stem
            # Remove seed suffix if present
            scenario_name = filename.replace(f"_seed_{seed}", "")
            # Map to shorter name
            display_name = self.scenario_mapping.get(scenario_name, scenario_name)
            
            try:
                # Load data
                df = pd.read_csv(csv_file)
                # Compute metrics
                if self.prediction_col is not None and self.prediction_col in df.columns:
                       bias_results, _ = metrics.evaluation_classifier(
                        df, self.sensitive_attrs, self.outcome_col, self.prediction_col)
                       
                else:
                    bias_results, _ = metrics.evaluation_data(
                        df, self.sensitive_attrs, self.outcome_col)
                    
                results[display_name] = bias_results

            except Exception as e:
                if verbose:
                    print(f"  ❌ Error processing {display_name}: {e}")
                results[display_name] = {}
        
        return results
    
    def analyze_multiple_seeds(self, seeds=None, verbose=True):
        """Analyze fairness metrics across multiple seeds"""
        if seeds is None:
            seeds = self.get_available_seeds()
        
        if verbose:
            print(f"🎯 Analyzing {len(seeds)} seeds: {seeds}")
        
        # Collect data from all seeds
        all_results = {}
        for seed in seeds:
            seed_results = self.analyzer_seed(seed, verbose=False)
            all_results[seed] = seed_results
        
        # Organize data by scenario
        scenario_data = {}
        scenarios = set()
        for seed_results in all_results.values():
            scenarios.update(seed_results.keys())
        
        for scenario in scenarios:
            scenario_data[scenario] = {}
            
            # Collect all metrics across seeds for this scenario
            all_metrics = set()
            for seed_results in all_results.values():
                if scenario in seed_results:
                    all_metrics.update(seed_results[scenario].keys())
            
            # Initialize metric collections
            for metric in all_metrics:
                scenario_data[scenario][metric] = []
            
            # Collect values across seeds
            for seed, seed_results in all_results.items():
                if scenario in seed_results:
                    for metric in all_metrics:
                        value = seed_results[scenario].get(metric, None)
                        if value is not None:
                            scenario_data[scenario][metric].append(value)
        
        # Compute summary statistics
        summary_data = {}
        for scenario, scenario_metrics in scenario_data.items():
            summary_data[scenario] = {}
            for metric, values in scenario_metrics.items():
                if values:  # Only if we have values
                    summary_data[scenario][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
        
        return {
            'scenario_data': scenario_data,
            'summary_data': summary_data,
            'seeds_analyzed': seeds
        }
    