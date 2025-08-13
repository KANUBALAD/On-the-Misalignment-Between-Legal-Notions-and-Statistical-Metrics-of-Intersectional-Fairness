import yaml
import numpy as np
import pandas as pd
import os

class SyntheticLoanDataGenerator:
    def __init__(self, config):
        """
        Initializes the generator with values from the config file.
        """
        self.sample_size = config["sample_size"]
        self.random_seed = config.get("random_seed", 42)  # Default seed if not provided
        print(f"Using random seed: {self.random_seed}")
        # self.random_seed = config["random_seed"]
        self.prob_gender = config["prob_gender"]
        self.prob_race = config["prob_race"]
        self.beta = config["beta"]
        self.beta_I = config["beta_I"]
        self.beta_S = config["beta_S"]
        self.gamma = config["gamma"]
        self.eta = config["eta"]
        self.delta = config["delta"]
        self.measure_bias_col = config["measure_bias_col"]
        self.thetas = config["thetas"]
        self.beta_coef = config["beta_coef"]
        self.rhos = config["rhos"]
        self.kappas = config["kappas"]
        self.nus = config["nus"]
        self.lambdas = config["lambdas"]
        
        np.random.seed(self.random_seed)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def generate_data_for_scenario(self):
        """
        Generates data based on the loaded config scenario.
        """
        return self._generate_data(self.thetas, self.beta_coef, self.rhos, self.kappas, self.nus, self.lambdas)

    def _generate_data(self, thetas, beta_coef, rhos, kappas, nus, lambdas):
        # The main data generation logic
        
        # Generate sensitive attributes
        G = np.random.binomial(1, self.prob_gender, self.sample_size)
        R = np.random.binomial(1, self.prob_race, self.sample_size)

        # Generate noises
        UE = np.random.normal(0, 0.25, self.sample_size)
        UI = np.random.normal(0, 4, self.sample_size)
        US = np.random.normal(0, 5, self.sample_size)
        UL = np.random.normal(0, 10, self.sample_size)
        UD = np.random.normal(0, 9, self.sample_size)
        UY = np.random.normal(0, self.eta, self.sample_size)

        
        # Covariates
        E = -0.5 + self.lambdas['E'] * (self.thetas['G'] * G + self.thetas['R'] * R + self.thetas['GR'] * (G * R)) + UE
        I = -4 + 3 * E + self.lambdas['I'] * (self.beta_coef['G'] * G + self.beta_coef['R'] * R + self.beta_coef['GR'] * (G * R)) + UI
        S = -4 + 1.5 * np.where(I > 0, I, 0) + US
        
        # Treatments
        L = 1 - self.beta_I * I - self.beta_S * S + self.lambdas['L'] * (self.rhos['G'] * G + self.rhos['R'] * R + self.rhos['GR'] * (G * R))  + UL
        D = -1 - self.beta_I * I  + L + self.lambdas['D'] * (self.kappas['G'] * G + self.kappas['R'] * R + self.kappas['GR'] * (G * R)) + UD

        # Outcome
        alpha = np.where((I > 0) & (S > 0), 1, -1)
        UY = np.random.normal(0, self.eta, self.sample_size)
        
        bias_term = self.lambdas['Y']*(self.nus['G'] * G + self.nus['R'] * R + self.nus['GR'] * (G * R))
        odds = 15.0 + self.delta * (-L - D) + 0.3 * (I + S + alpha * I * S)
        
        thresholds = 0.5 + self.gamma * self.lambdas['Y'] * bias_term
        Y = (self.sigmoid(odds) >= thresholds).astype(int)

        data_df = pd.DataFrame({
            'Gender': G,
            'Race': R,
            'Education': E,
            'Income': I,
            'Savings': S,
            'LoanAmount': L,
            'Duration': D,
            'Y': Y,
        })

        return {
            'df_array': np.vstack([G, R, E, I, S, L, D, Y]).T,
            'noises_U': np.vstack([UE, UI, US, UL, UD, UY]).T,
            'data_df': data_df,
            'thresholds': thresholds,
            'probabilities': self.sigmoid(odds)
        }

# Function to load the YAML config file
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def generate_data_for_scenarios(config_path, seed_list, output_folder):
    config = load_config(config_path)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for seed in seed_list:
        # Update seed for each run
        config['random_seed'] = seed
        generator = SyntheticLoanDataGenerator(config)
        
        # Generate data for the scenario
        data = generator.generate_data_for_scenario()
        
        # Create filename without seed suffix since it's now in a seed-specific folder
        scenario_name = os.path.basename(config_path).split('.')[0]  # e.g., "no_discrimination"
        file_name = f"{scenario_name}_seed_{seed}.csv"
        file_path = os.path.join(output_folder, file_name)
        
        # Save the data as CSV
        data['data_df'].to_csv(file_path, index=False)
        print(f"💾 Saved: {file_name}")