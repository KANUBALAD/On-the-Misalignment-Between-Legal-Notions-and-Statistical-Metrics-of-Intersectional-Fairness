import plots
import numpy as np
import pandas as pd
 

class SyntheticLoanDataGenerator:
    def __init__(self, sample_size=10000, random_seed=100, prob_gender=0.5, prob_race=0.5,
                 beta=0.03,beta_I =0.2, beta_S =0.03, gamma=0.0, eta=2.0, delta=0.5,measure_bias_col='Gender',
                 thetas=None, beta_coef=None, rhos=None, kappas=None, nus=None, lambdas=None):
        
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.prob_gender = prob_gender
        self.prob_race = prob_race
        
        # Model parameters
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.delta = delta
        self.beta_I = beta_I
        self.beta_S = beta_S
        
        # Discrimination coefficients
        self.thetas = thetas
        self.beta_coef = beta_coef
        self.rhos = rhos
        self.kappas = kappas
        self.nus = nus
        
        # Discrimination strengths
        self.lambdas = lambdas
        self.measure_bias_col = measure_bias_col
        
        np.random.seed(self.random_seed)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
 

    def synthetic_data_generation(self):
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

        
        Y = (self.sigmoid(odds)  <= 0.5 + self.gamma * (bias_term)).astype(int) # choose bias and gamma terms to be small
        thresholds = 0.5 + self.gamma * (bias_term)
        
        # thresholds = 0.5 + self.gamma * self.lambdas['Y'] * bias_term
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
            # 'discrimination_coefficients': {
            #     'theta': discrim_theta,
            #     'beta_coef': discrim_beta,
            #     'rho': discrim_rho,
            #     'kappa': discrim_kappa,
            #     'nu': discrim_nu
            # }
        }

    def discrimination_y(self, data, outcome_col='Y'):
        group_means = data.groupby(self.measure_bias_col)[outcome_col].mean()
        diff = group_means.max() - group_means.min()
        print(f"🔍 Outcome disparity by {self.measure_bias_col}:")
        print(group_means)
        print(f"Disparity (max - min): {diff:.4f}\n")

    def discrimination_treatment(self, data):
        treatment_cols = ['LoanAmount', 'Duration']
        for col in treatment_cols:
            group_means_treatment = data.groupby(self.measure_bias_col)[col].mean()
            diff_treatment = group_means_treatment.max() - group_means_treatment.min()
            print(f"🔍 Treatment disparity in {col} by {self.measure_bias_col}:")
            print(group_means_treatment)
            print(f"Disparity (max - min) in {col}: {diff_treatment:.4f}\n")

    def measure_discrimination(self, data, data_name = None, outcome_col='Y', plot=False, plot_intersectional=True):
        if plot:
            # plots.plot_features(data)
            plots.visualize_histograms(data, data_name = data_name, measure_bias_col=self.measure_bias_col, plot_intersectional=plot_intersectional)
            # plots.correlation_matrix(data)
        self.discrimination_y(data, outcome_col)
        self.discrimination_treatment(data)

    