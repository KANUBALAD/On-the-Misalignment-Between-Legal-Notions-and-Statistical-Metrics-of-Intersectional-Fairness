import pandas as pd
import numpy as np
from utlis import get_subgroups





def compute_demographic_parity_ratio(df, sensitive_attributes, outcome_col):
    subgroup_rates = {}
    for label, sub_df in get_subgroups(df, sensitive_attributes):
        if not sub_df.empty:
            subgroup_rates[label] = sub_df[outcome_col].mean()
    if not subgroup_rates:
        raise ValueError("No subgroups found.")
    rates = list(subgroup_rates.values())
    # print(f"Subgroup rates: {rates}")
    dpr = 1- (np.min(rates) / np.max(rates)) if np.max(rates) > 0 else None
    return subgroup_rates, dpr


def compute_elift(df, sensitive_attributes, outcome_col):
    """
    Compute the epsilon-lift (elift) metric:
        e^{-ε} ≤ P(y=1 | sg_i) / P(y=1) ≤ e^ε
    Returns the maximum ε across all subgroups.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing sensitive attributes and outcome.
    sensitive_attributes : list of str
        List of columns to define subgroups (e.g., ['Gender', 'Race']).
    outcome_col : str
        Name of the binary outcome column.
        
    Returns
    -------
    float : Maximum ε across all subgroups.
    dict : Probability of positive outcome per subgroup.
    """
    overall_rate = df[outcome_col].mean()
    if overall_rate == 0:
        raise ValueError("Overall outcome rate must be between 0 and 1.")

    subgroup_probs = {}
    for label, sub_df in get_subgroups(df, sensitive_attributes):
        if len(sub_df) > 0:
            p_sg = sub_df[outcome_col].mean()
            subgroup_probs[label] = p_sg

    epsilons = []
    for label, p_sg in subgroup_probs.items():
        # if p_sg == 0 or p_sg == 1:
        #     continue  
        eps = abs(np.log(p_sg / overall_rate))
        epsilons.append(eps)
    return max(epsilons), subgroup_probs
    
    
def compute_slift(df, sensitive_attributes, outcome_col):
    """
    Compute the subgroup-lift (slift) metric:
        e^{-ε} ≤ P(y=1 | sg_i) / P(y=1 | sg_j) ≤ e^ε
    Returns the maximum ε across all pairs of subgroups.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with sensitive attributes and outcome.
    sensitive_attributes : list of str
        Columns defining subgroup membership.
    outcome_col : str
        Binary outcome column.

    Returns
    -------
    float : Maximum ε over all subgroup pairs.
    dict : Positive outcome rate per subgroup.
    """
    subgroup_probs = {}
    for label, sub_df in get_subgroups(df, sensitive_attributes):
        if len(sub_df) > 0:
            p_sg = sub_df[outcome_col].mean()
            subgroup_probs[label] = p_sg

    epsilons = []
    labels = list(subgroup_probs.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pi = subgroup_probs[labels[i]]
            pj = subgroup_probs[labels[j]]
            if pi > 0 and pj > 0:
                eps = abs(np.log(pi / pj))
                epsilons.append(eps)

    return max(epsilons) if epsilons else 0.0, subgroup_probs


# Original Differential Fairness, accepts rates list
def compute_differentialFairnessBinaryOutcome(probabilitiesOfPositive):
    epsilonPerGroup = np.zeros(len(probabilitiesOfPositive))
    for i in range(len(probabilitiesOfPositive)):
        eps = 0.0
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            p_i, p_j = probabilitiesOfPositive[i], probabilitiesOfPositive[j]
            eps = max(eps, abs(np.log(p_i) - np.log(p_j)))
            eps = max(eps, abs(np.log(1-p_i) - np.log(1-p_j)))
        epsilonPerGroup[i] = eps
    return float(epsilonPerGroup.max())

def compute_SmoothedEDF(df, sensitive_attributes, predictions):
    preds = df[predictions]
    labels = [label for label, _ in get_subgroups(df, sensitive_attributes)]
    # Count occurrences and positive outcomes per subgroup
    counts = {label: [0,0] for label in labels}  # [total, positives]
    for idx, row in df.iterrows():
        for label, _ in get_subgroups(df, sensitive_attributes):
            if all(row[attr] == val for attr, val in zip(sensitive_attributes, label)):
                counts[label][0] += 1
                if preds[idx] == 1:
                    counts[label][1] += 1
                break
    # Dirichlet smoothing
    alpha = 1.0 / 2  # concentration/numClasses
    probs = []
    for total, pos in counts.values():
        probs.append((pos + alpha) / (total + 1.0))
    smoothed_edf_probs = compute_differentialFairnessBinaryOutcome(np.array(probs))
    # edf_probs = compute_differentialFairnessBinaryOutcome(np.array(preds))
    return smoothed_edf_probs, probs



def compute_subgroup_unfairness(df, sensitive_attributes, outcome_col):
    """
    Computes subgroup unfairness
    """
    overall_mean = df[outcome_col].mean()
    total_unfairness = 0.0
    subgroup_rates = {}
    for values, group_df in get_subgroups(df, sensitive_attributes):
        group_mean = group_df[outcome_col].mean()
        group_size = len(group_df) / len(df)
        total_unfairness += abs(overall_mean - group_mean) * group_size
        subgroup_rates[values] = group_mean
    return total_unfairness, subgroup_rates



def compute_equality_of_opportunity_ratio(df, sensitive_attributes, outcome_col, predictions):
    """
    Compute min-max Equality of Opportunity Ratio (EOpR) across subgroups.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing true labels and predictions.
    sensitive_attributes : list of str
        Columns defining subgroup membership.
    label_col : str
        Ground truth label column.
    pred_col : str
        Model prediction column.
    
    Returns
    -------
    float : EOpR score.
    dict : Dictionary of TPR per subgroup.
    """
    subgroup_tprs = {}
    df_pos = df[df[outcome_col] == 1]  
    for label, sub_df in get_subgroups(df_pos, sensitive_attributes):
        if len(sub_df) > 0:
            tpr = sub_df[predictions].mean()
            subgroup_tprs[label] = tpr

    tpr_values = list(subgroup_tprs.values())
    if not tpr_values or max(tpr_values) == 0:
        return None, subgroup_tprs 
    eopr = 1- (min(tpr_values) / max(tpr_values))
    return eopr, subgroup_tprs


def compute_group_benefit_ratio(df, sensitive_attributes, outcome_col):
    """
    Compute Group Benefit Ratio (GBR) across subgroups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing model outcome_col.
    sensitive_attributes : list of str
        Columns defining subgroup membership.
    pred_col : str
        Model prediction column.

    Returns
    -------
    float : GBR score (closer to 1 = unfairer).
    dict : Prediction rate per subgroup.
    """
    subgroup_preds = {}
    for label, sub_df in get_subgroups(df, sensitive_attributes):
        if len(sub_df) > 0:
            subgroup_preds[label] = sub_df[outcome_col].mean()

    pred_values = list(subgroup_preds.values())
    if not pred_values or max(pred_values) == 0:
        return None, subgroup_preds 
    gbr = 1- (min(pred_values) / max(pred_values))
    return gbr, subgroup_preds


def compute_fpr_disparity(df, sensitive_attributes, outcome_col, predictions):
    """
    Compute epsilon-style False Positive Rate Disparity across subgroups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing true labels and predictions.
    sensitive_attributes : list of str
        Columns defining subgroup membership.
    label_col : str
        Column with true binary labels.
    pred_col : str
        Column with predicted binary labels.

    Returns
    -------
    float : epsilon score (higher means more disparity).
    dict : FPR per subgroup.
    """
    subgroup_fprs = {}
    df_neg = df[df[outcome_col] == 0]  # Only negative ground truth

    for label, sub_df in get_subgroups(df_neg, sensitive_attributes):
        if len(sub_df) > 0:
            fpr = sub_df[predictions].mean()
            subgroup_fprs[label] = fpr

    fprs = list(subgroup_fprs.values())
    if not fprs or min(fprs) <= 0:
        return None, subgroup_fprs  # Avoid log(0)

    epsilon_fprs = max(abs(np.log(p1) - np.log(p2)) for p1 in fprs for p2 in fprs if p1 != p2)
    return epsilon_fprs, subgroup_fprs


def compute_tpr_disparity(df, sensitive_attributes, outcome_col, predictions):
    """
    Compute epsilon-style True Positive Rate (TPR) Disparity across subgroups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing true labels and predictions.
    sensitive_attributes : list of str
        Columns defining subgroup membership.
    label_col : str
        Column with true binary labels.
    pred_col : str
        Column with predicted binary labels.

    Returns
    -------
    float : epsilon (max log-ratio TPR disparity).
    dict : TPR per subgroup.
    """
    subgroup_tprs = {}
    df_pos = df[df[outcome_col] == 1]  

    for label, sub_df in get_subgroups(df_pos, sensitive_attributes):
        if len(sub_df) > 0:
            tpr = sub_df[predictions].mean()
            subgroup_tprs[label] = tpr

    tprs = list(subgroup_tprs.values())
    if not tprs or min(tprs) <= 0:
        return None, subgroup_tprs  

    epsilon_tprs = max(abs(np.log(p1) - np.log(p2)) for p1 in tprs for p2 in tprs if p1 != p2)
    return epsilon_tprs, subgroup_tprs


def compute_idd_metric(df, outcome_col, sensitive_attributes, mode=None):
    """
    Computes decomposed disparities: total, additive, and intersectional disparity.
    Allows switching between two intersectional definitions: 'old' (inclusion-exclusion)
    or 'new' (pure interaction term).

    Parameters:
    df (pd.DataFrame): Input data.
    outcome_col (str): Name of the binary outcome column.
    sensitive_attributes (list): List of sensitive attributes, e.g., ['Gender', 'Race'].
    which (str): 'old' or 'new' — chooses which intersectional metric to use.

    Returns:
    tuple: total_disparity, additive_disparity, intersectional_disparity
    """
    if len(sensitive_attributes) != 2:
        raise ValueError("This implementation supports exactly two sensitive attributes.")

    A1, A2 = sensitive_attributes

    # Base & marginal probabilities
    p_y = df[outcome_col].mean()
    p_y_a1 = df[df[A1] == 1][outcome_col].mean()
    p_y_a2 = df[df[A2] == 1][outcome_col].mean()

    # Joint group probabilities
    p_11 = df[(df[A1] == 1) & (df[A2] == 1)][outcome_col].mean()
    p_10 = df[(df[A1] == 1) & (df[A2] == 0)][outcome_col].mean()
    p_01 = df[(df[A1] == 0) & (df[A2] == 1)][outcome_col].mean()
    p_00 = df[(df[A1] == 0) & (df[A2] == 0)][outcome_col].mean()

    # Total disparity
    worst_disparity = np.abs(p_11 - p_00)

    # Additive disparity
    delta_g = p_y_a1 - p_y
    delta_r = p_y_a2 - p_y
    additive_disparity = (abs(delta_g) + abs(delta_r)) 
    # additive_disparity = (abs(delta_g) + abs(delta_r)) / 2

    # Intersectional disparity
    # inter_old = np.abs(p_11 - (p_y_a1 + p_y_a2 - p_y))
    # intersectional_disparity = inter_old

    inter_new = np.abs(p_11 - p_10 - p_01 + p_00)  # sames as AIES
    
    intersectional_disparity = inter_new
    

    return worst_disparity, additive_disparity, intersectional_disparity






# def compute_idd_metric(df, outcome_col, sensitive_attributes, mode=None):
#     """
#     Computes decomposed disparities using pure interaction-based definition
#     assuming binary sensitive attributes.

#     Parameters:
#     df (pd.DataFrame): Input data.
#     outcome_col (str): Name of the binary outcome column.
#     sensitive_attributes (list): List of sensitive attributes, e.g., ['Gender', 'Race'].

#     Returns:
#     tuple: total_disparity, additive_disparity, intersectional_disparity
#     """
#     if len(sensitive_attributes) != 2:
#         raise ValueError("This implementation supports exactly two sensitive attributes.")

#     A1, A2 = sensitive_attributes

#     # Base probabilities
#     p_y = df[outcome_col].mean()  # P(Y=1)
#     p_y_a1 = df[df[A1] == 1][outcome_col].mean()  # P(Y=1 | A1=1)
#     p_y_a2 = df[df[A2] == 1][outcome_col].mean()  # P(Y=1 | A2=1)

#     # Joint probabilities
#     p_11 = df[(df[A1] == 1) & (df[A2] == 1)][outcome_col].mean()  # P(Y=1 | A1=1, A2=1)
#     p_10 = df[(df[A1] == 1) & (df[A2] == 0)][outcome_col].mean()  # P(Y=1 | A1=1, A2=0)
#     p_01 = df[(df[A1] == 0) & (df[A2] == 1)][outcome_col].mean()  # P(Y=1 | A1=0, A2=1)
#     p_00 = df[(df[A1] == 0) & (df[A2] == 0)][outcome_col].mean()  # P(Y=1 | A1=0, A2=0)

#     # Total disparity between most and least privileged
#     worst_disparity = np.abs(p_11 - p_00)

#     # Additive (single-axis) disparity
#     delta_g = p_y_a1 - p_y
#     delta_r = p_y_a2 - p_y
#     print(f"delta_g: {delta_g}, delta_r: {delta_r}")
#     additive_disparity = (abs(delta_g) + abs(delta_r)) / 2

#     # Pure interaction-based intersectional disparity
#     intersectional_disparity = np.abs(p_11 - p_10 - p_01 + p_00)

#     return worst_disparity, additive_disparity, intersectional_disparity







# def compute_idd_metric_old(df, outcome_col, sensitive_attributes, mode=None):
#     # venn diagram of intersectionality
#     """
#     Computes decomposed disparities using inclusion-exclusion logic assuming binary sensitive attributes.
    
#     Parameters:
#     df (pd.DataFrame): Input data.
#     outcome_col (str): Name of the binary outcome column.
#     sensitive_attributes (list): List of sensitive attributes, e.g., ['Gender', 'Race'].
    
#     Returns:
#     tuple: total_disparity, additive_disparity, intersectional_disparity
#     """
#     if len(sensitive_attributes) != 2:
#         raise ValueError("This implementation supports exactly two sensitive attributes.")
    
#     A1, A2 = sensitive_attributes
#     p_y = df[outcome_col].mean()  # P(Y=1)
#     p_y_a1 = df[df[A1] == 1][outcome_col].mean()  # P(Y=1 | A1=1)
#     p_y_a2 = df[df[A2] == 1][outcome_col].mean()  # P(Y=1 | A2=1)
#     p_y_a1_a2 = df[(df[A1] == 1) & (df[A2] == 1)][outcome_col].mean()  # P(Y=1 | A1=1, A2=1)
#     p_y_not_a1_not_a2 = df[(df[A1] == 0) & (df[A2] == 0)][outcome_col].mean()  # P(Y=1 | A1=0, A2=0)

#     # Total disparity between most and least privileged
#     worst_disparity = np.abs(p_y_a1_a2 - p_y_not_a1_not_a2)
    
#     delta_g = p_y_a1 - p_y
#     delta_r = p_y_a2 - p_y
#     print(f"delta_g: {delta_g}, delta_r: {delta_r}")
#     additive_disparity = (abs(delta_g) + abs(delta_r)) / 2

#     # Intersectional disparity 
#     intersectional_disparity = np.abs(p_y_a1_a2 - (p_y_a1 + p_y_a2 - p_y))

#     return worst_disparity, additive_disparity, intersectional_disparity




    


def evaluation_data(df, sensitive_attributes, outcome_col, mode):
    eval_subgroup_rates = {}
    metrics_results = {}
    
    subgroup_rates, dpr = compute_demographic_parity_ratio(df, sensitive_attributes, outcome_col)
    metrics_results['demographic_disparity'] = dpr
    eval_subgroup_rates['demographic_disparity'] = subgroup_rates
    
    epsilons_elift, elift_subgroup_probs= compute_elift(df, sensitive_attributes, outcome_col)
    metrics_results['elift'] = epsilons_elift
    eval_subgroup_rates['elift'] = elift_subgroup_probs
    
    epsilons_slift, slift_subgroup_probs = compute_slift(df, sensitive_attributes, outcome_col)
    metrics_results['slift'] = epsilons_slift
    eval_subgroup_rates['slift'] = slift_subgroup_probs
    
    subgroup_unfairness, subgroup_rates= compute_subgroup_unfairness(df, sensitive_attributes, outcome_col)
    metrics_results['subgroup_unfairness'] = subgroup_unfairness
    eval_subgroup_rates['subgroup_unfairness'] = subgroup_rates
    
    D_worst_disparity, D_additive, D_intersectional = compute_idd_metric(df, outcome_col, sensitive_attributes, mode)
    # print(f"what is the value of D_subgroups: {D_subgroups}, D_additive: {D_additive}, D_intersectional: {D_intersectional}")
    metrics_results['idd_intersectional'] = D_intersectional
    metrics_results['idd_worst_disparity'] = D_worst_disparity
    metrics_results['idd_additive'] = D_additive
    
    return metrics_results, eval_subgroup_rates
    
def evaluation_classifier(df, sensitive_attributes, outcome_col, prediction_col, mode):
    eval_subgroup_rates = {}
    metrics_results = {}
    
    
    subgroup_rates, dpr = compute_demographic_parity_ratio(df, sensitive_attributes, prediction_col)
    metrics_results['demographic_disparity'] = dpr
    eval_subgroup_rates['demographic_disparity'] = subgroup_rates
    
    eopr, subgroup_eopr= compute_equality_of_opportunity_ratio(df, sensitive_attributes, outcome_col, prediction_col)
    metrics_results['eopr'] = eopr
    eval_subgroup_rates['eopr'] = subgroup_eopr
    
    gbr, subgroup_preds= compute_group_benefit_ratio(df, sensitive_attributes, prediction_col)
    metrics_results['group benefit'] = gbr
    eval_subgroup_rates['group benefit'] = subgroup_preds
    
    smoothed_edf_probs, _ = compute_SmoothedEDF(df, sensitive_attributes, prediction_col)
    metrics_results['smoothed_edf'] = smoothed_edf_probs


    epsilon_tprs, subgroup_tprs = compute_tpr_disparity(df, sensitive_attributes, outcome_col, prediction_col)
    metrics_results['tpr disparity'] = epsilon_tprs
    eval_subgroup_rates['tpr disparity'] = subgroup_tprs
    
    epsilon_fprs, subgroup_fprs= compute_fpr_disparity(df, sensitive_attributes, outcome_col, prediction_col)
    metrics_results['fpr disparity'] = epsilon_fprs
    eval_subgroup_rates['fpr disparity'] = subgroup_fprs
    
     
    epsilons_elift, elift_subgroup_probs= compute_elift(df, sensitive_attributes, prediction_col)
    metrics_results['elift'] = epsilons_elift
    eval_subgroup_rates['elift'] = elift_subgroup_probs
    
    # epsilons_slift, slift_subgroup_probs = compute_slift(df, sensitive_attributes, prediction_col)
    # metrics_results['slift'] = epsilons_slift
    # eval_subgroup_rates['slift'] = slift_subgroup_probs
    
    subgroup_unfairness, subgroup_rates= compute_subgroup_unfairness(df, sensitive_attributes, prediction_col)
    metrics_results['subgroup_unfairness'] = subgroup_unfairness
    eval_subgroup_rates['subgroup_unfairness'] = subgroup_rates
    
    
    D_worst_disparity, D_additive, D_intersectional = compute_idd_metric(df, prediction_col, sensitive_attributes, mode)
    # print(f"what is the value of D_subgroups: {D_subgroups}, D_additive: {D_additive}, D_intersectional: {D_intersectional}")
    metrics_results['idd_intersectional'] = D_intersectional
    metrics_results['idd_worst_disparity'] = D_worst_disparity
    metrics_results['idd_additive'] = D_additive
    return metrics_results, eval_subgroup_rates



    
    
    
    
    
    
    
    
    





