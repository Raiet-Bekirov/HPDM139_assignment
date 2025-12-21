import numpy as np
from itertools import product


def calculate_fairness(group_dict, model_predictions, true_statuses,
                       zero_accuracy_epsilon = 1e-10):
    """
    Calculate fairness metrics across demographic groups.
    
    This function computes accuracy for each subgroup (defined by the intersection
    of all provided demographic attributes) and calculates epsilon, which represents
    the maximum log-ratio between any two subgroup accuracies. A lower epsilon
    indicates more equitable performance across groups.
    
    Parameters
    ----------
    group_dict : dictionary
        Dictionary where keys are group names (e.g., 'race', 'gender', 'age') and
        values are lists of group labels for each sample.
        Example: {
            'gender': ['Man', 'Woman', 'Man', ...],
            'age': ['Young', 'Old', 'Young', ...]
        }
    model_predictions : list
        List of binary model predictions (0 or 1)
    true_statuses : list
        List of ground truth binary labels (0 or 1)
    zero_accuracy_epsilon : float, optional
        Small value to replace zero accuracies with (default: 1e-10).
        Prevents taking log(0) or division by 0.
    
    Returns
    -------
    epsilon : float
        Maximum log-ratio between subgroup accuracies (fairness metric)
    exp_epsilon : float
        Exponentiated epsilon (maximum ratio between subgroup accuracies)
    accuracies : dictionary
        Dictionary mapping subgroup names to their accuracies
    """
    n_samples = len(model_predictions)
    
    accurate_or_not = [pred == truth for pred, truth in zip(model_predictions, true_statuses)]
    
    group_names = sorted(group_dict.keys())
    unique_values = {}
    for group_name in group_names:
        unique_values[group_name] = sorted(set(group_dict[group_name]))
    
    all_combinations = list(product(*[unique_values[name] for name in group_names]))
    
    accuracies = {}
    for combination in all_combinations:
        mask = [True] * n_samples
        for i, group_name in enumerate(group_names):
            group_value = combination[i]
            for j in range(n_samples):
                if group_dict[group_name][j] != group_value:
                    mask[j] = False
        results = [acc for acc, include in zip(accurate_or_not, mask) if include]
        subgroup_name = "_".join(str(val) for val in combination)
        if len(results) > 0:
            accuracies[subgroup_name] = sum(results) / len(results)
    
    adjusted_accuracies = {}
    for group, acc in accuracies.items():
        if acc == 0:
            adjusted_accuracies[group] = zero_accuracy_epsilon
        else:
            adjusted_accuracies[group] = acc
    
    ratios = []
    for p_i in adjusted_accuracies.values():
        for p_j in adjusted_accuracies.values():
            ratios.append(p_i / p_j)
    
    epsilon = max(np.log(ratios))
    exp_epsilon = np.exp(epsilon)
    
    return epsilon, exp_epsilon, accuracies