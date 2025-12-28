import math
import numpy as np
from itertools import product


def group_acc(group_label, subject_labels, predictions, true_statuses):
    n_samples = len(predictions)

    accurate_or_not = [pred == truth
                       for pred, truth
                       in zip(predictions, true_statuses)]

    in_group = [False] * n_samples
    for observation in range(n_samples):
        if subject_labels[observation] == group_label:
            in_group[observation] = True
    group_results = [acc for acc, include
                     in zip(accurate_or_not, in_group)
                     if include is True]

    if len(group_results) > 0:
        accuracy = sum(group_results) / len(group_results)
    else:
        accuracy = np.nan

    return accuracy


def group_acc_diff(group_a_label, group_b_label, subject_labels,
                   predictions, true_statuses):
    group_a_accuracy = group_acc(group_label=group_a_label,
                                 subject_labels=subject_labels,
                                 predictions=predictions,
                                 true_statuses=true_statuses)
    group_b_accuracy = group_acc(group_label=group_b_label,
                                 subject_labels=subject_labels,
                                 predictions=predictions,
                                 true_statuses=true_statuses)

    if np.isnan(group_a_accuracy) or np.isnan(group_b_accuracy):
        diff = np.nan
    else:
        diff = abs(group_a_accuracy - group_b_accuracy)

    return diff


def group_acc_ratio(group_a_label, group_b_label, subject_labels,
                    predictions, true_statuses, natural_log=True):
    group_a_accuracy = group_acc(group_label=group_a_label,
                                 subject_labels=subject_labels,
                                 predictions=predictions,
                                 true_statuses=true_statuses)
    group_b_accuracy = group_acc(group_label=group_b_label,
                                 subject_labels=subject_labels,
                                 predictions=predictions,
                                 true_statuses=true_statuses)

    if np.isnan(group_a_accuracy) or np.isnan(group_b_accuracy):
        ratio = np.nan
    else:
        ratio_a_b = group_a_accuracy / group_b_accuracy
        ratio_b_a = group_b_accuracy / group_a_accuracy
        ratio = max(ratio_a_b, ratio_b_a)

    if natural_log is True:
        return np.log(ratio)
    else:
        return ratio


def intersect_acc(group_labels_dict, subject_labels_dict,
                  predictions, true_statuses):
    n_samples = len(predictions)
    categories = sorted(group_labels_dict.keys())

    accurate_or_not = [pred == truth
                       for pred, truth
                       in zip(predictions, true_statuses)]

    in_groups = [False] * n_samples
    for observation in range(n_samples):
        category_match = []
        for category in categories:
            group = group_labels_dict[category]
            if subject_labels_dict[category][observation] == group:
                category_match.append(1)
            else:
                category_match.append(0)
        in_intersectional_group = bool(math.prod(category_match))
        if in_intersectional_group is True:
            in_groups[observation] = True

    intersect_group_results = [acc for acc, include
                               in zip(accurate_or_not, in_groups)
                               if include is True]

    if len(intersect_group_results) > 0:
        accuracy = sum(intersect_group_results) / len(intersect_group_results)
    else:
        accuracy = np.nan

    return accuracy


def all_intersect_accs(subject_labels_dict, predictions, true_statuses):
    category_names = sorted(subject_labels_dict.keys())
    unique_groups = {}
    for category in category_names:
        unique_groups[category] = sorted(set(subject_labels_dict[category]))

    all_combinations = list(product(*[unique_groups[category]
                            for category in category_names]))

    accuracies = {}
    for combination in all_combinations:
        combination_dict = {}
        for i, category_name in enumerate(category_names):
            combination_dict[category_name] = combination[i]
        intersect_accuracy = intersect_acc(
                                group_labels_dict=combination_dict,
                                subject_labels_dict=subject_labels_dict,
                                predictions=predictions,
                                true_statuses=true_statuses)
        intersect_group_name = " + ".join(str(group) for group in combination)
        accuracies[intersect_group_name] = intersect_accuracy

    return accuracies


def max_intersect_acc_diff(subject_labels_dict, predictions, true_statuses):
    accuracies = all_intersect_accs(
                    subject_labels_dict=subject_labels_dict,
                    predictions=predictions,
                    true_statuses=true_statuses)
    accuracy_values = np.array(list(accuracies.values()))

    if any(np.isnan(accuracy_values)):
        max_diff = np.nan
    else:
        max_diff = max(accuracy_values) - min(accuracy_values)

    return max_diff


def max_intersect_acc_ratio(subject_labels_dict, predictions, true_statuses,
                            natural_log=True):
    accuracies = all_intersect_accs(
                    subject_labels_dict=subject_labels_dict,
                    predictions=predictions,
                    true_statuses=true_statuses)
    accuracy_values = np.array(list(accuracies.values()))

    if any(np.isnan(accuracy_values)):
        max_ratio = np.nan
    else:
        max_ratio = max(accuracy_values) / min(accuracy_values)

    if natural_log is True:
        return np.log(max_ratio)
    else:
        return max_ratio


def group_fnr(group_label, subject_labels, predictions, true_statuses):
    pass


def group_fnr_diff():
    pass


def group_fnr_ratio():
    pass


def intersect_fnr():
    pass


def all_intersect_fnrs():
    pass


def max_intersect_fnr_diff():
    pass


def max_intersect_fnr_ratio():
    pass


def group_fpr(group_label, subject_labels, predictions, true_statuses):
    pass


def group_fpr_diff():
    pass


def group_fpr_ratio():
    pass


def intersect_fpr():
    pass


def all_intersect_fprs():
    pass


def max_intersect_fpr_diff():
    pass


def max_intersect_fpr_ratio():
    pass


def group_for(group_label, subject_labels, predictions, true_statuses):
    n_samples = len(predictions)

    accurate_or_not = [pred == truth
                       for pred, truth
                       in zip(predictions, true_statuses)]

    in_group = [False] * n_samples
    for observation in range(n_samples):
        if subject_labels[observation] == group_label:
            in_group[observation] = True
    group_neg_results = [acc for acc, include, pos
                         in zip(accurate_or_not, in_group, predictions)
                         if include is True and bool(pos) is False]

    if len(group_neg_results) > 0:
        false_omi_rate = (len(group_neg_results)-sum(group_neg_results)) \
                         / len(group_neg_results)
    else:
        false_omi_rate = np.nan

    return false_omi_rate


def group_for_diff():
    pass


def group_for_ratio():
    pass


def intersect_for():
    pass


def all_intersect_fors():
    pass


def max_intersect_for_diff():
    pass


def max_intersect_for_ratio():
    pass


def group_fdr(group_label, subject_labels, predictions, true_statuses):
    n_samples = len(predictions)

    accurate_or_not = [pred == truth
                       for pred, truth
                       in zip(predictions, true_statuses)]

    in_group = [False] * n_samples
    for observation in range(n_samples):
        if subject_labels[observation] == group_label:
            in_group[observation] = True
    group_pos_results = [acc for acc, include, pos
                         in zip(accurate_or_not, in_group, predictions)
                         if include is True and bool(pos) is True]

    if len(group_pos_results) > 0:
        false_dis_rate = (len(group_pos_results)-sum(group_pos_results)) \
                         / len(group_pos_results)
    else:
        false_dis_rate = np.nan

    return false_dis_rate


def group_fdr_diff():
    pass


def group_fdr_ratio():
    pass


def intersect_fdr():
    pass


def all_intersect_fdrs():
    pass


def max_intersect_fdr_diff():
    pass


def max_intersect_fdr_ratio():
    pass
