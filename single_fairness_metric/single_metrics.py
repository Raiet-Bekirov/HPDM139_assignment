#Import necessary packages
import pandas as pd
import numpy as np
import itertools


def calculate_TP_FN_FP_TN(y_test,y_pred):
    """
    Computes the confusion matrix components: True Positives (TP),
    False Negatives (FN), True Negatives (TN), and False Positives (FP).

    This function compares the true binary labels with the predicted
    binary labels and counts how many predictions fall into each
    confusion matrix category.

    Parameters
    ----------
    y_test : array-like of shape (n_samples,)
        Ground-truth binary labels.
        Expected values: 0 (negative class) or 1 (positive class).

    y_pred : array-like of shape (n_samples,)
        Predicted binary labels produced by a classifier.
        Expected values: 0 (negative class) or 1 (positive class).

    Returns
    -------
    tp : int
        Number of true positives (y_test = 1 and y_pred = 1).

    fn : int
        Number of false negatives (y_test = 1 and y_pred = 0).

    tn : int
        Number of true negatives (y_test = 0 and y_pred = 0).

    fp : int
        Number of false positives (y_test = 0 and y_pred = 1).

    Notes
    -----
    - This function assumes binary classification with labels {0, 1}.
    - The order of returned values is (TP, FN, TN, FP)
    """
    tp = fp = tn = fn = 0

    for a, b in zip(y_test, y_pred):
        if a == 1 and b == 1:
            tp += 1
        elif a == 1 and b == 0:
            fn += 1
        elif a == 0 and b == 1:
            fp += 1
        elif a == 0 and b == 0:
            tn += 1

    return tp, fn, tn, fp

def calculate_TPR_TNR_FPR_FNR(tp,fn,tn,fp):
    """
    Compute classification rate metrics derived from the confusion matrix.

    This function calculates:
    - True Positive Rate (TPR) / Sensitivity / Recall
    - True Negative Rate (TNR) / Specificity
    - False Positive Rate (FPR)
    - False Negative Rate (FNR)

    Parameters
    ----------
    tp : int
        Number of true positives.

    fn : int
        Number of false negatives.

    tn : int
        Number of true negatives.

    fp : int
        Number of false positives.

    Returns
    -------
    tpr : float
        True Positive Rate, defined as TP / (TP + FN).
        Measures how well the model correctly identifies positive cases.

    tnr : float
        True Negative Rate, defined as TN / (TN + FP).
        Measures how well the model correctly identifies negative cases.

    fpr : float
        False Positive Rate, defined as FP / (FP + TN).
        Measures the proportion of negative cases incorrectly classified
        as positive.

    fnr : float
        False Negative Rate, defined as FN / (FN + TP).
        Measures the proportion of positive cases incorrectly classified
        as negative.

    Notes
    -----
    - All rates lie in the range [0, 1].
    """
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    FPR = fp / (tn + fp)
    FNR = fn / (tp + fn)
    return TPR, TNR, FPR, FNR

def calculate_EOD(y_test, y_pred, privileged_group):
    """
    Compute the Equal Opportunity Difference (EOD) between demographic groups.

    Equal Opportunity Difference measures the absolute difference in
    True Positive Rates (TPR) between the underprivileged and privileged
    groups. A lower EOD indicates fairer performance with respect to
    correctly identifying positive cases across groups.

    Parameters
    ----------
    y_test : array-like of shape (n_samples,)
        Ground-truth binary labels.
        Expected values: 0 (negative outcome) or 1 (positive outcome).

    y_pred : array-like of shape (n_samples,)
        Predicted binary labels from a classifier.
        Expected values: 0 (negative outcome) or 1 (positive outcome).

    privileged_group : array-like of shape (n_samples,)
        Binary indicator of group membership.
        - 1 indicates membership in the privileged group
        - 0 indicates membership in the underprivileged group

    Returns
    -------
    EOD : float
        Equal Opportunity Difference, defined as:

            EOD = |TPR_underprivileged − TPR_privileged|

        Values closer to 0 indicate better fairness.

    Notes
    -----
    - EOD focuses exclusively on the positive class (y = 1).
    """
    # Masks
    mask_priv = (privileged_group == 1)
    mask_unpriv = (privileged_group == 0)
    
    # Privileged group
    tp_p, fn_p, fp_p, tn_p = calculate_TP_FN_FP_TN(
        y_test[mask_priv],
        y_pred[mask_priv]
    )
    TPR_p, TNR_p, FPR_p, FNR_p = calculate_TPR_TNR_FPR_FNR(tp_p,fn_p,tn_p,fp_p)
    
    # Underprivileged group
    tp_u, fn_u, fp_u, tn_u = calculate_TP_FN_FP_TN(
        y_test[mask_unpriv],
        y_pred[mask_unpriv]
    )
    TPR_u, TNR_u, FPR_u, FNR_u = calculate_TPR_TNR_FPR_FNR(tp_u,fn_u,tn_u,fp_u)
    
    # Equal Opportunity Difference
    EOD = abs(TPR_u - TPR_p)
    
    return EOD

def calculate_AOD(y_test, y_pred, privileged_group):
    """
    Compute the Average Odds Difference (AOD) between demographic groups.

    Average Odds Difference measures the average difference in both
    True Positive Rates (TPR) and False Positive Rates (FPR) between the
    underprivileged and privileged groups. It captures disparities in
    model performance for both positive and negative outcomes.

    Parameters
    ----------
    y_test : array-like of shape (n_samples,)
        Ground-truth binary labels.
        Expected values: 0 (negative outcome) or 1 (positive outcome).

    y_pred : array-like of shape (n_samples,)
        Predicted binary labels from a classifier.
        Expected values: 0 (negative outcome) or 1 (positive outcome).

    privileged_group : array-like of shape (n_samples,)
        Binary indicator of group membership.
        - 1 indicates membership in the privileged group
        - 0 indicates membership in the underprivileged group

    Returns
    -------
    AOD : float
        Average Odds Difference, defined as:

            AOD = 0.5 × [ (FPR_underprivileged − FPR_privileged)
                        + (TPR_underprivileged − TPR_privileged) ]

        Values closer to 0 indicate better fairness.
    """
    # Masks
    mask_priv = (privileged_group == 1)
    mask_unpriv = (privileged_group == 0)
    
    
    print(y_test)
    print(mask_priv)
    print(y_test[mask_priv])
    
    # Privileged group
    tp_p, fn_p, fp_p, tn_p = calculate_TP_FN_FP_TN(
        y_test[mask_priv],
        y_pred[mask_priv]
    )
    TPR_p, TNR_p, FPR_p, FNR_p = calculate_TPR_TNR_FPR_FNR(tp_p,fn_p,tn_p,fp_p)
    
    # Underprivileged group
    tp_u, fn_u, fp_u, tn_u = calculate_TP_FN_FP_TN(
        y_test[mask_unpriv],
        y_pred[mask_unpriv]
    )
    TPR_u, TNR_u, FPR_u, FNR_u = calculate_TPR_TNR_FPR_FNR(tp_u,fn_u,tn_u,fp_u)
    
    # Average Odds Difference
    AOD = ((FPR_u - FPR_p) + (TPR_u - TPR_p)) / 2

    return AOD

def calculate_DI(y_pred, group):
    """
    Compute Disparate Impact (DI) between demographic groups.

    Disparate Impact measures the ratio of positive prediction rates
    between the underprivileged and privileged groups. It evaluates
    whether one group receives favorable outcomes less frequently
    than another, regardless of ground-truth labels.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels from a classifier.
        Expected values: 0 (negative outcome) or 1 (positive outcome).

    group : array-like of shape (n_samples,)
        Binary indicator of group membership.
        - 1 indicates membership in the privileged group
        - 0 indicates membership in the underprivileged group

    Returns
    -------
    DI : float
        Disparate Impact, defined as:

            DI = P(ŷ = 1 | underprivileged) / P(ŷ = 1 | privileged)

        where P(ŷ = 1 | group) is the positive prediction rate
        for the specified group.

    """
    mask_priv = (group == 1)
    mask_unpriv = (group == 0)

    P_priv = np.mean(y_pred[mask_priv] == 1)
    P_unpriv = np.mean(y_pred[mask_unpriv] == 1)

    return P_unpriv / P_priv