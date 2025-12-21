import numpy as np

group_a_or_b = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'B']
group_1_or_2 = [1, 2, 1, 2, 1, 1, 2, 2, 1, 2]
model_prediction = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
true_status = [1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
accurate_or_not = [pred == truth for pred, truth in zip(model_prediction, true_status)]

a_1_results = [acc for acc, group_ab, group_12 in zip(accurate_or_not, group_a_or_b, group_1_or_2) if group_ab == 'A' and group_12 == 1]
a_2_results = [acc for acc, group_ab, group_12 in zip(accurate_or_not, group_a_or_b, group_1_or_2) if group_ab == 'A' and group_12 == 2]
b_1_results = [acc for acc, group_ab, group_12 in zip(accurate_or_not, group_a_or_b, group_1_or_2) if group_ab == 'B' and group_12 == 1]
b_2_results = [acc for acc, group_ab, group_12 in zip(accurate_or_not, group_a_or_b, group_1_or_2) if group_ab == 'B' and group_12 == 2]

a_1_accuracy = sum(a_1_results) / len(a_1_results)
a_2_accuracy = sum(a_2_results) / len(a_2_results)
b_1_accuracy = sum(b_1_results) / len(b_1_results)
b_2_accuracy = sum(b_2_results) / len(b_2_results)

accuracies = {}
for group_ab in ['A', 'B']:
    for group_12 in [1, 2]:
        results = [acc for acc, a_or_b, one_or_two in zip(accurate_or_not, group_a_or_b, group_1_or_2) if a_or_b == group_ab and one_or_two == group_12]
        accuracies[f"{group_ab}_{group_12}"] = sum(results) / len(results)
        
ratios = []
for p_i in accuracies.values():
    for p_j in accuracies.values():
            ratios.append(p_i / p_j)
            
epsilon = max(np.log(ratios))
print(epsilon)
print(np.exp(epsilon))