# A python package for measuring intersectional fairness in machine learning with health datasets

## Fairness

Fairness in machine learning refers to the ethical requirement that models do not produce systematically biased or discriminatory outcomes for individuals or groups. In the context of health data science, unfair models may lead to unequal access to diagnosis, treatment, or follow-up.


Bias can arise when model predictions differ across protected attributes such as sex, age, ethnicity, or disability status. A growing number of tools exist to help researchers and practitioners evaluate fairness in machine learning models; however, many focus on single protected attributes in isolation.


## Intersectional Fairness

This package provides tools which allow researchers to evaluate fairness across intersections of protected attributes, rather than considering each attribute independently.

For example, instead of checking:
•	men vs women
•	younger vs older patients

We check:
•	young women
•	older women
•	young men
•	older men

This approach is motivated by the observation that unfairness can be hidden when outcomes are averaged over broad groups. Disparities often emerge at the intersections of attributes, where individuals may experience compounded or 'double' disadvantage. For instance, older women may be treated less favourably than either women or older patients considered as marginal groups alone.

In this package, intersectional groups are evaluated using differential fairness metrics to quantify worst-case disparities in model outcomes.