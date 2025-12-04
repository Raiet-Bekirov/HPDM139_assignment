
# **Meeting Minutes – Differential Fairness Toolkit Project**

**Meeting #4**
**Date:** 4th December 2025
**Time:** 10:30 AM (Online)

---

## **1. Attendance**

All group members present.

---

## **2. Progress Summary**

The group reported **good progress** since the previous meeting.
A shared understanding is emerging of the **core structure** of the fairness toolkit package, including:

* A modular layout (datasets, metrics, preprocessing, workflows, examples)
* Clear separation between reusable package code and demonstration scripts
* Agreement on aiming for a simple, functional **v0.1** of the toolkit

---

## **3. Main Discussion Points**

### **3.1 Agreement on Core Structure**

The group discussed the essential components that should make up the first version of the package.
Consensus was reached on the following:

* A minimal but clean folder structure
* One dataset loader
* One or two fairness metrics
* A simple model pipeline for demonstration
* Documentation in `/docs/`
* An examples folder showing end-to-end usage

This forms the baseline that everyone will work from individually.

---

### **3.2 Individual Pipelines for Comparison**

The group agreed that each member will now independently create a **small working pipeline**, including:

* Loading a dataset
* Applying a fairness metric
* Showing evaluation outputs (e.g., accuracy, parity difference)

These pipelines will be brought to the next meeting for comparison.
The goal is to review different approaches and then decide which components should be committed to the **`dev` branch** as part of the shared toolkit.

---

## **4. Actions**

### **Before the next meeting:**

* **All members** will create and test a minimal working fairness pipeline.
* **Next meeting** will involve sharing pipelines, discussing design choices, and deciding which implementation details to merge into the `dev` branch.

---

## **5. Date of Next Meeting**

**11th December 2025 at 10:30 AM (Online)**


