# DataMiningProject

This repository contains the code and data for our data mining course project on predicting artwork prices using metadata from an online art marketplace. We explore both tree-based classification models and a neural network regression model.

---

## Files

### MainDataset.csv  
The main dataset used throughout the project. It contains artwork metadata (style, medium, size, delivery info, etc.) along with the listed price. All models use this file as input.

---

### Preprocessing_and_NeuralNetwork.ipynb  
This notebook covers the full preprocessing and neural network pipeline. It includes:
- Data cleaning and feature engineering
- Exploratory data analysis
- Log transformation of prices
- One-hot encoding and standardization
- Neural network training and evaluation

The notebook is self-contained and can be run top to bottom to reproduce the neural network results.

---

### Preprocessing_for_Trees.py  
This script handles preprocessing specifically for the tree-based models. It loads the dataset, encodes categorical variables, scales numeric features, and prepares the train/test split used by the classifiers.

---

### TreeBasedClassification.py  
This script trains and evaluates the tree-based models, including Random Forest and Gradient Boosting. It uses the preprocessed data to run both multi-class and binary classification experiments and outputs evaluation metrics such as accuracy and confusion matrices.

---

### .gitignore  
Specifies files and folders that should not be tracked by Git.

---

## Running the Code

- To run the neural network: open `Preprocessing_and_NeuralNetwork.ipynb` and run all cells.
- To run tree-based models:
  1. Run `Preprocessing_for_Trees.py`
  2. Then run `TreeBasedClassification.py`

---

## Notes

The neural network treats price as a continuous variable using regression, while the tree-based models predict discretized price categories. The two approaches are compared in the final report.

