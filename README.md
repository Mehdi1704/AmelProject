## Introduction

This Python project uses machine learning to predict the risk of Cardiovascular Disease (CVD) based on lifestyle habits and health data. The dataset comes from the Behavioral Risk Factor Surveillance System (BRFSS) and includes information on the health behaviors and conditions of people in the U.S.

The goal is to create a model that can classify whether someone might develop CVD, supporting early detection and prevention.

## Project Architecture
```
README.md                     # Project documentation
data_cleaning.ipynb           # Jupyter notebook for data cleaning processes
helpers.py                    # General helper functions used across different parts of the project
helpers_analysis.py           # Functions focused on data analysis, including accuracy computations and plot generation
helpers_implementations.py    # Helper functions for machine learning models
helpers_preprocessing.py      # Data preprocessing helper functions for cleaning and preparing data for model training
implementations.py            # Contains the six requested implementations
run.py                        # Main script for submission
run_implementation.ipynb      # Jupyter notebook for running all models, plot generation, and exploring results interactively
report.pdf                    # Two page report explaining our reasoning and results
```
## Team
This project was made by team AmelProject  
[AI-Crowd link](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/teams/amel_project)  

Mehdi Bouchoucha            @Mehdi1704  
Mohamed Hedi Hidri          @Oraxi  
Ali Ridha Mrad              @Ali-Mrad

## Best Model

For our binary classification task, we chose regularized logistic regression as the primary model due to its effectiveness in handling binary outcomes, offering a good balance of interpretability and computational efficiency. The model performed well in predicting cardiovascular disease risk, achieving an F1 score of 0.434, an accuracy of 0.861, and an AUC of 0.8586, as shown by the ROC curve.
