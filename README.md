# Forest Cover Type Classification

A machine learning project that predicts forest cover types using cartographic variables and environmental features. This project implements a complete ML pipeline including data preprocessing, feature selection, hyperparameter optimization, and model evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project classifies forest cover types based on cartographic variables derived from US Geological Survey (USGS) and US Forest Service (USFS) data. The goal is to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data).

The project uses various machine learning algorithms including:
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree

## 📊 Dataset

The dataset contains cartographic variables for 30x30 meter cells obtained from US Geological Survey and US Forest Service data. Each observation corresponds to a 30m x 30m patch of forest in the Roosevelt National Forest of northern Colorado.

### Target Classes (Cover Types):
1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

### Input Features:
- **Elevation** - Elevation in meters
- **Aspect** - Aspect in degrees azimuth
- **Slope** - Slope in degrees
- **Horizontal_Distance_To_Hydrology** - Horizontal distance to nearest surface water features
- **Vertical_Distance_To_Hydrology** - Vertical distance to nearest surface water features
- **Horizontal_Distance_To_Roadways** - Horizontal distance to nearest roadway
- **Hillshade_9am/Noon/3pm** - Hillshade index at different times
- **Horizontal_Distance_To_Fire_Points** - Horizontal distance to nearest wildfire ignition points
- **Wilderness_Area** (4 binary columns) - Wilderness area designation
- **Soil_Type** (40 binary columns) - Soil type designation

## 🛠️ Features

- **Data Preprocessing**: Feature scaling using StandardScaler
- **Feature Selection**: Automated feature selection using ExtraTreesClassifier
- **Skewness Correction**: Yeo-Johnson transformation for non-normal distributions
- **Hyperparameter Optimization**: Optuna-based optimization for multiple algorithms
- **Model Comparison**: Comprehensive evaluation of multiple ML algorithms
- **Cross-validation**: 5-fold cross-validation for robust model evaluation
- **Visualization**: Distribution plots, correlation heatmaps, and confusion matrices


## 📁 Project Structure

```
forest_cover_prediction/
│
├── README.md                    # Project documentation
├── train.ipynb                  # Main Jupyter notebook with complete pipeline
├── train.csv                    # Training dataset
├── random_forest_model.pkl      # Trained model (generated after running)
├── optuna_trials.csv           # Optimization results for basic classifiers
└── optuna_tree_trials.csv      # Optimization results for tree-based classifiers
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Missing Value Check**: Verification of data completeness
- **Duplicate Removal**: Ensuring data quality
- **Feature Scaling**: StandardScaler for numerical features
- **Skewness Correction**: Yeo-Johnson transformation

### 2. Feature Engineering
- **Feature Selection**: ExtraTreesClassifier for importance-based selection
- **Dimensionality Reduction**: Automatic selection of most relevant features
- **Distribution Analysis**: KDE plots for feature distribution visualization

### 3. Model Selection
- **Baseline Model**: DummyClassifier for performance comparison
- **Multiple Algorithms**: Comparison of 7 different ML algorithms
- **Hyperparameter Optimization**: Optuna for automated parameter tuning
- **Cross-Validation**: 5-fold CV for robust performance estimation

### 4. Evaluation
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix**: Detailed per-class performance analysis
- **Visualization**: Performance plots and distribution analysis

## 📊 Results

The project provides:
- Detailed classification reports for all models
- Feature importance analysis
- Correlation analysis between features
- Before/after comparison of skewness correction
- Comprehensive model comparison through Optuna optimization
