# Customer-Churn-Prediction-with-Amazon-SageMaker-AI

# Customer Churn Prediction with Amazon SageMaker AI

[![AWS](https://img.shields.io/badge/AWS-SageMaker-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MLOps](https://img.shields.io/badge/MLOps-Pipeline-0066CC?style=for-the-badge&logo=mlflow&logoColor=white)](https://aws.amazon.com/sagemaker/pipelines/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements a complete end-to-end machine learning pipeline for predicting customer churn using Amazon SageMaker AI. It demonstrates the full ML lifecycle from business understanding to model deployment and monitoring, following MLOps best practices.

### Key Results

| Metric | Score | Status |
|--------|-------|--------|
| **AUC Score** | **0.8631** | ✅ PASS |
| Accuracy | 82.3% | ✅ PASS |
| Precision | 75.2% | ✅ PASS |
| Recall | 70.1% | ✅ PASS |
| F1 Score | 72.5% | ✅ PASS |

### Business Impact

| Metric | Value |
|--------|-------|
| **Annual Savings** | **$304,770** |
| **Customers Saved** | **441** |
| **ROI** | **312%** |
| **Payback Period** | **2.9 months** |

## Architecture

## Architecture

### System Architecture

```mermaid
flowchart TB
    subgraph Data["Data Layer"]
        S3[("S3 Bucket<br/>Churn.csv")]
    end
    
    subgraph Processing["Processing Layer"]
        FE[Feature Engineering<br/>• Tenure Segments<br/>• Service Adoption Score<br/>• Churn Risk Indicators]
        Split[Train/Test Split<br/>60/20/20]
    end
    
    subgraph Training["Training Layer"]
        SM[SageMaker Training]
        HT[Hyperparameter Tuning<br/>Bayesian Optimization]
        XGB[XGBoost Model]
    end
    
    subgraph Deployment["Deployment Layer"]
        EP[("SageMaker Endpoint<br/>Real-time Inference")]
        MB[ModelBuilder]
    end
    
    subgraph Monitoring["Monitoring & MLOps"]
        CM[CloudWatch<br/>Metrics]
        Reg[Model Registry<br/>Versioning]
    end
    
    subgraph Business["Business Applications"]
        CRM[CRM Integration]
        Dash[Dashboard<br/>QuickSight]
        Alert[Alert System<br/>SNS]
    end
    
    S3 --> FE
    FE --> Split
    Split --> SM
    SM --> HT
    HT --> XGB
    XGB --> MB
    MB --> EP
    EP --> CRM
    EP --> Dash
    EP --> Alert
    CM -.-> EP
    Reg -.-> XGB
```

### Data Flow

```mermaid
flowchart LR
    A[Customer Data] --> B[Data Cleaning]
    B --> C[Feature Engineering]
    C --> D[XGBoost Model]
    D --> E[Churn Probability]
    E --> F[Risk Score]
    F --> G[Retention Actions]
```

### SageMaker V3 Components

```mermaid
flowchart TB
    subgraph V3["SageMaker V3"]
        MT[ModelTrainer]
        MB2[ModelBuilder]
        EP2[Endpoint]
        ID[InputData]
        Comp[Compute]
    end
    
    ID --> MT
    Comp --> MT
    MT --> MB2
    MB2 --> EP2
```

### Business Decision Flow

```mermaid
flowchart TD
    A[Customer Data] --> B[SageMaker Endpoint]
    B --> C[Churn Score 0.0-1.0]
    
    C --> D{Score > 0.7?}
    D -->|Yes| E[High Risk<br/>Retention Call]
    D -->|No| F{Score > 0.4?}
    
    F -->|Yes| G[Medium Risk<br/>Email Campaign]
    F -->|No| H{Score > 0.2?}
    
    H -->|Yes| I[Low Risk<br/>Regular Check-in]
    H -->|No| J[Loyal<br/>Referral Program]
    
    E --> K[Revenue Protection]
    G --> K
    I --> K
    J --> K
    
    K --> L[$304,770 Annual Savings]
```


## Project Checklist

| # | Item | Status |
|---|------|--------|
| 1 | Business Understanding | ✅ |
| 2 | Data Loading | ✅ |
| 3 | Data Exploration (EDA) | ✅ |
| 4 | Data Cleaning | ✅ |
| 5 | Feature Engineering | ✅ |
| 6 | Train-Test Split | ✅ |
| 7 | Model Training (SageMaker XGBoost) | ✅ |
| 8 | Hyperparameter Tuning | ✅ |
| 9 | Model Deployment | ✅ |
| 10 | Inference | ✅ |
| 11 | Evaluation | ✅ |
| 12 | Cleanup | ✅ |


## Results

## Key Insights
Month-to-month contracts have the highest churn rate (26.7%)

New customers (<6 months tenure) are most at risk

Electronic check users show higher churn probability

Service adoption significantly reduces churn risk


## Technology Stack

AWS SageMaker: Model training and deployment

XGBoost: Classification algorithm

Scikit-learn: Data preprocessing

Pandas: Data manipulation

Matplotlib/Seaborn: Visualization

Author
Hammad Shaikh
