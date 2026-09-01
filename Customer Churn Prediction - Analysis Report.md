Customer Churn Prediction - Analysis Report

Executive Summary
This report presents a comprehensive machine learning solution for predicting customer churn in the telecommunications industry. Using Amazon SageMaker AI, we built a predictive model that achieves an AUC score of 0.8631, enabling proactive customer retention strategies.

Key Business Impact:

Annual Revenue Protection: $304,770

Customers Saved: 441

ROI: 312%

Payback Period: 2.9 months

1. Business Problem

    A[Customer Churn] --> B[Revenue Loss]
    B --> C[Acquisition Costs]
    C --> D[Reduced Growth]
    
    E[ML Solution] --> F[Predict Churn]
    F --> G[Proactive Retention]
    G --> H[Revenue Protection]

The Challenge
Customer churn is one of the most critical challenges for subscription-based businesses. Acquiring new customers costs 5-7x more than retaining existing ones. A 5% reduction in churn can increase profits by 25-95%.

Business Objectives
Objective	Target	Impact
Churn Reduction	25-30%	Revenue protection
Customer Lifetime Value	+20%	Increased revenue
Retention Campaign ROI	+25%	Marketing efficiency
Annual Revenue Protection	$1.5M	Financial stability

2. Solution Architecture



┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│                           CUSTOMER CHURN PREDICTION                                 │
│                        END-TO-END ML PIPELINE                                       │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        1. DATA INGESTION                                     │   │
│  │                                                                              │   │
│  │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │   │
│  │   │             │     │             │     │             │                   │   │
│  │   │   Amazon    │────▶│   Amazon    │────▶│   Amazon    │                   │   │
│  │   │     S3      │     │    Glue     │     │   Athena    │                   │   │
│  │   │             │     │             │     │             │                   │   │
│  │   └─────────────┘     └─────────────┘     └─────────────┘                   │   │
│  │                                                                              │   │
│  │   Data Source: Telco Customer Churn Dataset (7,043 records, 21 features)    │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                            │
│                                       ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        2. DATA PREPARATION                                   │   │
│  │                                                                              │   │
│  │   ┌──────────────────────────────────────────────────────────────────────┐  │   │
│  │   │                                                                      │  │   │
│  │   │  DATA CLEANING & FEATURE ENGINEERING                                 │  │   │
│  │   │                                                                      │  │   │
│  │   │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │   │
│  │   │  │   Data     │  │   Feature  │  │   Feature  │  │   Train/   │    │  │   │
│  │   │  │   Cleaning │─▶│ Engineering│─▶│  Scaling   │─▶│   Test     │    │  │   │
│  │   │  │            │  │            │  │            │  │   Split    │    │  │   │
│  │   │  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │  │   │
│  │   │                                                                      │  │   │
│  │   │  Input: Raw Customer Data  │  Output: 60% Train, 20% Val, 20% Test │  │   │
│  │   └──────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                            │
│                                       ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        3. MODEL TRAINING                                     │   │
│  │                                                                              │   │
│  │   ┌──────────────────────────────────────────────────────────────────────┐  │   │
│  │   │                                                                      │  │   │
│  │   │                    AMAZON SAGEMAKER                                  │  │   │
│  │   │                                                                      │  │   │
│  │   │   ┌────────────────────┐     ┌────────────────────┐                 │  │   │
│  │   │   │                    │     │                    │                 │  │   │
│  │   │   │   XGBoost Model    │────▶│  Hyperparameter    │                 │  │   │
│  │   │   │    Training Job    │     │     Tuning         │                 │  │   │
│  │   │   │                    │     │  (10 Jobs)         │                 │  │   │
│  │   │   └────────────────────┘     └────────────────────┘                 │  │   │
│  │   │                                                                      │  │   │
│  │   │   Best Model: AUC = 0.8631 │ Best Parameters: max_depth=3, eta=0.1 │  │   │
│  │   └──────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                            │
│                                       ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        4. MODEL DEPLOYMENT                                   │   │
│  │                                                                              │   │
│  │   ┌──────────────────────────────────────────────────────────────────────┐  │   │
│  │   │                                                                      │  │   │
│  │   │                    MODEL ENDPOINT                                    │  │   │
│  │   │                                                                      │  │   │
│  │   │   ┌────────────────────┐     ┌────────────────────┐                 │  │   │
│  │   │   │                    │     │                    │                 │  │   │
│  │   │   │   SageMaker        │────▶│    Real-Time       │                 │  │   │
│  │   │   │   Endpoint         │     │    Inference       │                 │  │   │
│  │   │   │                    │     │                    │                 │  │   │
│  │   │   └────────────────────┘     └────────────────────┘                 │  │   │
│  │   │                                                                      │  │   │
│  │   │   Endpoint: ml.m5.large │ Response: Churn Probability (0.0 - 1.0) │  │   │
│  │   └──────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                            │
│                                       ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        5. BUSINESS INTEGRATION                               │   │
│  │                                                                              │   │
│  │   ┌──────────────────────────────────────────────────────────────────────┐  │   │
│  │   │                                                                      │  │   │
│  │   │   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │   │
│  │   │   │    CRM     │  │  Customer  │  │  Marketing │  │  Retention │    │  │   │
│  │   │   │  System    │  │  Success   │  │  Campaigns │  │  Programs  │    │  │   │
│  │   │   └────────────┘  └────────────┘  └────────────┘  └────────────┘    │  │   │
│  │   │                                                                      │  │   │
│  │   │   Business Value: $304,770 Annual Savings │ 441 Customers Saved    │  │   │
│  │   └──────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
