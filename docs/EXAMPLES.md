# 🧪 **Examples & Use Cases**

This guide provides practical examples for using the Deepline MCP Server with common data science workflows.

## **📊 Basic Data Analysis Examples**

### **Example 1: Customer Data Analysis**
```
# Load customer data
"Load customer_data.csv"

# Basic exploration
"Show me basic info about customer_data"
"Get statistical summary for all numeric columns"
"Create correlation heatmap for customer_data"

# Quality assessment
"Analyze missing data patterns in customer_data"
"Generate data quality report for customer_data"
"Detect outliers using IQR method"
```

**Expected Output:**
- Dataset shape: (5000, 12)
- Memory usage: 2.3 MB
- Missing data: 3 columns with >10% missing
- Quality score: 82/100
- Outliers detected: 47 customers

### **Example 2: Sales Performance Analysis**
```
# Load sales data
"Load sales_2023.csv and sales_2024.csv"

# Compare datasets
"Compare sales_2023 vs sales_2024 for drift analysis"
"What's the drift count and percentage?"

# Visualizations
"Create histogram of sales_amount for both datasets"
"Show boxplot of sales_amount by quarter"
"Create scatter plot of marketing_spend vs sales_amount"
```

**Expected Output:**
- Drift detected: 4 out of 8 columns (50%)
- Significant changes in: price, customer_segment, region
- Marketing ROI correlation: 0.67

---

## **🔍 Data Quality Assessment Examples**

### **Example 3: Healthcare Dataset Quality Check**
```
# Load healthcare data
"Load patient_records.csv"

# Comprehensive quality analysis
"Infer schema for patient_records"
"Analyze missing data with recommendations"
"Generate comprehensive quality report"
"Create missing data visualization"

# Outlier detection
"Detect outliers in patient_records using isolation forest"
"Show outliers with 3% contamination rate"
```

**Expected Results:**
- **Schema detection**: 15 columns with patterns (email, phone, date)
- **Missing data**: MCAR pattern detected, 12% overall missing
- **Quality score**: 76/100
- **Outliers**: 23 patients with unusual vital signs

### **Example 4: Financial Transaction Monitoring**
```
# Load transaction data
"Load transactions_jan.csv and transactions_feb.csv"

# Schema validation
"Infer schema for transactions_jan"
"What patterns are detected in transaction_amount?"

# Drift monitoring
"Analyze drift between transactions_jan and transactions_feb"
"Generate drift analysis report"

# Anomaly detection
"Detect outliers in transaction_amount using LOF method"
```

**Expected Results:**
- **Pattern detection**: Credit card numbers, account IDs, timestamps
- **Drift analysis**: 3 columns showing significant drift (transaction_amount, merchant_category, location)
- **Anomalies**: 156 potentially fraudulent transactions

---

## **📈 Model Performance Monitoring Examples**

### **Example 5: Regression Model Evaluation**
```
# Load model predictions
"Load baseline_predictions.csv and current_predictions.csv"

# Performance evaluation
"Evaluate regression performance with my predictions"
# (You'll provide y_true and y_pred arrays)

# Drift analysis
"Compare baseline_predictions vs current_predictions for drift"
"What's the model performance degradation?"
```

**Expected Metrics:**
- **RMSE**: 0.234 (baseline) vs 0.287 (current)
- **MAE**: 0.156 (baseline) vs 0.198 (current)
- **R²**: 0.89 (baseline) vs 0.82 (current)
- **Drift**: 2 features showing significant drift

### **Example 6: Classification Model Monitoring**
```
# Load classification results
"Load model_predictions_week1.csv and model_predictions_week2.csv"

# Classification metrics
"Evaluate classification performance with my predictions"
# (Provide y_true and y_pred for both weeks)

# Performance comparison
"Compare model_predictions_week1 vs model_predictions_week2 for drift"
"Generate model performance report"
```

**Expected Metrics:**
- **Accuracy**: 0.94 (week1) vs 0.91 (week2)
- **Precision**: 0.92 (week1) vs 0.89 (week2)
- **Recall**: 0.95 (week1) vs 0.93 (week2)
- **F1-score**: 0.93 (week1) vs 0.91 (week2)

---

## **🔧 Feature Engineering Examples**

### **Example 7: Feature Transformation Pipeline**
```
# Load raw data
"Load raw_features.csv"

# Apply transformations
"Apply Box-Cox transformation to raw_features"
"Apply log transformation excluding target_column"
"Apply quantile binning to continuous variables"

# Advanced transformations
"Reduce cardinality in categorical columns"
"Apply multiple transformations: boxcox, log, binning"
```

**Expected Results:**
- **Box-Cox**: 8 columns transformed, skewness reduced by 60%
- **Log transformation**: 5 columns transformed with automatic shifting
- **Binning**: 12 continuous variables discretized into 5 bins
- **Cardinality reduction**: 3 categorical columns grouped (threshold: 0.5%)

### **Example 8: Multi-step Feature Engineering**
```
# Load training data
"Load training_data.csv"

# Step 1: Data quality
"Analyze missing data patterns in training_data"
"Detect outliers using IQR method"

# Step 2: Feature transformation
"Apply feature transformation with target column target_var"
"Show VIF analysis for multicollinearity"

# Step 3: Final validation
"Generate data quality report for transformed data"
"Create correlation heatmap for final features"
```

**Expected Pipeline:**
1. **Missing data**: 3 columns with >20% missing → imputation recommended
2. **Outliers**: 67 rows identified → treatment applied
3. **Transformation**: 15 features created from 10 original
4. **VIF analysis**: 2 features with high multicollinearity removed
5. **Final quality**: 94/100 score

---

## **📊 Real-World Use Cases**

### **Use Case 1: E-commerce Customer Analysis**
```
# Monthly customer behavior analysis
"Load customer_behavior_jan.csv, customer_behavior_feb.csv, customer_behavior_mar.csv"

# Trend analysis
"Compare customer_behavior_jan vs customer_behavior_feb for drift"
"Compare customer_behavior_feb vs customer_behavior_mar for drift"

# Customer segmentation preparation
"Detect outliers in customer_behavior_mar using isolation forest"
"Apply feature transformation for clustering preparation"
"Generate comprehensive quality report"
```

**Business Value:**
- **Drift detection**: Identify changing customer behavior patterns
- **Outlier detection**: Find unusual customer segments
- **Quality assurance**: Ensure data reliability for ML models

### **Use Case 2: Manufacturing Quality Control**
```
# Daily production quality monitoring
"Load production_data_today.csv"
"Load production_baseline.csv"

# Quality monitoring
"Compare production_baseline vs production_data_today for drift"
"Detect outliers in production_data_today using IQR method"

# Process validation
"Infer schema for production_data_today"
"Analyze missing data patterns"
"Generate drift analysis report"
```

**Business Value:**
- **Process drift**: Early detection of manufacturing issues
- **Quality control**: Automated anomaly detection
- **Compliance**: Data quality validation for regulatory requirements

### **Use Case 3: Financial Risk Assessment**
```
# Credit risk model monitoring
"Load credit_applications_baseline.csv"
"Load credit_applications_current.csv"

# Risk model validation
"Compare credit_applications_baseline vs credit_applications_current for drift"
"Detect outliers in credit_applications_current using LOF method"

# Compliance reporting
"Generate comprehensive quality report"
"Create missing data visualization"
"Infer schema for pattern validation"
```

**Business Value:**
- **Model stability**: Monitor for concept drift
- **Fraud detection**: Identify unusual application patterns
- **Regulatory compliance**: Automated data quality reporting

---

## **🚀 Advanced Workflows**

### **Advanced Workflow 1: ML Pipeline Monitoring**
```
# Complete ML pipeline monitoring
"Load training_data.csv, validation_data.csv, production_data.csv"

# Multi-dataset comparison
"Compare training_data vs validation_data for drift"
"Compare validation_data vs production_data for drift"

# Feature quality analysis
"Apply feature transformation to all datasets"
"Detect outliers in production_data using multiple methods"
"Generate quality reports for all datasets"
```

### **Advanced Workflow 2: Time Series Data Quality**
```
# Time series data validation
"Load timeseries_baseline.csv, timeseries_current.csv"

# Temporal drift analysis
"Compare timeseries_baseline vs timeseries_current for drift"
"Detect outliers in timeseries_current using isolation forest"

# Pattern analysis
"Infer schema for temporal patterns"
"Analyze missing data with time-based clustering"
"Create comprehensive quality assessment"
```

---

## **💡 Best Practices from Examples**

### **Data Loading Best Practices**
- **Use descriptive names** for datasets
- **Load related datasets together** for comparison
- **Check basic info** immediately after loading

### **Quality Assessment Best Practices**
- **Always start with missing data analysis**
- **Use schema inference** for pattern validation
- **Generate quality reports** for documentation

### **Monitoring Best Practices**
- **Establish baselines** for drift detection
- **Use multiple outlier detection methods**
- **Document findings** with generated reports

### **Performance Optimization**
- **Sample large datasets** for visualization
- **Use appropriate contamination rates** for outlier detection
- **Monitor memory usage** with basic info

---

## **🔧 Troubleshooting Examples**

### **Common Issue 1: Large Dataset Handling**
```
# Problem: Dataset too large for processing
"Load large_dataset.csv"  # Memory error

# Solution: Check size first
"Show basic info about large_dataset"
"Get statistical summary with sampling"
"Create visualizations with reduced sample size"
```

### **Common Issue 2: Missing Data Patterns**
```
# Problem: High missing data percentage
"Analyze missing data patterns in sparse_dataset"

# Solution: Understand patterns
"Create missing data visualization"
"Infer schema for data type validation"
"Generate quality report with recommendations"
```

### **Common Issue 3: Drift Detection Sensitivity**
```
# Problem: Too many columns showing drift
"Compare dataset1 vs dataset2 for drift"  # 80% drift

# Solution: Adjust sensitivity
"Use stricter contamination threshold for drift"
"Focus on key business metrics for drift analysis"
"Generate detailed drift report for investigation"
```

---

**🎯 Ready to try these examples? Start with the basic data analysis examples and work your way up to the advanced workflows!** 