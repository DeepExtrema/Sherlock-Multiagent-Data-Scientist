# 📖 **User Guide**

## **Getting Started**

Once you have Deepline installed and running, you can start analyzing data through natural language conversations in Claude Desktop.

### **Basic Workflow**
1. **Load data** → 2. **Explore** → 3. **Analyze quality** → 4. **Monitor performance**

---

## **🔍 Data Loading & Exploration**

### **Loading Data**
```
"Load the sales_data.csv dataset"
"Load customer_data.xlsx and name it customers"
"Load the JSON file user_events.json"
```

**Supported formats**: CSV, Excel (.xlsx, .xls), JSON

### **Basic Information**
```
"Show me basic info about sales_data"
"What's the shape and data types of customers?"
"Give me an overview of the loaded datasets"
```

**Returns**: Dataset shape, column types, memory usage, sample rows

### **Statistical Summary**
```
"Get statistical summary of sales_data"
"Show descriptive statistics for all numeric columns"
"What's the correlation between price and sales?"
```

**Includes**: Mean, median, std, min, max, quartiles, correlation matrix

---

## **📊 Data Visualization**

### **Available Plot Types**

#### **Histogram**
```
"Create a histogram of sales_amount"
"Show distribution of customer_age"
```

#### **Box Plot**
```
"Create a boxplot of price by category"
"Show boxplot of sales_amount"
```

#### **Scatter Plot**
```
"Create scatter plot of price vs sales_amount"
"Plot customer_age against lifetime_value"
```

#### **Correlation Heatmap**
```
"Create correlation heatmap for sales_data"
"Show correlation matrix for all numeric columns"
```

#### **Missing Data Visualization**
```
"Visualize missing data patterns in sales_data"
"Show missing data matrix"
```

---

## **🛡️ Data Quality Analysis**

### **Missing Data Analysis**
```
"Analyze missing data in sales_data"
"What's the missing data pattern?"
"Show me missing data recommendations"
```

**Provides**:
- Missing data percentages by column
- Missing data patterns and clustering
- Little's test for missing mechanism (MCAR/MAR/MNAR)
- Imputation strategy recommendations
- Quality impact assessment

### **Schema Inference**
```
"Infer schema for sales_data"
"What data types are detected in customers?"
"Show me column patterns and constraints"
```

**Detects**:
- Data types (numeric, string, datetime)
- Patterns (email, phone, UUID, etc.)
- Constraints (min/max values, unique counts)
- Sample values and nullability

### **Data Quality Reports**
```
"Generate data quality report for sales_data"
"Create comprehensive quality assessment"
```

**Includes**:
- Interactive HTML reports
- Quality scoring (0-100%)
- Distribution analysis
- Completeness assessment

---

## **🔍 Outlier Detection**

### **Detection Methods**

#### **IQR Method (Default)**
```
"Detect outliers in sales_data using IQR"
"Find outliers with factor 2.0"
```

#### **Isolation Forest**
```
"Detect outliers using isolation forest"
"Find anomalies with 5% contamination"
```

#### **Local Outlier Factor (LOF)**
```
"Detect outliers using LOF method"
"Find outliers with LOF and 10% contamination"
```

### **Parameters**
- **Factor**: IQR multiplier (default: 1.5)
- **Contamination**: Expected outlier fraction (default: 0.05)
- **Method**: `iqr`, `isolation_forest`, `lof`

---

## **📈 Model Performance & Monitoring**

### **Drift Analysis**
```
"Compare baseline_data vs current_data for drift"
"Analyze drift between jan_sales and feb_sales"
"What's the drift count and share?"
```

**Provides**:
- Number of drifted columns
- Drift percentage
- Statistical significance tests
- Interactive HTML reports

### **Model Performance Evaluation**

#### **Regression Models**
```
"Evaluate regression performance with my predictions"
# You'll need to provide y_true and y_pred lists
```

**Metrics**: RMSE, MAE, R²

#### **Classification Models**
```
"Evaluate classification performance with my predictions"
# You'll need to provide y_true and y_pred lists
```

**Metrics**: Accuracy, Precision, Recall, F1-score (weighted)

---

## **🔧 Feature Engineering**

### **Available Transformations**

#### **Box-Cox Transformation**
```
"Apply Box-Cox transformation to sales_data"
"Transform skewed columns with Box-Cox"
```

#### **Logarithmic Transformation**
```
"Apply log transformation to sales_data"
"Transform with log and automatic shifting"
```

#### **Binning**
```
"Apply quantile binning to sales_data"
"Discretize continuous variables"
```

#### **Cardinality Reduction**
```
"Reduce cardinality in categorical columns"
"Group rare categories (threshold 0.5%)"
```

### **Multiple Transformations**
```
"Apply Box-Cox, log, and binning transformations to sales_data"
"Transform features excluding target_column"
```

---

## **💡 Advanced Usage**

### **Batch Operations**
```
"Load sales_jan.csv, sales_feb.csv, and sales_mar.csv"
"Compare all three datasets for drift"
"Generate quality reports for all loaded datasets"
```

### **Custom Parameters**
```
"Detect outliers with IQR factor 2.0"
"Analyze drift with 10% contamination"
"Create histogram with 50 bins"
```

### **Multi-step Analysis**
```
"Load customer_data.csv"
"Analyze missing data patterns"
"Detect outliers using isolation forest"
"Generate comprehensive quality report"
"Create correlation heatmap"
```

---

## **📊 Output Formats**

### **Text Reports**
- Statistical summaries
- Dataset overviews
- Quality assessments
- Outlier counts

### **Visualizations**
- PNG images saved to `reports/` directory
- Base64 encoded images in responses
- Interactive plots for missing data

### **HTML Reports**
- Data quality reports (Evidently)
- Drift analysis reports
- Model performance reports
- Saved to `reports/` directory

### **Structured Data**
- Schema definitions (YAML contracts)
- Outlier indices and counts
- Performance metrics
- Quality scores

---

## **🔧 Configuration**

### **Viewing Current Configuration**
```
"What are the current quality thresholds?"
"Show me the outlier detection parameters"
```

### **Common Settings**
- **Column drop threshold**: 50% missing (configurable)
- **Correlation sample size**: 10,000 rows (configurable)
- **Outlier contamination**: 5% (configurable)
- **Visualization sample size**: 10,000 rows (configurable)

---

## **🚨 Error Handling**

### **Common Errors**

#### **Dataset Not Found**
```
Error: "Dataset 'sales_data' not found"
Solution: "Load sales_data.csv first"
```

#### **Incompatible Data Types**
```
Error: "No numeric columns found"
Solution: Check data types with "Show basic info"
```

#### **Memory Issues**
```
Error: "Dataset too large for processing"
Solution: Use sampling or reduce dataset size
```

### **Data Quality Issues**
```
Warning: "High missing data percentage (>50%)"
Action: Review missing data analysis recommendations
```

---

## **💡 Best Practices**

### **Data Exploration Workflow**
1. **Load data** and check basic info
2. **Analyze missing data** patterns first
3. **Generate quality report** for comprehensive assessment
4. **Visualize distributions** before analysis
5. **Detect outliers** and understand their impact

### **Model Monitoring Workflow**
1. **Load baseline** and current datasets
2. **Analyze drift** between datasets
3. **Generate drift report** for detailed analysis
4. **Monitor key metrics** over time
5. **Set up alerts** for significant drift

### **Performance Tips**
- **Sample large datasets** (>10,000 rows) for visualization
- **Use appropriate outlier methods** (IQR for simple, ML for complex)
- **Monitor memory usage** with basic info
- **Save important reports** for future reference

---

## **📚 Example Workflows**

### **New Dataset Analysis**
```
1. "Load customer_data.csv"
2. "Show me basic info about the dataset"
3. "Analyze missing data patterns"
4. "Generate data quality report"
5. "Create correlation heatmap"
6. "Detect outliers using IQR method"
```

### **Model Performance Monitoring**
```
1. "Load baseline_model_data.csv and current_model_data.csv"
2. "Compare baseline_model_data vs current_model_data for drift"
3. "What's the drift count and share?"
4. "Generate drift analysis report"
```

### **Data Quality Assessment**
```
1. "Load transaction_data.csv"
2. "Infer schema for all columns"
3. "Analyze missing data with recommendations"
4. "Create missing data visualization"
5. "Generate comprehensive quality report"
```

---

## **🔗 Integration with Claude Desktop**

### **Natural Language Interface**
- Use **conversational language** for all requests
- **Ask follow-up questions** for clarification
- **Request specific visualizations** or analyses
- **Combine multiple operations** in one request

### **File Management**
- **Files are referenced by name** (not full path)
- **Once loaded, datasets persist** in the session
- **Use clear, descriptive names** for datasets
- **List loaded datasets** anytime with `"What datasets are loaded?"`

### **Interactive Analysis**
- **Build on previous results** in conversation
- **Ask for explanations** of findings
- **Request different visualizations** of the same data
- **Modify parameters** and re-run analyses

---

## **🆘 Getting Help**

### **In-Chat Help**
```
"What tools are available?"
"How do I detect outliers?"
"What parameters can I use for drift analysis?"
```

### **Documentation**
- **Installation Guide**: [INSTALLATION.md](INSTALLATION.md)
- **Examples**: [EXAMPLES.md](EXAMPLES.md)
- **Configuration**: [CONFIGURATION.md](CONFIGURATION.md)
- **API Reference**: [API.md](API.md)

### **Troubleshooting**
- **Check server logs** in `reports/` directory
- **Run diagnostics** with `python verify_setup.py`
- **Test individual tools** with test scripts
- **Restart server** if experiencing issues

---

**🚀 Ready to start analyzing your data? Load a dataset and begin exploring!** 