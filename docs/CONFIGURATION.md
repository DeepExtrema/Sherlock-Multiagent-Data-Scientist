# ⚙️ **Configuration Guide**

This guide explains how to customize and configure the Deepline MCP Server for your specific needs.

## 📋 **Configuration Overview**

The Deepline server uses a layered configuration approach:

1. **Default settings** (built-in)
2. **Configuration file** (`config.yaml`)
3. **Environment variables** (runtime)
4. **Command-line arguments** (session-specific)

---

## 🔧 **Core Configuration File**

### **Location**
```
mcp-server/config.yaml
```

### **Structure**
```yaml
# Data Processing Configuration
data_processing:
  # Maximum rows for correlation analysis
  correlation_sample_size: 10000
  
  # Maximum rows for visualization
  visualization_sample_size: 10000
  
  # Memory usage thresholds
  memory_warning_threshold_mb: 500
  memory_error_threshold_mb: 1000

# Data Quality Configuration
data_quality:
  # Missing data thresholds
  column_drop_threshold: 0.5  # 50% missing
  row_drop_threshold: 0.8     # 80% missing
  
  # Quality scoring weights
  completeness_weight: 0.3
  consistency_weight: 0.2
  validity_weight: 0.3
  uniqueness_weight: 0.2

# Outlier Detection Configuration
outlier_detection:
  # IQR method parameters
  iqr_factor: 1.5
  
  # Isolation Forest parameters
  isolation_forest_contamination: 0.05
  isolation_forest_n_estimators: 100
  
  # LOF parameters
  lof_contamination: 0.05
  lof_n_neighbors: 20

# Visualization Configuration
visualization:
  # Plot dimensions
  figure_width: 12
  figure_height: 8
  
  # Color schemes
  color_palette: "viridis"
  
  # DPI for saved images
  dpi: 300

# Reporting Configuration
reporting:
  # Output directory
  output_dir: "reports"
  
  # File naming convention
  timestamp_format: "%Y%m%d_%H%M%S"
  
  # Report formats
  html_reports: true
  json_reports: true
  
# Performance Configuration
performance:
  # Async processing
  max_concurrent_operations: 4
  
  # Timeout settings
  operation_timeout_seconds: 300
  
  # Caching
  enable_caching: true
  cache_size_mb: 100

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/deepline.log"
  max_size_mb: 10
  backup_count: 5
```

---

## 🎛️ **Customization Options**

### **Data Processing Settings**

#### **Memory Management**
```yaml
data_processing:
  # Adjust based on your system resources
  correlation_sample_size: 5000    # Reduce for limited memory
  visualization_sample_size: 5000  # Reduce for limited memory
  
  # Memory thresholds
  memory_warning_threshold_mb: 250  # Lower for constrained systems
  memory_error_threshold_mb: 500    # Lower for constrained systems
```

#### **Sample Size Optimization**
```yaml
# For large datasets (>1M rows)
data_processing:
  correlation_sample_size: 50000
  visualization_sample_size: 20000

# For small datasets (<10K rows)
data_processing:
  correlation_sample_size: 1000
  visualization_sample_size: 1000
```

### **Data Quality Tuning**

#### **Missing Data Tolerance**
```yaml
data_quality:
  # Strict quality requirements
  column_drop_threshold: 0.2  # Drop columns with >20% missing
  row_drop_threshold: 0.3     # Drop rows with >30% missing
  
  # Lenient quality requirements
  column_drop_threshold: 0.8  # Drop columns with >80% missing
  row_drop_threshold: 0.9     # Drop rows with >90% missing
```

#### **Quality Scoring Weights**
```yaml
data_quality:
  # Emphasize completeness
  completeness_weight: 0.5
  consistency_weight: 0.2
  validity_weight: 0.2
  uniqueness_weight: 0.1
  
  # Emphasize validity
  completeness_weight: 0.2
  consistency_weight: 0.2
  validity_weight: 0.5
  uniqueness_weight: 0.1
```

### **Outlier Detection Tuning**

#### **Conservative Settings (Fewer Outliers)**
```yaml
outlier_detection:
  iqr_factor: 2.0  # More conservative
  isolation_forest_contamination: 0.02  # Expect fewer outliers
  lof_contamination: 0.02
```

#### **Aggressive Settings (More Outliers)**
```yaml
outlier_detection:
  iqr_factor: 1.0  # More aggressive
  isolation_forest_contamination: 0.1  # Expect more outliers
  lof_contamination: 0.1
```

### **Visualization Customization**

#### **High-Quality Publication Settings**
```yaml
visualization:
  figure_width: 16
  figure_height: 10
  dpi: 600
  color_palette: "Set1"
```

#### **Fast Preview Settings**
```yaml
visualization:
  figure_width: 8
  figure_height: 6
  dpi: 150
  color_palette: "tab10"
```

---

## 🌐 **Environment Variables**

### **Common Environment Variables**
```powershell
# Set output directory
$env:DEEPLINE_OUTPUT_DIR = "C:\my_reports"

# Set log level
$env:DEEPLINE_LOG_LEVEL = "DEBUG"

# Set memory limits
$env:DEEPLINE_MEMORY_LIMIT_MB = "2048"

# Enable/disable caching
$env:DEEPLINE_ENABLE_CACHE = "true"

# Set number of workers
$env:DEEPLINE_MAX_WORKERS = "8"
```

### **Development Environment**
```powershell
# Enable debug mode
$env:DEEPLINE_DEBUG = "true"

# Enable verbose logging
$env:DEEPLINE_LOG_LEVEL = "DEBUG"

# Disable caching for development
$env:DEEPLINE_ENABLE_CACHE = "false"
```

### **Production Environment**
```powershell
# Production logging
$env:DEEPLINE_LOG_LEVEL = "INFO"

# Optimize for production
$env:DEEPLINE_ENABLE_CACHE = "true"
$env:DEEPLINE_MAX_WORKERS = "16"
$env:DEEPLINE_MEMORY_LIMIT_MB = "8192"
```

---

## 🚀 **Performance Tuning**

### **High-Performance Configuration**
```yaml
performance:
  max_concurrent_operations: 8
  operation_timeout_seconds: 600
  enable_caching: true
  cache_size_mb: 500

data_processing:
  correlation_sample_size: 100000
  visualization_sample_size: 50000
  memory_warning_threshold_mb: 2000
  memory_error_threshold_mb: 4000
```

### **Low-Resource Configuration**
```yaml
performance:
  max_concurrent_operations: 2
  operation_timeout_seconds: 180
  enable_caching: false
  cache_size_mb: 50

data_processing:
  correlation_sample_size: 2000
  visualization_sample_size: 1000
  memory_warning_threshold_mb: 100
  memory_error_threshold_mb: 200
```

### **Balanced Configuration**
```yaml
performance:
  max_concurrent_operations: 4
  operation_timeout_seconds: 300
  enable_caching: true
  cache_size_mb: 100

data_processing:
  correlation_sample_size: 10000
  visualization_sample_size: 10000
  memory_warning_threshold_mb: 500
  memory_error_threshold_mb: 1000
```

---

## 📊 **Industry-Specific Configurations**

### **Healthcare Analytics**
```yaml
data_quality:
  # Strict requirements for patient data
  column_drop_threshold: 0.1
  row_drop_threshold: 0.2
  
  # Emphasize completeness and validity
  completeness_weight: 0.4
  consistency_weight: 0.2
  validity_weight: 0.4
  uniqueness_weight: 0.0

outlier_detection:
  # Conservative outlier detection
  iqr_factor: 2.0
  isolation_forest_contamination: 0.02
  lof_contamination: 0.02
```

### **Financial Services**
```yaml
data_quality:
  # Moderate requirements
  column_drop_threshold: 0.3
  row_drop_threshold: 0.5
  
  # Emphasize consistency and validity
  completeness_weight: 0.2
  consistency_weight: 0.4
  validity_weight: 0.3
  uniqueness_weight: 0.1

outlier_detection:
  # Aggressive fraud detection
  iqr_factor: 1.0
  isolation_forest_contamination: 0.1
  lof_contamination: 0.1
```

### **Manufacturing IoT**
```yaml
data_processing:
  # Large sensor datasets
  correlation_sample_size: 50000
  visualization_sample_size: 20000

data_quality:
  # Lenient for sensor data
  column_drop_threshold: 0.6
  row_drop_threshold: 0.8
  
  # Emphasize consistency
  completeness_weight: 0.1
  consistency_weight: 0.6
  validity_weight: 0.2
  uniqueness_weight: 0.1

outlier_detection:
  # Moderate anomaly detection
  iqr_factor: 1.5
  isolation_forest_contamination: 0.05
  lof_contamination: 0.05
```

---

## 🔍 **Advanced Configuration**

### **Custom Thresholds**
```yaml
# Custom drift detection thresholds
drift_detection:
  p_value_threshold: 0.05
  effect_size_threshold: 0.1
  
# Custom schema inference settings
schema_inference:
  uniqueness_threshold: 0.95
  pattern_confidence_threshold: 0.8
  
# Custom feature transformation settings
feature_transformation:
  cardinality_threshold: 0.005  # 0.5%
  vif_threshold: 10.0
  skewness_threshold: 0.5
```

### **Integration Settings**
```yaml
# Claude Desktop integration
claude_integration:
  max_response_length: 10000
  include_images: true
  base64_encode_images: true
  
# External tool integration
external_tools:
  enable_jupyter: false
  enable_plotly: true
  enable_dash: false
```

### **Security Settings**
```yaml
security:
  # File access restrictions
  allowed_file_extensions: [".csv", ".xlsx", ".json"]
  max_file_size_mb: 100
  
  # Path restrictions
  allowed_paths: [".", "./data", "./uploads"]
  forbidden_paths: ["C:\\Windows", "C:\\Program Files"]
```

---

## 🔄 **Configuration Management**

### **Loading Custom Configuration**
```powershell
# Use custom config file
$env:DEEPLINE_CONFIG_FILE = "my_custom_config.yaml"
python launch_server.py
```

### **Validating Configuration**
```powershell
# Validate configuration
python -c "from config import load_config; print(load_config())"
```

### **Configuration Backup**
```powershell
# Backup current configuration
copy config.yaml config_backup_$(Get-Date -Format "yyyyMMdd_HHmmss").yaml
```

### **Resetting to Defaults**
```powershell
# Reset to default configuration
python -c "from config import create_default_config; create_default_config()"
```

---

## 🐛 **Configuration Troubleshooting**

### **Common Issues**

#### **Invalid YAML Syntax**
```
Error: yaml.scanner.ScannerError: mapping values are not allowed here
```
**Solution**: Check YAML indentation and syntax

#### **Missing Configuration File**
```
Error: FileNotFoundError: config.yaml not found
```
**Solution**: Create default configuration or check file path

#### **Invalid Configuration Values**
```
Error: ValidationError: correlation_sample_size must be positive
```
**Solution**: Check value ranges and data types

### **Debug Configuration**
```powershell
# Enable configuration debugging
$env:DEEPLINE_DEBUG_CONFIG = "true"
python launch_server.py
```

### **Configuration Validation**
```powershell
# Validate configuration
python -c "
from config import load_config, validate_config
try:
    config = load_config()
    validate_config(config)
    print('✅ Configuration valid')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

---

## 📚 **Configuration Examples**

### **Example 1: Data Science Team**
```yaml
# Optimized for data science workflows
data_processing:
  correlation_sample_size: 25000
  visualization_sample_size: 15000

data_quality:
  column_drop_threshold: 0.4
  completeness_weight: 0.3
  validity_weight: 0.4

outlier_detection:
  iqr_factor: 1.5
  isolation_forest_contamination: 0.05

visualization:
  figure_width: 14
  figure_height: 10
  dpi: 300
```

### **Example 2: Production Monitoring**
```yaml
# Optimized for production monitoring
performance:
  max_concurrent_operations: 8
  enable_caching: true
  cache_size_mb: 200

data_quality:
  column_drop_threshold: 0.3
  consistency_weight: 0.5

outlier_detection:
  isolation_forest_contamination: 0.03
  lof_contamination: 0.03

reporting:
  html_reports: true
  json_reports: true
```

### **Example 3: Educational Use**
```yaml
# Optimized for learning and education
data_processing:
  correlation_sample_size: 5000
  visualization_sample_size: 3000

data_quality:
  column_drop_threshold: 0.6
  completeness_weight: 0.4

visualization:
  figure_width: 10
  figure_height: 8
  color_palette: "bright"

logging:
  level: "DEBUG"
```

---

**⚙️ Need help with configuration? Check the [Troubleshooting Guide](USER_GUIDE.md#troubleshooting) or reach out via [GitHub Discussions](https://github.com/your-org/deepline/discussions).** 