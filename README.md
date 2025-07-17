<div align="center">
  <img src="deepline-logo.png" alt="Deepline Logo" width="200"/>
</div>

# 🔬 **Deepline MCP Server**

> **End-to-End Data Science & MLOps Platform**  
> *Seamlessly integrate data analysis, quality monitoring, and model performance tracking with Claude Desktop*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Hybrid-green.svg)](#license)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)](https://microsoft.com/windows)
[![MCP](https://img.shields.io/badge/MCP-1.10.1-orange.svg)](https://modelcontextprotocol.io)

## 🚀 **Quick Start (Windows)**

### **Prerequisites**
- Windows 10/11
- Python 3.12 or higher
- Claude Desktop app

### **1. One-Click Setup**
```powershell
# Clone and setup
git clone https://github.com/your-org/deepline.git
cd deepline/mcp-server

# Install dependencies
pip install -r requirements-python313.txt

# Verify installation
python verify_setup.py
```

### **2. Launch Server**
```powershell
python launch_server.py
```

### **3. Connect to Claude Desktop**
Your Claude Desktop will automatically detect the server. Start with:
```
"Load the iris.csv dataset and show me basic info"
```

### **4. Launch Infrastructure (Optional)**
For advanced workflows with multi-agent orchestration:
```powershell
# Start Kafka, MongoDB, and Dashboard
docker-compose up -d

# Start Dashboard Backend (in new terminal)
cd dashboard/backend
python main.py

# Start Dashboard Frontend (in new terminal)
cd dashboard/dashboard-frontend
npm install
npm start
```

Access the dashboard at: `http://localhost:3000`

---

## 📊 **What This Does**

Deepline transforms your data science workflow by providing **17 powerful tools** accessible through natural language in Claude Desktop:

### **🔍 Data Exploration**
- **Load datasets** from CSV, Excel, JSON
- **Inspect data** with automatic profiling
- **Visualize patterns** with 5 plot types
- **Detect outliers** using 3 methods (IQR, Isolation Forest, LOF)

### **🛡️ Data Quality**
- **Missing data analysis** with 6-step pipeline
- **Schema inference** with pattern detection
- **Data quality reports** with Evidently
- **Automated quality scoring**

### **📈 Model Performance**
- **Drift detection** comparing datasets
- **Regression metrics** (RMSE, MAE, R²)
- **Classification metrics** (accuracy, precision, recall, F1)
- **Performance monitoring** with interactive reports

### **🔧 Feature Engineering**
- **Feature transformation** (Box-Cox, log, binning)
- **Cardinality reduction** for categorical variables
- **Multicollinearity detection** (VIF analysis)
- **Target-guided discretization**

---

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Claude Desktop │────│   MCP Protocol  │────│  Deepline Server │
│   (Frontend)     │    │   (Transport)   │    │   (Analytics)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │  Tool Ecosystem  │
                                               │  • EDA Tools     │
                                               │  • Quality Tools │
                                               │  • ML Tools      │
                                               │  • Viz Tools     │
                                               └─────────────────┘
```

## 🔧 **Infrastructure Components**

### **🚀 Messaging Backbone: Kafka (Local)**
- **Apache Kafka**: High-throughput event streaming for inter-component communication
- **Zookeeper**: Coordination service for Kafka cluster management
- **Topics**: `workflow.commands`, `task.requests`, `task.events`
- **Docker Compose**: Zero-cost local deployment

### **🗄️ Persistent Storage: MongoDB (Local)**
- **MongoDB 6.0**: Document database for workflow and task metadata
- **Motor Driver**: Async Python driver for high-performance operations
- **Collections**: `runs`, `tasks`, `workflows`, `agents`
- **Automatic Indexing**: Optimized queries on `run_id` and `task_id`

### **☸️ Container Orchestration: Kubernetes (Local)**
- **Kind Cluster**: Local Kubernetes development environment
- **Resource Quotas**: CPU and memory limits for cost control
- **Agent Deployment**: Containerized agent instances
- **Parallel Execution**: Multi-agent workflow processing

### **📊 Observability Dashboard**
- **FastAPI Backend**: REST API with WebSocket support for real-time events
- **React Frontend**: Interactive dashboard with live agent monitoring
- **Real-time Updates**: Live event streaming and agent status tracking
- **Workflow Management**: Visual workflow templates and execution control

### **⚙️ Orchestrator & Workload Management**
- **Concurrency Control**: Global workflow limits (configurable, default 1 concurrent workflow)
- **Intelligent Retry Logic**: Exponential backoff with configurable max retries (3) and delays (30-300s)
- **SLA Monitoring**: Automated detection of stale tasks (10min) and workflows (1hr) with real-time alerts
- **Workload Estimation**: Pre-configured capacity planning (30 tasks/hour, 4min avg duration)
- **Quality Gates**: Automatic workflow throttling prevents system overload
- **Real-time Alerting**: Live SLA violation notifications via WebSocket dashboard

**Benefits:**
- **System Stability**: Prevents resource exhaustion through controlled concurrency
- **Reliability**: Automatic retry with smart backoff reduces transient failures  
- **Observability**: Proactive SLA monitoring catches performance issues early
- **Predictability**: Workload estimates enable better capacity planning

---

## 🛠️ **Installation Guide**

### **Method 1: Standard Installation (Recommended)**
```powershell
# 1. Create project directory
mkdir deepline-workspace
cd deepline-workspace

# 2. Clone repository
git clone https://github.com/your-org/deepline.git
cd deepline/mcp-server

# 3. Install Python dependencies
pip install -r requirements-python313.txt

# 4. Test installation
python -c "import pandas, numpy, evidently, missingno; print('✅ All dependencies installed')"
```

### **Method 2: Virtual Environment (Isolated)**
```powershell
# 1. Create virtual environment
python -m venv deepline-env
deepline-env\Scripts\activate

# 2. Install in virtual environment
pip install -r requirements-python313.txt

# 3. Verify installation
python verify_setup.py
```

### **Method 3: Development Setup**
```powershell
# 1. Install with development tools
pip install -r requirements-python313.txt
pip install pytest black ruff mypy

# 2. Run tests
python -m pytest test_*.py -v

# 3. Check code quality
black --check .
ruff check .
```

---

## 📋 **Claude Desktop Integration**

### **Automatic Setup**
The server automatically configures Claude Desktop. Just verify the connection:

```powershell
# Check Claude Desktop configuration
python verify_setup.py
```

### **Manual Setup (if needed)**
1. **Locate Claude Desktop config**: `%APPDATA%\Claude\claude_desktop_config.json`
2. **Configuration is automatically added** when you run the server
3. **Restart Claude Desktop** to load the new server

### **Verification**
Open Claude Desktop and try:
```
"Load the iris.csv dataset and show basic info"
```

You should see the server respond with data analysis results.

---

## 📊 **Tool Reference**

### **Data Loading & Inspection**
| Tool | Purpose | Example Usage |
|------|---------|---------------|
| `load_data` | Load CSV/Excel/JSON files | `"Load sales_data.csv"`  |
| `basic_info` | Show dataset overview | `"Show basic info for sales_data"` |
| `list_datasets` | List loaded datasets | `"What datasets are loaded?"` |

### **Data Quality Analysis**
| Tool | Purpose | Example Usage |
|------|---------|---------------|
| `missing_data_analysis` | Analyze missing patterns | `"Analyze missing data in sales_data"` |
| `infer_schema` | Detect data types & patterns | `"Infer schema for sales_data"` |
| `data_quality_report` | Generate quality report | `"Create quality report for sales_data"` |

### **Statistical Analysis**
| Tool | Purpose | Example Usage |
|------|---------|---------------|
| `statistical_summary` | Descriptive statistics | `"Get statistical summary of sales_data"` |
| `detect_outliers` | Find outliers using IQR/ML | `"Detect outliers in sales_data using IQR"` |
| `create_visualization` | Generate plots | `"Create histogram of sales_amount"` |

### **Model Performance**
| Tool | Purpose | Example Usage |
|------|---------|---------------|
| `drift_analysis` | Compare datasets for drift | `"Compare sales_jan vs sales_feb for drift"` |
| `model_performance_report` | Evaluate model metrics | `"Evaluate model performance for predictions"` |

### **Feature Engineering**
| Tool | Purpose | Example Usage |
|------|---------|---------------|
| `feature_transformation` | Transform features | `"Apply Box-Cox transformation to sales_data"` |

---

## 🔬 **Example Workflows**

### **Workflow 1: Data Exploration**
```
1. "Load the customer_data.csv dataset"
2. "Show me basic info about the dataset"
3. "Analyze missing data patterns"
4. "Create a correlation heatmap"
5. "Detect outliers using isolation forest"
```

### **Workflow 2: Data Quality Assessment**
```
1. "Load the transaction_data.csv dataset"
2. "Generate a data quality report"
3. "Infer the schema for all columns"
4. "Analyze missing data with recommendations"
5. "Create missing data visualization"
```

### **Workflow 3: Model Performance Monitoring**
```
1. "Load baseline_data.csv and current_data.csv"
2. "Analyze drift between baseline_data and current_data"
3. "Generate drift analysis report"
4. "Evaluate model performance with my predictions"
```

---

## 🎯 **Key Features**

### **🔍 Smart Data Analysis**
- **Automatic type detection** with 9 pattern types (email, phone, UUID, etc.)
- **Missing data clustering** with correlation analysis
- **Outlier detection** using 3 methods (IQR, Isolation Forest, LOF)
- **Distribution analysis** with skewness and kurtosis

### **🛡️ Data Quality Gates**
- **Quality scoring** (0-100%) with impact assessment
- **Automated recommendations** for data cleanup
- **Little's test** for missing data mechanism detection
- **Pattern validation** with regex matching

### **📈 Performance Monitoring**
- **Drift detection** with statistical tests
- **Model performance** evaluation (regression & classification)
- **Interactive HTML reports** with Evidently
- **Threshold-based alerting**

### **🔧 Advanced Engineering**
- **Feature transformation** with Box-Cox and log transforms
- **Cardinality reduction** for high-cardinality categories
- **VIF analysis** for multicollinearity detection
- **Target-guided discretization**

### **⚙️ Configuration Management**
- **Centralized config.yaml** for all system parameters
- **Orchestrator settings**: Concurrency limits, retry policies, SLA thresholds
- **EDA parameters**: Quality thresholds, visualization limits, performance settings
- **Hot-reloadable**: Configuration changes without server restart

---

## 🗂️ **Project Structure**

```
deepline/
├── mcp-server/                      # Core MCP Server
│   ├── server.py                    # Main MCP server (17 tools)
│   ├── launch_server.py             # Server launcher
│   ├── verify_setup.py              # Setup verification
│   ├── config.py                    # Configuration management
│   ├── config.yaml                  # Tunable parameters (EDA + Orchestrator)
│   ├── utils.py                     # Utility functions
│   ├── requirements-python313.txt   # Python dependencies
│   ├── iris.csv                     # Sample dataset
│   └── reports/                     # Generated reports
├── dashboard/                       # Observability Dashboard
│   ├── backend/
│   │   └── main.py                  # FastAPI server with WebSocket
│   ├── dashboard-frontend/          # React dashboard
│   │   ├── src/
│   │   │   ├── App.js              # Main dashboard component
│   │   │   └── App.css             # Dashboard styling
│   │   └── package.json            # Frontend dependencies
│   └── README.md                    # Dashboard documentation
├── docker-compose.yml               # Kafka, MongoDB, infrastructure
├── resource-quota.yaml              # Kubernetes resource limits
├── docs/                            # Documentation
├── tests/                           # Test files
├── LICENSE-APACHE                   # SDK/Client license
├── LICENSE-BUSL                     # Core license
└── README.md                        # This file
```

---

## 🏗️ **Infrastructure Setup**

### **🐳 Docker Infrastructure**
```powershell
# Start all infrastructure services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check service logs
docker-compose logs kafka
docker-compose logs mongo
```

### **📊 Dashboard Setup**
```powershell
# Backend setup (Terminal 1)
cd dashboard/backend
pip install fastapi uvicorn motor confluent-kafka
python main.py

# Frontend setup (Terminal 2)
cd dashboard/dashboard-frontend
npm install
npm start
```

### **☸️ Kubernetes Setup (Optional)**
```powershell
# Install kind (if not already installed)
# Windows: choco install kind
# Linux: curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.25.0/kind-linux-amd64

# Create local cluster
kind create cluster --name deepline

# Apply resource quota
kubectl apply -f resource-quota.yaml

# Verify cluster
kubectl get nodes
kubectl get quota
```

### **🔧 Kafka Topic Management**
```powershell
# Create topics manually (if needed)
docker exec -it $(docker ps -qf "ancestor=confluentinc/cp-kafka") bash
kafka-topics --create --topic workflow.commands --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
kafka-topics --create --topic task.requests --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
kafka-topics --create --topic task.events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
exit
```

### **🗄️ MongoDB Initialization**
```powershell
# Connect to MongoDB
docker exec -it $(docker ps -qf "ancestor=mongo:6.0") mongo

# Create indexes for performance
use deepline
db.runs.createIndex({ run_id: 1 })
db.tasks.createIndex({ run_id: 1, task_id: 1 })
exit
```

### **📈 Dashboard Features**
- **Multi-Agent Workflow Management**: Start and monitor complex workflows
- **Real-time Agent Status**: Live updates of agent states and activities
- **Interactive Charts**: Visual representation of event activity
- **Workflow Templates**: Pre-configured workflows for common tasks
- **Live Event Streaming**: Real-time event monitoring via WebSocket

---

## 🧪 **Testing**

### **Run All Tests**
```powershell
# Run comprehensive tests
python test_evidently_tools.py      # Data quality & drift tests
python test_infer_schema.py         # Schema inference tests
python test_model_performance.py    # Model performance tests
```

### **Quick Health Check**
```powershell
# Verify server is working
python verify_setup.py

# Test with sample data
python -c "
import asyncio
from server import load_data, basic_info
asyncio.run(load_data('iris.csv', 'test'))
result = asyncio.run(basic_info('test'))
print(result)
"
```

### **Test Claude Desktop Integration**
1. **Start server**: `python launch_server.py`
2. **Open Claude Desktop**
3. **Try**: `"Load iris.csv and show basic info"`
4. **Expected**: Data analysis results displayed

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **❌ "ModuleNotFoundError: No module named 'evidently'"**
```powershell
# Solution: Install missing dependencies
pip install -r requirements-python313.txt
```

#### **❌ "KeyError: Dataset not found"**
```powershell
# Solution: Check dataset name and load first
# List loaded datasets
python -c "from server import list_datasets; print(asyncio.run(list_datasets()))"
```

#### **❌ "Claude Desktop not connecting"**
```powershell
# Solution: Verify setup and restart Claude
python verify_setup.py
# Then restart Claude Desktop completely
```

#### **❌ "Permission denied writing to reports/"**
```powershell
# Solution: Check directory permissions
mkdir reports
# Or run as administrator if needed
```

### **Debug Mode**
```powershell
# Enable debug logging
$env:DEBUG = "true"
python launch_server.py
```

### **Infrastructure Issues**

#### **❌ "Docker services not starting"**
```powershell
# Check Docker is running
docker --version

# Reset Docker Compose
docker-compose down
docker-compose up -d

# Check service health
docker-compose ps
```

#### **❌ "Kafka connection failed"**
```powershell
# Verify Kafka is accessible
docker exec -it $(docker ps -qf "ancestor=confluentinc/cp-kafka") bash
kafka-topics --list --bootstrap-server localhost:9092
exit
```

#### **❌ "MongoDB connection refused"**
```powershell
# Check MongoDB status
docker exec -it $(docker ps -qf "ancestor=mongo:6.0") mongo --eval "db.adminCommand('ping')"
```

#### **❌ "Dashboard not loading"**
```powershell
# Check backend is running
curl http://localhost:8000/

# Check frontend dependencies
cd dashboard/dashboard-frontend
npm install
npm start
```

#### **❌ "WebSocket connection failed"**
```powershell
# Verify WebSocket endpoint
curl -H "Connection: Upgrade" -H "Upgrade: websocket" http://localhost:8000/ws/events
```

### **Get Help**
- **Check logs**: Look in `reports/` directory for error logs
- **Run diagnostics**: `python verify_setup.py`
- **Test tools individually**: Use test files in project root
- **Infrastructure logs**: `docker-compose logs [service-name]`

---

## 🔄 **Updates & Maintenance**

### **Update Dependencies**
```powershell
# Update to latest versions
pip install --upgrade -r requirements-python313.txt

# Verify after update
python verify_setup.py
```

### **Backup Configuration**
```powershell
# Backup your config (if customized)
copy config.yaml config.yaml.backup
```

### **Clean Installation**
```powershell
# Remove and reinstall
pip uninstall -y -r requirements-python313.txt
pip install -r requirements-python313.txt
```

---

## 💡 **Tips & Best Practices**

### **Performance**
- **Large datasets**: Use sampling features for >10,000 rows
- **Memory usage**: Monitor with `basic_info` memory reporting
- **Batch processing**: Load multiple datasets for comparative analysis

### **Data Quality**
- **Always run** `missing_data_analysis` before modeling
- **Use** `infer_schema` to validate data types
- **Check** `data_quality_report` for comprehensive assessment

### **Visualization**
- **Start with** correlation heatmaps for feature relationships
- **Use** `detect_outliers` with visualization for data cleaning
- **Generate** missing data patterns for understanding gaps

### **Model Monitoring**
- **Compare** baseline vs current with `drift_analysis`
- **Track** performance metrics over time
- **Set** quality gates with threshold-based alerts

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**
```powershell
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/your-username/deepline.git
cd deepline/mcp-server

# 3. Install development dependencies
pip install -r requirements-python313.txt
pip install pytest black ruff mypy

# 4. Run tests
python -m pytest test_*.py -v

# 5. Format code
black .
ruff check --fix .
```

---

## 📄 **License**

This project uses a **Hybrid License** approach:

- **SDK/Client Components**: [Apache License 2.0](LICENSE-APACHE)
- **Core Server**: [Business Source License 1.1](LICENSE-BUSL)
  - Converts to Apache 2.0 after 3 years
  - Allows unrestricted use for non-commercial purposes

See [LICENSE.md](LICENSE.md) for complete details.

---

## 🗺️ **Roadmap**

### **Phase 1: ✅ EDA Foundation** 
- Data loading, inspection, visualization
- Missing data analysis and quality assessment
- Basic statistical analysis and outlier detection

### **Phase 2: ✅ Quality & Monitoring**
- Data drift detection and performance monitoring
- Automated quality scoring and recommendations
- Interactive HTML reports with Evidently

### **Phase 3: ✅ Infrastructure & Orchestration**
- Kafka messaging backbone for event streaming
- MongoDB persistent storage for workflow metadata
- Kubernetes support for container orchestration
- Real-time observability dashboard with multi-agent workflow management

### **Phase 4: 🚧 Advanced Analytics** 
- Feature engineering and transformation
- Model training and evaluation
- Automated ML pipeline integration

### **Phase 5: 🔄 Production Ready**
- Advanced multi-agent orchestration
- API endpoints for external integration
- Scalable cloud deployment options
- Advanced monitoring and alerting

---

## 📞 **Support**

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/deepline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/deepline/discussions)

---

**Made with ❤️ for Data Scientists**  
*Transforming data analysis through natural language interfaces*