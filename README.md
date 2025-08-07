# Deepline - AI-Powered MLOps Platform

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%7C%20BUSL%201.1-green.svg)](LICENSE.md)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/your-org/deepline)
[![Version](https://img.shields.io/badge/Version-2.1.0-orange.svg)](https://github.com/your-org/deepline/releases)

**Deepline** is a comprehensive AI-powered MLOps platform that automates machine learning workflows through intelligent agent orchestration. The platform combines natural language processing, specialized AI agents, and real-time monitoring to streamline the entire ML lifecycle from data analysis to model deployment.

## 🎯 **Overview**

Deepline revolutionizes machine learning operations by providing:

- **🤖 Natural Language Interface** - Convert plain English requests into executable ML workflows
- **🔄 Intelligent Agent Orchestration** - Coordinate specialized AI agents for data analysis, feature engineering, and model training
- **🛡️ Production Reliability** - Deadlock monitoring, graceful cancellation, and comprehensive error handling
- **📊 Real-Time Observability** - Interactive dashboard for monitoring workflows, metrics, and system health
- **🔒 Enterprise Security** - Rate limiting, authentication, and secure data handling

### **🚀 Key Features**

- **🔄 Natural Language Processing** - Convert user requests to structured workflows
- **🛡️ Deadlock Detection** - Automatic identification and recovery from stuck workflows
- **⚡ Workflow Management** - Start, monitor, and cancel workflows through API
- **🧠 Intelligent Scheduling** - Priority-based task execution with retry logic
- **📊 Real-time Dashboard** - Live monitoring with metrics and performance tracking
- **🔒 Security & Rate Limiting** - Production-grade protection and access control
- **🤖 Specialized AI Agents** - EDA, ML, Refinery, and custom agents for different tasks
- **📈 Complete ML Pipeline** - End-to-end automation from data analysis to model deployment

## 🏗️ **Architecture & Core Components**

Deepline follows a microservices architecture with a central orchestrator coordinating specialized AI agents:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              User Interface Layer                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  • React Dashboard (Real-time monitoring & control)                                    │
│  • REST API (Workflow management & status)                                             │
│  • CLI Tools (Command-line interface)                                                  │
│  • SDK (Python client library)                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           Master Orchestrator Service                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  • Natural Language Processing (Convert requests to workflows)                         │
│  • Workflow Management (Start, monitor, cancel workflows)                              │
│  • Task Scheduling (Priority-based execution)                                          │
│  • Deadlock Monitor (Detect and recover stuck workflows)                               │
│  • Security & Rate Limiting (Access control and protection)                            │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              Specialized AI Agents                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  📊 EDA Agent          │  🤖 ML Agent           │  🔧 Refinery Agent    │  🎯 Custom Agents │
│  • Data Analysis       │  • Model Training      │  • Feature Engineering│  • Domain-specific │
│  • Visualizations      │  • Hyperparameter Tune │  • Data Quality       │  • Custom Logic    │
│  • Schema Inference    │  • Experiment Tracking │  • Drift Detection    │  • Integration     │
│  • Outlier Detection   │  • Model Evaluation    │  • Pipeline Validation│  • Extensions      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           Observability & Control                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  • Interactive Dashboard (Real-time workflow monitoring)                               │
│  • Metrics Collection (Prometheus integration)                                         │
│  • Event Streaming (Kafka-based real-time events)                                      │
│  • Health Monitoring (System status and alerts)                                        │
│  • Workflow Control (Start, pause, cancel operations)                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### **🤖 AI Agents**

- **📊 EDA Agent** - Exploratory Data Analysis with automated insights, visualizations, and data quality assessment
- **🔧 Feature Engineering Agent** - Automated feature creation, selection, and transformation pipelines
- **🤖 ML Agent** - Complete ML workflow including class imbalance handling, model training, and experiment tracking
- **🔍 Refinery Agent** - Data quality monitoring, drift detection, and pipeline validation
- **🎯 Custom Agents** - Extensible framework for domain-specific AI agents

### **🧠 Core Orchestration Components**

- **Master Orchestrator** - Central workflow coordination and decision engine
- **Workflow Engine** - Priority-based task scheduling and execution
- **Translation Queue** - Async natural language to DSL conversion
- **Deadlock Monitor** - Automatic detection and recovery from stuck workflows
- **SLA Monitor** - Performance tracking and timeout management

## 🔧 **Key Capabilities**

### **🔄 Natural Language to Workflow Translation**
Convert natural language requests into executable ML workflows:
```bash
# Submit translation request
curl -X POST "http://localhost:8000/workflows/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Create a workflow that processes data and generates reports"}'

# Poll for results
curl "http://localhost:8000/translation/{token}"
```

### **📊 Automated Data Analysis**
- **Exploratory Data Analysis** - Automated insights, visualizations, and data quality reports
- **Schema Inference** - Automatic data type detection and validation
- **Outlier Detection** - Multiple algorithms (IQR, Isolation Forest, LOF)
- **Missing Data Analysis** - Pattern detection and imputation strategies

### **🤖 Machine Learning Workflows**
- **Class Imbalance Handling** - SMOTE, ADASYN, and other sampling strategies
- **Model Training** - Cross-validation, hyperparameter tuning, and overfitting detection
- **Baseline Models** - Random, majority, and naïve Bayes baselines
- **Experiment Tracking** - MLflow integration for reproducibility

### **🛡️ Production Reliability**
- **Deadlock Detection** - Automatic identification and recovery from stuck workflows
- **Graceful Cancellation** - Multi-endpoint API for workflow management
- **Retry Logic** - Configurable retry policies with exponential backoff
- **SLA Monitoring** - Performance tracking and timeout management

### **📈 Real-Time Observability**
- **Live Dashboard** - React-based monitoring interface
- **Metrics Collection** - Prometheus integration for system metrics
- **Event Streaming** - Kafka-based real-time event processing
- **Health Monitoring** - Comprehensive health checks and alerting

## 🚀 **Installation Instructions**

### **Prerequisites**
- **Python** 3.12 or higher
- **Docker** and Docker Compose
- **Redis** 6.0+ (or Docker)
- **MongoDB** 5.0+ (or Docker)
- **Kafka** (optional, for advanced event streaming)

### **Quick Start**

#### **1. Clone Repository**
```bash
git clone https://github.com/your-org/deepline.git
cd deepline
```

#### **2. Start Infrastructure Services**
```bash
# Start Redis, MongoDB, and Kafka
docker-compose up -d
```

#### **3. Install Dependencies**
```bash
cd mcp-server
pip install -r requirements-exact.txt
```

#### **4. Configure Environment**
```bash
# Copy and edit configuration
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
```

#### **5. Start Services**
```bash
# Start the orchestrator
python master_orchestrator_api.py

# Start the dashboard (optional)
cd ../dashboard
npm install
npm start
```

### **Development Setup**
```bash
# Install with development tools
pip install -r requirements-exact.txt
pip install pytest black ruff mypy

# Run tests
python -m pytest tests/ -v

# Format code
black .
ruff check .
```

## 📦 **Dependencies**

### **Core Framework**
- **FastAPI** - Modern web framework for building APIs
- **Pydantic** - Data validation and settings management
- **MCP** - Model Context Protocol for AI model interactions

### **Data Science & ML**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **scipy** - Scientific computing
- **evidently** - ML model monitoring
- **ydata-profiling** - Automated data quality reports

### **Infrastructure**
- **Redis** - Caching and message queuing
- **MongoDB** - Document database for workflow state
- **Kafka** - Event streaming (optional)
- **Prometheus** - Metrics collection

### **Development & Monitoring**
- **OpenTelemetry** - Distributed tracing
- **MLflow** - Experiment tracking
- **Docker** - Containerization
- **React** - Frontend dashboard

### **External Services (Optional)**
- **OpenAI API** - LLM integration for translation
- **Claude API** - Alternative LLM provider
- **Slack/PagerDuty** - Alerting and notifications

## 🔧 **Configuration**

### **Environment Variables**
```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key

# Infrastructure
REDIS_URL=redis://localhost:6379
MONGODB_URL=mongodb://localhost:27017
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Security
SECRET_KEY=your_secret_key
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### **Configuration File (`config.yaml`)**
```yaml
master_orchestrator:
  llm:
    model_version: "claude-3-sonnet-20240229"
    max_input_length: 10000
    temperature: 0.0
  
  orchestrator:
    max_concurrent_workflows: 10
    deadlock:
      check_interval_s: 60
      pending_stale_s: 900
      cancel_on_deadlock: true
    
    retry:
      max_retries: 3
      backoff_base_s: 30
```

## 📊 **Usage Examples**

### **Basic Workflow Execution**
```python
import requests

# Submit workflow
response = requests.post("http://localhost:8000/workflows/dsl", json={
    "dsl": """
    workflow:
      name: data_analysis_workflow
      tasks:
        - name: load_data
          agent: eda
          action: load_data
          params: {"file": "data.csv"}
        - name: analyze_data
          agent: eda
          action: analyze_data
          depends_on: [load_data]
    """,
    "client_id": "client123"
})

run_id = response.json()["run_id"]

# Check status
status = requests.get(f"http://localhost:8000/runs/{run_id}/status")
print(status.json())
```

### **Natural Language Workflow**
```python
# Submit natural language request
response = requests.post("http://localhost:8000/workflows/translate", json={
    "text": "Analyze the customer data, detect outliers, and train a classification model",
    "client_id": "client123"
})

token = response.json()["token"]

# Poll for completion
while True:
    result = requests.get(f"http://localhost:8000/translation/{token}")
    if result.json()["status"] == "done":
        dsl = result.json()["dsl"]
        break
    time.sleep(5)
```

### **Workflow Cancellation**
```python
# Cancel a running workflow
requests.put(f"http://localhost:8000/runs/{run_id}/cancel", json={
    "reason": "user-requested",
    "force": False
})
```

## 📁 **Project Structure**

```
Deepline/
├── mcp-server/                          # Core orchestration engine
│   ├── api/                            # FastAPI routers
│   │   ├── hybrid_router.py           # Translation API endpoints
│   │   ├── cancel_router.py           # Cancellation API endpoints
│   │   └── agent_router.py            # Agent management endpoints
│   ├── orchestrator/                   # Core orchestration logic
│   │   ├── translation_queue.py       # Async translation system
│   │   ├── workflow_manager.py        # Workflow lifecycle management
│   │   ├── deadlock_monitor.py        # Deadlock detection & recovery
│   │   ├── guards.py                  # Security & rate limiting
│   │   └── sla_monitor.py             # SLA tracking
│   ├── workflow_engine/               # Task execution engine
│   │   ├── scheduler.py               # Priority-based task scheduling
│   │   ├── worker_pool.py             # Worker management with cancellation
│   │   └── retry_tracker.py           # Retry logic with Redis
│   ├── agents/                        # AI agent implementations
│   │   ├── eda_agent.py              # Exploratory Data Analysis
│   │   ├── ml_agent.py               # Machine Learning workflows
│   │   ├── refinery_agent.py         # Data quality & monitoring
│   │   └── custom_agent.py           # Extensible agent framework
│   ├── config.py                      # Configuration management
│   ├── config.yaml                    # System configuration
│   └── master_orchestrator_api.py     # Main FastAPI application
├── dashboard/                         # React-based monitoring UI
│   ├── backend/                      # FastAPI backend for dashboard
│   └── dashboard-frontend/           # React frontend
├── docs/                             # Comprehensive documentation
│   ├── DEADLOCK_MONITORING.md        # Deadlock system guide
│   ├── USER_GUIDE.md                 # User documentation
│   ├── INSTALLATION.md               # Setup instructions
│   ├── CONFIGURATION.md              # Configuration guide
│   └── CONTRIBUTING.md               # Development guidelines
├── docker-compose.yml                # Container orchestration
├── requirements-exact.txt            # Exact dependency versions
├── pyproject.toml                    # Project metadata
└── README.md                         # This file
```

## 📄 **Licensing**

Deepline uses a hybrid license approach to balance open innovation with business sustainability:

### **🔓 Apache 2.0 - SDK/Client Components**
**Components covered:**
- Client SDKs and libraries
- Integration examples
- Documentation and tutorials
- Testing frameworks
- Development tools

**You can:**
- ✅ Use commercially without restrictions
- ✅ Modify and distribute freely
- ✅ Create derivative works
- ✅ Grant patent rights
- ✅ Sell products using these components

### **🛡️ BUSL 1.1 - Core Server**
**Components covered:**
- Main MCP server and core orchestration engine
- Core analysis tools and algorithms
- Data processing pipeline
- Quality assessment engines
- Model performance monitoring

**You can:**
- ✅ Use for development and testing
- ✅ Use for non-commercial purposes
- ✅ Modify for internal use
- ✅ Contribute back to the project
- ⚠️ Commercial use requires agreement

**Restrictions:**
- ❌ Cannot use in competing commercial products
- ❌ Cannot offer as a service without permission
- ❌ Cannot redistribute commercially

### **🔄 Automatic Conversion**
**After 3 years (≈ 2027):**
- Core components automatically convert to Apache 2.0
- Full open-source availability
- No commercial restrictions
- Complete ecosystem freedom

For commercial licensing inquiries, contact: **licensing@deepline.ai**

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on:

- **Code Style** - Black formatting, Ruff linting, MyPy type checking
- **Testing** - Comprehensive test coverage with pytest
- **Documentation** - Clear docstrings and updated guides
- **Pull Request Process** - Review guidelines and merge criteria
- **Development Setup** - Local development environment

### **Development Workflow**
```bash
# Fork and clone
git clone https://github.com/your-username/deepline.git
cd deepline

# Create feature branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -r requirements-exact.txt
pip install pytest black ruff mypy

# Make changes and test
python -m pytest tests/ -v
black .
ruff check .

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

## 📞 **Contact & Support**

### **Technical Support**
- **GitHub Issues**: [Report bugs and feature requests](https://github.com/your-org/deepline/issues)
- **Discussions**: [Community support](https://github.com/your-org/deepline/discussions)
- **Documentation**: [Complete guides](https://deepline.ai/docs)

### **Commercial Support**
- **Licensing**: licensing@deepline.ai
- **Partnerships**: partnerships@deepline.ai
- **Enterprise**: enterprise@deepline.ai

### **Community**
- **Discord**: [Join our community](https://discord.gg/deepline)
- **Twitter**: [@deepline_ai](https://twitter.com/deepline_ai)
- **Blog**: [Latest updates](https://deepline.ai/blog)

## 🏆 **Production Status**

**✅ PRODUCTION READY** - Version 2.1.0

The Deepline platform is production-ready with:
- ✅ Comprehensive testing and validation
- ✅ Deadlock monitoring and graceful cancellation
- ✅ Hybrid API with async translation workflows
- ✅ Complete workflow engine with retry logic
- ✅ Security and rate limiting
- ✅ Real-time monitoring and alerting
- ✅ Extensive documentation and examples
- ✅ Enterprise-grade reliability and scalability

**Ready for enterprise deployment!** 🚀

---

**© 2024 Deepline. All rights reserved.**  
*Building the future of data analytics through sustainable open source innovation.*
