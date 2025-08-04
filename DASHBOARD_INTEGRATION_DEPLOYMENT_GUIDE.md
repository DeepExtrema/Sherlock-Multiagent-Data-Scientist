# Enhanced Deepline Dashboard - Complete Integration & Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB (local or cloud)
- Redis (optional, for caching)

### 1. Backend Setup

```bash
# Navigate to dashboard backend
cd dashboard/backend

# Install dependencies
pip install fastapi uvicorn motor httpx pydantic python-multipart

# Set environment variables
export MONGODB_URL="mongodb://localhost:27017/deepline_dashboard"
export JWT_SECRET_KEY="your-super-secret-jwt-key-here"
export ENCRYPTION_KEY="your-32-byte-encryption-key-here"

# Start the enhanced dashboard backend
uvicorn enhanced_dashboard:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup

```bash
# Navigate to dashboard frontend
cd dashboard/dashboard-frontend

# Install dependencies
npm install

# Start the React development server
npm start
```

### 3. Start Individual Agents

```bash
# Terminal 1: EDA Agent
cd mcp-server
python start_eda_service.py

# Terminal 2: Refinery Agent  
cd mcp-server
python refinery_agent.py

# Terminal 3: ML Agent
cd mcp-server
python ml_agent.py

# Terminal 4: Master Orchestrator
cd mcp-server
python start_master_orchestrator.py
```

## 🔧 Implementation Details

### Enhanced Dashboard Backend (`enhanced_dashboard.py`)

**Key Features:**
- Real-time agent health monitoring
- Workflow execution orchestration
- WebSocket event broadcasting
- MongoDB persistence
- Step-by-step workflow tracking

**API Endpoints:**
```python
GET /agents                    # Get agent configurations
GET /agents/health            # Get all agent health status
GET /agents/{name}/health     # Get specific agent health
GET /agents/{name}/metrics    # Get agent metrics
POST /agents/{name}/actions/{action}  # Execute agent action
POST /workflows/execute       # Start new workflow
GET /workflows/{id}          # Get workflow status
GET /workflows               # List all workflows
PUT /workflows/{id}/status   # Update workflow status
WS /ws/dashboard             # WebSocket for real-time updates
```

**Agent Configuration:**
```python
AGENT_CONFIG = {
    "eda_agent": {
        "url": "http://localhost:8001",
        "health_endpoint": "/health",
        "poll_interval": 30000,
        "actions": ["load_data", "basic_info", "statistical_summary"],
        "description": "Exploratory Data Analysis Agent",
        "color": "#48bb78"
    },
    # ... other agents
}
```

### Enhanced Dashboard Frontend (`EnhancedDashboard.js`)

**Key Features:**
- Real-time WebSocket connection
- Interactive agent status grid
- Workflow execution modal
- Step-by-step progress tracking
- Real-time logs display
- Agent metrics visualization

**State Management:**
```javascript
const [agents, setAgents] = useState({});
const [agentHealth, setAgentHealth] = useState({});
const [workflows, setWorkflows] = useState([]);
const [events, setEvents] = useState([]);
const [workflowActive, setWorkflowActive] = useState(false);
const [stepStates, setStepStates] = useState({});
const [activeWorkflowId, setActiveWorkflowId] = useState(null);
```

### Security Module (`authentication.py`)

**Key Features:**
- JWT-based authentication
- Role-based access control (RBAC)
- Secure credential management
- PII detection and anonymization
- Rate limiting
- Audit logging

**Roles & Permissions:**
```python
ROLES = {
    "admin": ["read", "write", "delete", "execute", "manage_users"],
    "ml_engineer": ["read", "write", "execute"],
    "data_scientist": ["read", "write"],
    "data_engineer": ["read", "write", "execute"],
    "viewer": ["read"]
}
```

## 🔐 Security Implementation

### 1. Create Users
```python
from security.authentication import AuthenticationManager

auth_manager = AuthenticationManager()
user = auth_manager.create_user(
    username="admin",
    email="admin@deepline.com",
    password="secure_password",
    role="admin"
)
```

### 2. Login & Get Token
```python
token = auth_manager.authenticate_user("admin", "secure_password")
# Returns: {"access_token": "...", "refresh_token": "..."}
```

### 3. Use Secure Configuration
```python
from security.authentication import SecureConfig

config = SecureConfig()
db_password = config.get_secure_value("DB_PASSWORD")
api_key = config.get_secure_value("API_KEY")
```

### 4. PII Detection
```python
from security.authentication import DataGovernance

governance = DataGovernance()
has_pii = governance.detect_pii(data)
if has_pii:
    anonymized_data = governance.anonymize_data(data)
```

## 📊 Dashboard Usage

### 1. Access Dashboard
- Open browser to `http://localhost:3000`
- Dashboard shows real-time agent health status
- Connection status indicator in top-right

### 2. Monitor Agent Health
- Agent cards show status (healthy/unhealthy/error)
- Click agent cards to view detailed metrics
- Use "View Metrics" button for detailed analytics
- Auto-refresh every 30 seconds

### 3. Execute Agent Actions
- Click "Health Check" on agent cards
- Actions are executed via API calls
- Results broadcast via WebSocket
- Real-time feedback in events list

### 4. Monitor Workflows
- Click "Start Full Pipeline" to open workflow modal
- Select workflow type (Data Analysis, ML Pipeline, Full Pipeline)
- Upload dataset and enter mission DSL
- Watch real-time progress in step-by-step bar
- View detailed logs in real-time

### 5. View Real-time Events
- Events list shows all system activity
- Filter by workflow ID for specific workflows
- Timestamps and event types clearly marked
- JSON details for debugging

## 🎨 Customization

### Add New Agents
1. **Backend**: Add to `AGENT_CONFIG` in `enhanced_dashboard.py`
2. **Frontend**: Add to `agentConfig` in `EnhancedDashboard.js`
3. **Restart**: Both backend and frontend services

```python
# Backend
"new_agent": {
    "url": "http://localhost:8006",
    "health_endpoint": "/health",
    "metrics_endpoint": "/metrics",
    "poll_interval": 30000,
    "actions": ["custom_action1", "custom_action2"],
    "description": "Custom Agent Description",
    "color": "#ff6b6b"
}
```

### Custom Metrics
```python
# In agent's metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return {
        "custom_metric_1": value1,
        "custom_metric_2": value2,
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Custom Workflows
```python
# Add to workflow execution logic
def get_step_endpoint(step: str, workflow_type: str) -> str:
    endpoints = {
        "new_agent": {
            "custom_workflow": "execute",
            "full_pipeline": "execute"
        }
    }
    return endpoints.get(step, {}).get(workflow_type, "execute")
```

## 🔧 Troubleshooting

### Dashboard Connection Issues
```bash
# Check backend is running
curl http://localhost:8000/health

# Check WebSocket connection
# Open browser dev tools → Network → WS
# Should see connection to ws://localhost:8000/ws/dashboard
```

### Agent Health Issues
```bash
# Test individual agent health
curl http://localhost:8001/health  # EDA Agent
curl http://localhost:8005/health  # Refinery Agent
curl http://localhost:8002/health  # ML Agent

# Check agent logs for errors
tail -f agent_logs.log
```

### Authentication Issues
```bash
# Verify JWT secret is set
echo $JWT_SECRET_KEY

# Check MongoDB connection
mongo --eval "db.runCommand('ping')"
```

### Database Issues
```bash
# Check MongoDB status
sudo systemctl status mongod

# Check collections
mongo deepline_dashboard --eval "db.workflows.find().pretty()"
```

## ⚡ Performance Optimization

### MongoDB Indexing
```javascript
// Create indexes for better performance
db.workflows.createIndex({"workflow_id": 1})
db.workflows.createIndex({"status": 1})
db.workflows.createIndex({"created_at": -1})
```

### Connection Pooling
```python
# In enhanced_dashboard.py
client = motor.motor_asyncio.AsyncIOMotorClient(
    MONGODB_URL,
    maxPoolSize=10,
    minPoolSize=5
)
```

### Redis Caching (Optional)
```python
import redis.asyncio as redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Cache agent health for 30 seconds
await redis_client.setex(f"agent_health:{agent_name}", 30, health_data)
```

### React Optimization
```javascript
// Use React.memo for expensive components
const AgentCard = React.memo(({ agent, health }) => {
  // Component logic
});

// Use useCallback for event handlers
const handleAgentAction = useCallback((agent, action) => {
  // Action logic
}, []);
```

## 🔄 Updates and Maintenance

### Regular Security Updates
```bash
# Update dependencies
pip install --upgrade fastapi uvicorn motor httpx
npm update

# Rotate JWT secrets monthly
export JWT_SECRET_KEY="new-secret-key-$(date +%s)"
```

### Database Maintenance
```bash
# Backup workflows
mongodump --db deepline_dashboard --collection workflows

# Clean old workflows (older than 30 days)
mongo deepline_dashboard --eval "
  db.workflows.deleteMany({
    created_at: { \$lt: new Date(Date.now() - 30*24*60*60*1000) }
  })
"
```

### Monitoring and Alerting
```bash
# Set up health check monitoring
curl -f http://localhost:8000/health || echo "Dashboard down!"

# Monitor disk space
df -h | grep -E "(/$|/data)"

# Monitor memory usage
free -h
```

## 📚 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [JWT.io](https://jwt.io/)

### Security Best Practices
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [GDPR Compliance](https://gdpr.eu/)

### Monitoring Tools
- [Prometheus](https://prometheus.io/) - Metrics collection
- [Grafana](https://grafana.com/) - Visualization
- [ELK Stack](https://www.elastic.co/what-is/elk-stack) - Logging

## 🚀 Next Steps

### Phase 1: Security & Authentication ✅
- [x] JWT authentication implementation
- [x] Role-based access control
- [x] Secure credential management
- [x] PII detection and anonymization

### Phase 2: Dashboard Integration ✅
- [x] Real-time agent monitoring
- [x] Workflow execution interface
- [x] Step-by-step progress tracking
- [x] Real-time logs and events

### Phase 3: Advanced Features (Future)
- [ ] Advanced analytics and reporting
- [ ] Custom dashboard widgets
- [ ] Workflow templates and scheduling
- [ ] Integration with external monitoring tools
- [ ] Advanced security features (2FA, SSO)
- [ ] Performance optimization and scaling

### Phase 4: Production Deployment
- [ ] Docker containerization
- [ ] Kubernetes orchestration
- [ ] CI/CD pipeline setup
- [ ] Production monitoring and alerting
- [ ] Disaster recovery procedures

---

**🎉 Congratulations!** You now have a fully functional, secure, and feature-rich MLOps dashboard that provides comprehensive monitoring and control over your Deepline agents and workflows. 