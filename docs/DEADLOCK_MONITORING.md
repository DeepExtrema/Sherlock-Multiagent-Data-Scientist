# Deadlock Monitoring & Graceful Cancellation
## Production Workflow Protection System

**Version**: 2.1.0  
**Status**: ✅ Production Ready  
**Last Updated**: January 2024

---

## 📋 **Table of Contents**

- [Overview](#overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Monitoring & Alerting](#monitoring--alerting)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## 🎯 **Overview**

The Deadlock Monitoring system provides automatic detection and recovery for stuck workflows in the Deepline orchestrator. It prevents resource waste and ensures system reliability by identifying workflows where all tasks are stuck in pending states.

### **Key Features**

- **🔍 Automatic Detection**: MongoDB aggregation pipelines scan for stuck workflows
- **⚡ Fast Recovery**: Configurable thresholds (15min default) for rapid response
- **🔔 Smart Alerting**: Slack/PagerDuty integration with rich context
- **🛡️ Graceful Cancellation**: Multi-endpoint API for workflow management
- **🚧 Worker Protection**: Redis signals prevent wasted task execution
- **📊 Production Monitoring**: Health endpoints and comprehensive statistics

### **When Deadlocks Occur**

Common scenarios that trigger deadlock detection:
- **Dependency Cycles**: Task A depends on B, B depends on A
- **Resource Starvation**: All workers busy, new tasks can't start
- **External Service Failures**: Tasks waiting indefinitely for responses
- **Agent Failures**: Worker agents become unresponsive
- **Configuration Errors**: Invalid task configurations preventing execution

---

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Deadlock       │───▶│    MongoDB      │◀───│   Workflow      │
│  Monitor        │    │   Aggregation   │    │   Manager       │
│  (Background)   │    │   Scanning      │    │   (State)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                       │
         v                        v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Alert System   │    │  Cancellation   │    │   Redis Cache   │
│  (Webhooks)     │    │  API Router     │    │  (Signals)      │
│  - Slack        │    │  - Cancel       │    │  - Cancelled    │
│  - PagerDuty    │    │  - Status       │    │    Runs         │
│  - Custom       │    │  - List         │    │  - Rate Limits  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                       │
         └────────────────────────┼───────────────────────┘
                                  v
                       ┌─────────────────┐
                       │  Worker Pools   │
                       │  (Protection)   │
                       │  - Pre-exec     │
                       │    Checks       │
                       │  - Task Abort   │
                       └─────────────────┘
```

### **Component Overview**

1. **DeadlockMonitor** - Background service scanning MongoDB
2. **CancelRouter** - FastAPI endpoints for workflow cancellation
3. **WorkflowManager** - Enhanced with cancellation functions
4. **WorkerPool** - Protected with pre-execution checks
5. **Redis Cache** - Cancellation signals and rate limiting

---

## ⚙️ **Configuration**

### **Config File (`config.yaml`)**

```yaml
orchestrator:
  deadlock:
    check_interval_s: 60          # How often to scan (seconds)
    pending_stale_s: 900          # Task staleness threshold (15 min)
    workflow_stale_s: 3600        # Workflow timeout (1 hour)
    cancel_on_deadlock: true      # Auto-cancel detected deadlocks
    alert_webhook: ""             # Slack/PagerDuty webhook URL
    max_dependency_depth: 50      # Prevent infinite dependency chains
```

### **Configuration Options**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `check_interval_s` | 60 | Deadlock scan frequency |
| `pending_stale_s` | 900 | Task considered stuck after this time |
| `workflow_stale_s` | 3600 | Workflow timeout threshold |
| `cancel_on_deadlock` | true | Automatically cancel deadlocked workflows |
| `alert_webhook` | "" | HTTP webhook for alerts (optional) |
| `max_dependency_depth` | 50 | Maximum task dependency chain length |

### **Environment Variables**

```bash
# Override config values
DEEPLINE_DEADLOCK_CHECK_INTERVAL=30
DEEPLINE_DEADLOCK_PENDING_STALE=600
DEEPLINE_DEADLOCK_WEBHOOK="https://hooks.slack.com/services/..."
```

---

## 🔌 **API Reference**

### **Cancel a Running Workflow**

```http
PUT /runs/{run_id}/cancel
Content-Type: application/json

{
  "reason": "user requested cancellation",
  "force": false
}
```

**Response:**
```json
{
  "run_id": "workflow_abc123",
  "status": "CANCELLING",
  "message": "Workflow cancellation initiated successfully",
  "cancelled_at": "2024-01-15T10:30:00Z",
  "cancelled_tasks": 5,
  "reason": "user requested cancellation"
}
```

### **Check Cancellation Status**

```http
GET /runs/{run_id}/cancel
```

**Response:**
```json
{
  "run_id": "workflow_abc123",
  "is_cancelled": true,
  "status": "CANCELLED",
  "cancellation_reason": "auto-cancel: deadlock detected",
  "cancelled_at": "2024-01-15T10:30:00Z",
  "cancelled_by": "system"
}
```

### **List Cancelled Workflows**

```http
GET /runs/cancelled?limit=50&offset=0&client_id=analytics_team
```

**Response:**
```json
[
  {
    "run_id": "workflow_abc123",
    "workflow_name": "Data Analysis Pipeline",
    "status": "CANCELLED",
    "cancelled_at": "2024-01-15T10:30:00Z",
    "cancellation_reason": "auto-cancel: deadlock detected",
    "cancelled_by": "system",
    "task_count": 8,
    "client_id": "analytics_team"
  }
]
```

### **Force Complete Cancellation (Admin)**

```http
DELETE /runs/{run_id}/cancel
```

**Response:** `204 No Content`

---

## 📊 **Monitoring & Alerting**

### **Health Endpoint**

```http
GET /health
```

**Response includes deadlock monitor status:**
```json
{
  "deadlock_monitor": {
    "running": true,
    "check_interval_s": 60,
    "pending_stale_s": 900,
    "cancel_on_deadlock": true,
    "alert_webhook_configured": true,
    "last_scan": "2024-01-15T10:29:30Z"
  }
}
```

### **Statistics Endpoint**

```http
GET /stats
```

**Response includes deadlock statistics:**
```json
{
  "deadlock_monitor": {
    "deadlocks_detected": 12,
    "workflows_cancelled": 10,
    "alerts_sent": 12,
    "avg_detection_time_s": 45.3,
    "scan_errors": 0
  }
}
```

### **Slack Alert Format**

```json
{
  "text": "🔴 Deadlock Alert: Workflow workflow_abc123",
  "attachments": [
    {
      "color": "danger",
      "fields": [
        {"title": "Workflow ID", "value": "workflow_abc123", "short": true},
        {"title": "Workflow Name", "value": "Data Analysis", "short": true},
        {"title": "Stuck Tasks", "value": "5", "short": true},
        {"title": "Last Update", "value": "20 minutes ago", "short": true},
        {"title": "Auto-Cancel", "value": "true", "short": true}
      ]
    }
  ]
}
```

### **Manual Deadlock Scan**

For troubleshooting, you can trigger manual scans:

```python
from orchestrator.deadlock_monitor import DeadlockMonitor

# Get potentially deadlocked workflows without taking action
results = await monitor.manual_scan()
print(f"Found {len(results)} potentially deadlocked workflows")
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. False Positive Deadlock Detection**

**Symptoms:** Workflows cancelled but tasks were actually progressing
```bash
# Check task update timestamps
db.tasks.find({"run_id": "workflow_123"}).sort({"updated_at": -1})

# Adjust staleness threshold
deadlock:
  pending_stale_s: 1800  # Increase to 30 minutes
```

#### **2. Deadlock Monitor Not Starting**

**Symptoms:** No deadlock detection despite stuck workflows
```bash
# Check logs for startup errors
grep -i "deadlock" logs/orchestrator.log

# Verify MongoDB connection
curl http://localhost:8001/health | jq .deadlock_monitor
```

#### **3. Webhook Alerts Not Sent**

**Symptoms:** Deadlocks detected but no notifications
```bash
# Test webhook manually
curl -X POST $WEBHOOK_URL \
  -H "Content-Type: application/json" \
  -d '{"text": "Test alert"}'

# Check webhook configuration
grep -i webhook config.yaml
```

#### **4. Worker Tasks Not Cancelling**

**Symptoms:** Tasks continue executing after cancellation
```bash
# Check Redis cancellation signals
redis-cli SMEMBERS cancelled_runs

# Verify worker cancellation checks
grep -i "cancelled" logs/worker.log
```

### **Debug Commands**

```bash
# Check deadlock monitor status
curl http://localhost:8001/health | jq .deadlock_monitor

# View recent cancellations
curl "http://localhost:8001/runs/cancelled?limit=5"

# Manual deadlock detection (via Python)
python -c "
from orchestrator.deadlock_monitor import DeadlockMonitor
from config import load_config
import asyncio

async def scan():
    config = load_config()
    # Manual scan implementation
    print('Manual scan results...')

asyncio.run(scan())
"
```

### **Performance Tuning**

```yaml
# For high-throughput systems
deadlock:
  check_interval_s: 30        # More frequent checks
  pending_stale_s: 600        # Shorter timeout
  max_dependency_depth: 100   # Deeper dependency chains

# For long-running workflows
deadlock:
  check_interval_s: 300       # Less frequent checks
  pending_stale_s: 3600       # Longer timeout
  workflow_stale_s: 7200      # 2-hour workflow timeout
```

---

## 🎯 **Best Practices**

### **1. Configuration Tuning**

- **Development**: Use shorter thresholds (5-10 minutes) for faster feedback
- **Production**: Use conservative thresholds (15-30 minutes) to avoid false positives
- **Long-running workflows**: Increase `workflow_stale_s` appropriately

### **2. Monitoring Setup**

```yaml
# Recommended production config
deadlock:
  check_interval_s: 60
  pending_stale_s: 900
  workflow_stale_s: 3600
  cancel_on_deadlock: true
  alert_webhook: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

### **3. Webhook Integration**

```python
# Slack webhook setup
webhook_url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"

# PagerDuty integration
webhook_url = "https://events.pagerduty.com/v2/enqueue"
# Include routing_key and event_action in payload
```

### **4. Operational Procedures**

1. **Monitor deadlock statistics** - Track trends and tune thresholds
2. **Review cancellation reasons** - Identify systematic issues
3. **Set up alerting dashboards** - Visualize deadlock patterns
4. **Test cancellation workflows** - Verify system behavior under load
5. **Document escalation procedures** - Clear steps for manual intervention

### **5. Development Guidelines**

```python
# Always check for cancellation in long-running tasks
async def execute_task(task_meta):
    run_id = task_meta["run_id"]
    
    # Check cancellation before expensive operations
    if await is_workflow_cancelled(run_id):
        await emit_event("TASK_CANCELLED", task_meta)
        return
    
    # Perform task work...
    await long_running_operation()
    
    # Check again after major operations
    if await is_workflow_cancelled(run_id):
        await emit_event("TASK_CANCELLED", task_meta)
        return
```

---

## 📈 **Metrics & Observability**

### **Key Metrics**

- **Deadlock Detection Rate**: Workflows/hour identified as deadlocked
- **False Positive Rate**: Cancelled workflows that weren't actually stuck
- **Mean Time to Detection**: How quickly deadlocks are identified
- **Cancellation Success Rate**: Percentage of successful cancellations
- **Alert Response Time**: Time from detection to notification

### **Grafana Dashboard Queries**

```promql
# Deadlock detection rate
rate(deadlock_detected_total[5m])

# Worker cancellation events
rate(task_cancelled_total[5m])

# Average detection time
avg(deadlock_detection_duration_seconds)
```

### **Logging**

The system provides structured logging for observability:

```
INFO  Deadlock detected for workflow workflow_123 with 5 stuck tasks
WARN  Workflow workflow_456 cancelled: auto-cancel: deadlock detected
ERROR Failed to send deadlock alert: webhook timeout
DEBUG No deadlocks detected in current scan
```

---

This comprehensive deadlock monitoring system ensures your Deepline workflows remain healthy and responsive, automatically recovering from stuck states while providing detailed observability into system behavior. 