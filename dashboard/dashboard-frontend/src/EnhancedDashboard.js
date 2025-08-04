import React, { useEffect, useState, useCallback } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area
} from 'recharts';
import { 
  Play, Pause, Activity, Database, Settings, Cpu, TrendingUp, Package, 
  Brain, AlertCircle, CheckCircle, XCircle, ChevronRight, Users, Zap, 
  BarChart3, Eye, EyeOff, RefreshCw, Target, Shield, Clock, AlertTriangle
} from 'lucide-react';
import './EnhancedDashboard.css';

const EnhancedDashboard = () => {
  const [agents, setAgents] = useState({});
  const [agentHealth, setAgentHealth] = useState({});
  const [workflows, setWorkflows] = useState([]);
  const [events, setEvents] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [agentMetrics, setAgentMetrics] = useState({});
  const [workflowActive, setWorkflowActive] = useState(false);
  const [activeWorkflow, setActiveWorkflow] = useState(null);
  const [showMetrics, setShowMetrics] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showWorkflowModal, setShowWorkflowModal] = useState(false);
  const [workflowForm, setWorkflowForm] = useState({
    workflow_type: 'full_pipeline',
    dataset: null,
    mission_dsl: '',
    options: {}
  });
  const [stepStates, setStepStates] = useState({});
  const [activeWorkflowId, setActiveWorkflowId] = useState(null);

  // Agent configurations
  const agentConfig = {
    eda_agent: {
      name: 'EDA Agent',
      icon: <BarChart3 className="icon" />,
      color: '#48bb78',
      description: 'Exploratory Data Analysis',
      category: 'Data Analysis'
    },
    refinery_agent: {
      name: 'Refinery Agent',
      icon: <Settings className="icon" />,
      color: '#4299e1',
      description: 'Data Quality & Feature Engineering',
      category: 'Data Processing'
    },
    ml_agent: {
      name: 'ML Agent',
      icon: <Cpu className="icon" />,
      color: '#9f7aea',
      description: 'Machine Learning & Model Training',
      category: 'Machine Learning'
    },
    master_orchestrator: {
      name: 'Master Orchestrator',
      icon: <Brain className="icon" />,
      color: '#667eea',
      description: 'Workflow Orchestration & Management',
      category: 'Orchestration'
    }
  };

  // WebSocket connection
  const [ws, setWs] = useState(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const websocket = new WebSocket('ws://localhost:8000/ws/dashboard');
    
    websocket.onopen = () => {
      setConnectionStatus('Connected');
      console.log('Enhanced Dashboard WebSocket Connected');
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    websocket.onclose = () => {
      setConnectionStatus('Disconnected');
      console.log('Enhanced Dashboard WebSocket Disconnected');
    };

    websocket.onerror = (error) => {
      setConnectionStatus('Error');
      console.error('Enhanced Dashboard WebSocket Error:', error);
    };

    setWs(websocket);

    return () => {
      websocket.close();
    };
  }, []);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((data) => {
    switch (data.type) {
      case 'initial_status':
        setAgentHealth(data.agents);
        break;
      case 'agent_health_update':
        setAgentHealth(prev => ({
          ...prev,
          [data.agent_name]: data.health_status
        }));
        break;
      case 'workflow_started':
        setWorkflowActive(true);
        setActiveWorkflowId(data.workflow_id);
        // Reset step states for this workflow
        const newStepStates = {};
        data.agents.forEach(step => {
          newStepStates[step] = "pending";
        });
        setStepStates(newStepStates);
        setEvents(prev => [...prev, data]);
        break;
      case 'step.update':
        setStepStates(prev => ({
          ...prev,
          [data.step]: data.status
        }));
        setEvents(prev => [...prev, data]);
        break;
      case 'workflow_status_updated':
        setWorkflowActive(data.status === "running");
        if (data.status === "completed" || data.status === "failed") {
          setActiveWorkflowId(null);
        }
        setEvents(prev => [...prev, data]);
        break;
      case 'agent_action_executed':
        setEvents(prev => [...prev, data]);
        break;
      default:
        setEvents(prev => [...prev, data]);
    }
  }, []);

  // Fetch initial data
  useEffect(() => {
    fetchAgentHealth();
    fetchWorkflows();
  }, []);

  // Auto-refresh agent health
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchAgentHealth();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [autoRefresh]);

  // API functions
  const fetchAgentHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/agents/health');
      const data = await response.json();
      setAgentHealth(data.agents);
    } catch (error) {
      console.error('Error fetching agent health:', error);
    }
  };

  const fetchWorkflows = async () => {
    try {
      const response = await fetch('http://localhost:8000/workflows');
      const data = await response.json();
      setWorkflows(data.workflows);
    } catch (error) {
      console.error('Error fetching workflows:', error);
    }
  };

  const fetchAgentMetrics = async (agentName) => {
    try {
      const response = await fetch(`http://localhost:8000/agents/${agentName}/metrics`);
      const data = await response.json();
      setAgentMetrics(prev => ({
        ...prev,
        [agentName]: data.metrics
      }));
    } catch (error) {
      console.error(`Error fetching metrics for ${agentName}:`, error);
    }
  };

  const executeAgentAction = async (agentName, action, parameters = {}) => {
    try {
      const response = await fetch(`http://localhost:8000/agents/${agentName}/actions/${action}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action, parameters })
      });
      
      const result = await response.json();
      console.log(`Action ${action} executed on ${agentName}:`, result);
      return result;
    } catch (error) {
      console.error(`Error executing action ${action} on ${agentName}:`, error);
      throw error;
    }
  };

  const startWorkflow = async () => {
    try {
      const workflowData = {
        workflow_type: workflowForm.workflow_type,
        agents: getWorkflowAgents(workflowForm.workflow_type),
        parameters: {
          dataset: workflowForm.dataset,
          mission_dsl: workflowForm.mission_dsl,
          options: workflowForm.options
        }
      };

      const response = await fetch('/api/workflows/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(workflowData),
      });

      if (response.ok) {
        const result = await response.json();
        setShowWorkflowModal(false);
        setWorkflowForm({
          workflow_type: 'full_pipeline',
          dataset: null,
          mission_dsl: '',
          options: {}
        });
        // Refresh workflows list
        fetchWorkflows();
      } else {
        console.error('Failed to start workflow:', await response.text());
      }
    } catch (error) {
      console.error('Error starting workflow:', error);
    }
  };

  const getWorkflowAgents = (workflowType) => {
    switch (workflowType) {
      case 'data_analysis':
        return ['eda_agent'];
      case 'ml_pipeline':
        return ['refinery_agent', 'ml_agent'];
      case 'full_pipeline':
        return ['eda_agent', 'refinery_agent', 'ml_agent'];
      default:
        return ['eda_agent', 'refinery_agent', 'ml_agent'];
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setWorkflowForm(prev => ({
        ...prev,
        dataset: file.name
      }));
    }
  };

  // Helper functions
  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="icon status-icon healthy" />;
      case 'unhealthy':
        return <XCircle className="icon status-icon unhealthy" />;
      case 'error':
        return <AlertTriangle className="icon status-icon error" />;
      default:
        return <AlertCircle className="icon status-icon unknown" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return '#10b981';
      case 'unhealthy':
        return '#ef4444';
      case 'error':
        return '#f59e0b';
      default:
        return '#6b7280';
    }
  };

  const getResponseTimeColor = (responseTime) => {
    if (responseTime < 1) return '#10b981';
    if (responseTime < 3) return '#f59e0b';
    return '#ef4444';
  };

  // Calculate statistics
  const healthyAgents = Object.values(agentHealth).filter(h => h.status === 'healthy').length;
  const totalAgents = Object.keys(agentHealth).length;
  const healthPercentage = totalAgents > 0 ? Math.round((healthyAgents / totalAgents) * 100) : 0;

  const avgResponseTime = Object.values(agentHealth).reduce((sum, h) => sum + (h.response_time || 0), 0) / totalAgents || 0;

  return (
    <div className="enhanced-dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-content">
          <div className="header-left">
            <h1>
              <Zap className="icon header-icon" />
              Enhanced Deepline Dashboard
            </h1>
            <div className="header-status">
              <span className={`status ${connectionStatus === 'Connected' ? 'connected' : 'disconnected'}`}>
                {connectionStatus}
              </span>
              {workflowActive && activeWorkflow && (
                <span className="active-workflow">
                  Active Workflow: {activeWorkflow}
                </span>
              )}
            </div>
          </div>
          
          <div className="header-controls">
            <button 
              onClick={fetchAgentHealth}
              className="control-button"
              title="Refresh Agent Health"
            >
              <RefreshCw className="icon" />
            </button>
            <button 
              onClick={() => setShowMetrics(!showMetrics)}
              className="control-button"
              title={showMetrics ? "Hide Metrics" : "Show Metrics"}
            >
              {showMetrics ? <EyeOff className="icon" /> : <Eye className="icon" />}
            </button>
            <button 
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`control-button ${autoRefresh ? 'active' : ''}`}
              title={autoRefresh ? "Disable Auto-refresh" : "Enable Auto-refresh"}
            >
              <Clock className="icon" />
            </button>
          </div>
        </div>
      </header>

      {/* Statistics Overview */}
      <div className="stats-overview">
        <div className="stat-card">
          <div className="stat-content">
            <div className="stat-info">
              <p className="stat-label">Agent Health</p>
              <p className="stat-value">{healthPercentage}%</p>
              <p className="stat-detail">{healthyAgents}/{totalAgents} Healthy</p>
            </div>
            <div className="stat-icon">
              <Shield className="icon" style={{ color: getStatusColor(healthPercentage >= 75 ? 'healthy' : 'unhealthy') }} />
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-content">
            <div className="stat-info">
              <p className="stat-label">Avg Response Time</p>
              <p className="stat-value">{avgResponseTime.toFixed(2)}s</p>
              <p className="stat-detail">Agent Health Checks</p>
            </div>
            <div className="stat-icon">
              <Target className="icon" style={{ color: getResponseTimeColor(avgResponseTime) }} />
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-content">
            <div className="stat-info">
              <p className="stat-label">Active Workflows</p>
              <p className="stat-value">{workflows.filter(w => w.status === 'running').length}</p>
              <p className="stat-detail">Currently Running</p>
            </div>
            <div className="stat-icon">
              <Activity className="icon" style={{ color: workflowActive ? '#10b981' : '#6b7280' }} />
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-content">
            <div className="stat-info">
              <p className="stat-label">Total Events</p>
              <p className="stat-value">{events.length}</p>
              <p className="stat-detail">Real-time Updates</p>
            </div>
            <div className="stat-icon">
              <TrendingUp className="icon" style={{ color: '#667eea' }} />
            </div>
          </div>
        </div>
      </div>

      <div className="dashboard-content">
        {/* Agent Status Grid */}
        <div className="agent-grid">
          <h2 className="section-title">
            <Activity className="icon" />
            Agent Status
          </h2>
          
          <div className="agents-container">
            {Object.entries(agentConfig).map(([agentId, config]) => {
              const health = agentHealth[agentId];
              const isHealthy = health?.status === 'healthy';
              
              return (
                <div 
                  key={agentId}
                  className={`agent-card ${isHealthy ? 'healthy' : 'unhealthy'} ${selectedAgent === agentId ? 'selected' : ''}`}
                  onClick={() => setSelectedAgent(selectedAgent === agentId ? null : agentId)}
                >
                  <div className="agent-card-header">
                    <div className="agent-info">
                      <div className="agent-icon" style={{ backgroundColor: `${config.color}20` }}>
                        {React.cloneElement(config.icon, { color: config.color })}
                      </div>
                      <div className="agent-details">
                        <h3 className="agent-name">{config.name}</h3>
                        <p className="agent-description">{config.description}</p>
                        <span className="agent-category">{config.category}</span>
                      </div>
                    </div>
                    <div className="agent-status">
                      {health ? getStatusIcon(health.status) : <AlertCircle className="icon status-icon unknown" />}
                      <span className="status-text" style={{ color: getStatusColor(health?.status || 'unknown') }}>
                        {health?.status || 'unknown'}
                      </span>
                    </div>
                  </div>

                  {health && (
                    <div className="agent-health-details">
                      <div className="health-metric">
                        <span className="metric-label">Response Time:</span>
                        <span className="metric-value" style={{ color: getResponseTimeColor(health.response_time) }}>
                          {health.response_time.toFixed(3)}s
                        </span>
                      </div>
                      <div className="health-metric">
                        <span className="metric-label">Last Check:</span>
                        <span className="metric-value">
                          {new Date(health.last_check).toLocaleTimeString()}
                        </span>
                      </div>
                      {health.error && (
                        <div className="health-error">
                          <span className="error-label">Error:</span>
                          <span className="error-message">{health.error}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {selectedAgent === agentId && (
                    <div className="agent-actions">
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          fetchAgentMetrics(agentId);
                        }}
                        className="action-button"
                      >
                        <BarChart3 className="icon" />
                        View Metrics
                      </button>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          executeAgentAction(agentId, 'health');
                        }}
                        className="action-button"
                      >
                        <RefreshCw className="icon" />
                        Health Check
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Metrics and Events */}
        <div className="metrics-events-section">
          {/* Agent Metrics */}
          {showMetrics && selectedAgent && agentMetrics[selectedAgent] && (
            <div className="metrics-card">
              <h3 className="card-title">
                <BarChart3 className="icon" />
                {agentConfig[selectedAgent]?.name} Metrics
              </h3>
              <div className="metrics-content">
                <pre className="metrics-json">
                  {JSON.stringify(agentMetrics[selectedAgent], null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Real-time Events */}
          <div className="events-card">
            <h3 className="card-title">
              <Activity className="icon" />
              Real-time Events
            </h3>
            <div className="events-list">
              {events.length === 0 ? (
                <p className="no-events">No events received yet</p>
              ) : (
                events.slice(-10).reverse().map((event, index) => (
                  <div key={index} className="event-item">
                    <div className="event-header">
                      <span className="event-type">{event.type}</span>
                      <span className="event-time">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="event-content">
                      <pre className="event-data">
                        {JSON.stringify(event, null, 2)}
                      </pre>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Workflow Management Section */}
        <div className="workflow-section">
          <h2>Workflow Management</h2>
          <div className="workflow-controls">
            <button 
              className="workflow-btn primary"
              onClick={() => setShowWorkflowModal(true)}
              disabled={workflowActive}
            >
              Start Full Pipeline
            </button>
            <button 
              className="workflow-btn secondary"
              onClick={() => {
                setWorkflowForm(prev => ({ ...prev, workflow_type: 'data_analysis' }));
                setShowWorkflowModal(true);
              }}
              disabled={workflowActive}
            >
              Data Analysis Only
            </button>
            <button 
              className="workflow-btn secondary"
              onClick={() => {
                setWorkflowForm(prev => ({ ...prev, workflow_type: 'ml_pipeline' }));
                setShowWorkflowModal(true);
              }}
              disabled={workflowActive}
            >
              ML Pipeline Only
            </button>
          </div>

          {/* Step-by-Step Progress Bar */}
          {workflowActive && activeWorkflowId && (
            <div className="pipeline-progress">
              <h3>Pipeline Progress</h3>
              <ol className="pipeline-steps">
                {Object.entries(stepStates).map(([step, status]) => (
                  <li key={step} className={`step ${status}`}>
                    <span className="step-icon">
                      {status === 'completed' && '✓'}
                      {status === 'in_progress' && '⟳'}
                      {status === 'failed' && '✗'}
                      {status === 'pending' && '○'}
                    </span>
                    <span className="step-name">{step.replace('_agent', '').toUpperCase()}</span>
                    <span className="step-badge">{status}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {/* Real-time Logs */}
          {workflowActive && activeWorkflowId && (
            <div className="workflow-logs">
              <h3>Real-time Logs</h3>
              <div className="logs-container">
                {events
                  .filter(ev => ev.workflow_id === activeWorkflowId)
                  .slice(-20)
                  .map((ev, index) => (
                    <pre key={index} className="log-entry">
                      [{new Date(ev.timestamp).toLocaleTimeString()}] {ev.type} → {JSON.stringify(ev, null, 2)}
                    </pre>
                  ))}
              </div>
            </div>
          )}

          {/* Recent Workflows */}
          <div className="recent-workflows">
            <h3>Recent Workflows</h3>
            <div className="workflows-list">
              {workflows.slice(0, 5).map(workflow => (
                <div key={workflow.workflow_id} className="workflow-item">
                  <div className="workflow-info">
                    <span className="workflow-type">{workflow.workflow_type}</span>
                    <span className={`workflow-status ${workflow.status}`}>
                      {workflow.status}
                    </span>
                  </div>
                  <div className="workflow-meta">
                    <span>{new Date(workflow.created_at).toLocaleString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Workflow Modal */}
        {showWorkflowModal && (
          <div className="modal-overlay">
            <div className="modal-content">
              <div className="modal-header">
                <h2>Run New Workflow</h2>
                <button 
                  className="modal-close"
                  onClick={() => setShowWorkflowModal(false)}
                >
                  ×
                </button>
              </div>
              <div className="modal-body">
                <div className="form-group">
                  <label>Workflow Type:</label>
                  <select
                    value={workflowForm.workflow_type}
                    onChange={(e) => setWorkflowForm(prev => ({
                      ...prev,
                      workflow_type: e.target.value
                    }))}
                  >
                    <option value="full_pipeline">Full Pipeline</option>
                    <option value="data_analysis">Data Analysis Only</option>
                    <option value="ml_pipeline">ML Pipeline Only</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Dataset:</label>
                  <input
                    type="file"
                    onChange={handleFileChange}
                    accept=".csv,.json,.parquet"
                  />
                </div>

                <div className="form-group">
                  <label>Mission DSL:</label>
                  <textarea
                    value={workflowForm.mission_dsl}
                    onChange={(e) => setWorkflowForm(prev => ({
                      ...prev,
                      mission_dsl: e.target.value
                    }))}
                    placeholder="Enter your mission description..."
                    rows={4}
                  />
                </div>

                <div className="form-group">
                  <details>
                    <summary>Advanced Options</summary>
                    <textarea
                      value={JSON.stringify(workflowForm.options, null, 2)}
                      onChange={(e) => {
                        try {
                          const options = JSON.parse(e.target.value);
                          setWorkflowForm(prev => ({
                            ...prev,
                            options
                          }));
                        } catch (error) {
                          // Invalid JSON, ignore
                        }
                      }}
                      placeholder='{"option": "value"}'
                      rows={3}
                    />
                  </details>
                </div>
              </div>
              <div className="modal-footer">
                <button 
                  className="btn-secondary"
                  onClick={() => setShowWorkflowModal(false)}
                >
                  Cancel
                </button>
                <button 
                  className="btn-primary"
                  onClick={startWorkflow}
                >
                  Start Workflow
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedDashboard; 