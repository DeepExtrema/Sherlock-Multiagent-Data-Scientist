import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Play, Upload, Activity, Database, Settings, Terminal, TrendingUp, Cpu, HardDrive, Zap } from 'lucide-react'
import './styles.css'

const API = {
  orchestratorBase: 'http://localhost:8000',
  edaBase: 'http://localhost:8001',
  refineryBase: 'http://localhost:8005',
  mlBase: 'http://localhost:8002',
}

function useInterval(callback, delayMs) {
  const savedCb = useRef(callback)
  useEffect(() => { savedCb.current = callback }, [callback])
  useEffect(() => {
    if (delayMs == null) return
    const id = setInterval(() => savedCb.current(), delayMs)
    return () => clearInterval(id)
  }, [delayMs])
}

async function fetchJson(url, options) {
  const res = await fetch(url, options)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

function Header({ orchestrator, eda, refinery, ml }) {
  const [activeTab, setActiveTab] = React.useState('overview')
  
  const agents = [
    { id: 'orchestrator', label: 'Orchestrator', icon: Cpu, healthy: !!orchestrator?.healthy },
    { id: 'eda', label: 'EDA', icon: TrendingUp, healthy: !!eda?.healthy },
    { id: 'refinery', label: 'Refinery', icon: Settings, healthy: !!refinery?.healthy },
    { id: 'ml', label: 'ML', icon: Zap, healthy: !!ml?.healthy }
  ]

  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-left">
          <h1 className="brand">DEEPLINE</h1>
          <nav className="nav">
            {agents.map(agent => (
              <button
                key={agent.id}
                onClick={() => setActiveTab(agent.id)}
                className={`nav-item ${activeTab === agent.id ? 'active' : ''}`}
              >
                <agent.icon className="nav-icon" />
                <span>{agent.label}</span>
              </button>
            ))}
          </nav>
        </div>
        <div className="header-right">
          <div className="system-status">
            <div className="status-dot healthy" />
            <span>System Healthy</span>
          </div>
        </div>
      </div>
    </header>
  )
}

function ConsolePanel({ onSubmitPrompt, lastResult, busy }) {
  const [nlPrompt, setNlPrompt] = useState('explanation and then train 3 separate models, random forest, linear regression and decision tree on this and visualize the results')
  const [consoleOutput, setConsoleOutput] = useState([
    { id: 1, type: 'success', message: 'Workflow "ML Run - 4:54:25 PM" started successfully' },
    { id: 2, type: 'info', message: 'Model training initialized...' }
  ])

  const handleSubmit = () => {
    if (nlPrompt.trim()) {
      const newOutput = { id: Date.now(), type: 'command', message: `> ${nlPrompt}` }
      setConsoleOutput(prev => [...prev, newOutput])
      onSubmitPrompt(nlPrompt)
      setNlPrompt('')
    }
  }

  return (
    <div className="console-container">
      <div className="console-header">
        <Terminal className="console-icon" />
        <h2>Console</h2>
      </div>
      <div className="console-output">
        {consoleOutput.map(output => (
          <div key={output.id} className={`console-line ${output.type}`}>
            {output.message}
          </div>
        ))}
        {lastResult && (
          <div className="console-line success">
            {typeof lastResult === 'string' ? lastResult : JSON.stringify(lastResult, null, 2)}
          </div>
        )}
      </div>
      <div className="console-input">
        <input
          type="text"
          value={nlPrompt}
          onChange={(e) => setNlPrompt(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
          className="console-prompt"
          placeholder="Ask Deepline to analyze your data..."
          disabled={busy}
        />
        <button
          onClick={handleSubmit}
          disabled={busy || !nlPrompt.trim()}
          className="console-submit"
        >
          <Play className="console-submit-icon" />
        </button>
      </div>
    </div>
  )
}

function DatasetsPanel({ datasets, onUpload }) {
  const [name, setName] = useState('uploaded')
  const [file, setFile] = useState(null)
  
  return (
    <div className="datasets-container">
      <div className="datasets-header">
        <Database className="datasets-icon" />
        <h2>Datasets</h2>
      </div>
      
      <div className="datasets-upload">
        {!datasets?.datasets?.length ? (
          <div className="upload-area">
            <Upload className="upload-icon" />
            <div className="upload-text">No datasets uploaded</div>
            <input
              type="file"
              id="file-upload"
              onChange={e => setFile(e.target.files?.[0] ?? null)}
              className="upload-input"
            />
            <label htmlFor="file-upload" className="upload-button">
              Upload Dataset
            </label>
          </div>
        ) : (
          <div className="upload-controls">
            <input
              type="file"
              onChange={e => setFile(e.target.files?.[0] ?? null)}
              className="file-input"
            />
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="dataset name"
              className="name-input"
            />
            <button
              className="upload-btn"
              onClick={() => file && onUpload(file, name)}
              disabled={!file}
            >
              <Upload className="upload-btn-icon" />
              Upload
            </button>
          </div>
        )}
      </div>

      {datasets?.datasets?.length > 0 && (
        <div className="datasets-list">
          <div className="datasets-table">
            <div className="dataset-row dataset-header">
              <div>Name</div>
              <div>Rows×Cols</div>
              <div>Memory</div>
            </div>
            {datasets.datasets.map((d, i) => (
              <div key={i} className="dataset-row">
                <div className="dataset-name">{d.name}</div>
                <div className="dataset-shape">{d.shape?.[0]} × {d.shape?.[1]}</div>
                <div className="dataset-memory">{d.memory_usage}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="resource-usage">
        <h3 className="resource-header">
          <HardDrive className="resource-icon" />
          System Resources
        </h3>
        <div className="resource-metrics">
          <div className="resource-metric">
            <div className="metric-label">CPU Usage</div>
            <div className="metric-value cpu">23%</div>
            <div className="metric-bar">
              <div className="metric-fill cpu-fill" style={{ width: '23%' }} />
            </div>
          </div>
          <div className="resource-metric">
            <div className="metric-label">Memory</div>
            <div className="metric-value memory">67%</div>
            <div className="metric-bar">
              <div className="metric-fill memory-fill" style={{ width: '67%' }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function WorkflowsPanel({ runs, startBasicInfo }) {
  const [starting, setStarting] = useState(false)
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'COMPLETED': return 'status-completed'
      case 'RUNNING': return 'status-running'
      case 'QUEUED': return 'status-queued'
      case 'FAILED': return 'status-failed'
      default: return 'status-default'
    }
  }

  const getProgressBar = (progress, status) => {
    const isRunning = status === 'RUNNING'
    return (
      <div className="progress-container">
        <div className={`progress-bar ${getStatusColor(status)} ${isRunning ? 'animate' : ''}`} 
             style={{ width: `${progress}%` }} />
      </div>
    )
  }

  const onStart = async () => {
    setStarting(true)
    try { await startBasicInfo() } finally { setStarting(false) }
  }

  return (
    <div className="workflows-container">
      <div className="workflows-header">
        <Activity className="workflows-icon" />
        <h2>Workflows</h2>
      </div>
      <div className="workflows-content">
        {(runs?.runs ?? []).slice().reverse().slice(0, 6).map((workflow) => (
          <div key={workflow.run_id} className="workflow-card">
            <div className="workflow-info">
              <div className="workflow-meta">
                <div className="workflow-id">{workflow.run_id.substring(0, 8)}...</div>
                <div className={`workflow-status ${getStatusColor(workflow.status)}`}>
                  {workflow.status}
                </div>
              </div>
              <div className="workflow-name">Run {workflow.run_id.substring(0, 12)}...</div>
            </div>
            <div className="workflow-progress">
              {getProgressBar(Math.round(workflow.progress), workflow.status)}
              <span className="progress-text">{Math.round(workflow.progress)}%</span>
            </div>
          </div>
        ))}
        {!runs?.runs?.length && (
          <div className="no-workflows">
            <Activity className="no-workflows-icon" />
            <span>No workflows running</span>
          </div>
        )}
      </div>
    </div>
  )
}

function ProcessesPanel({ orchestrator, eda, refinery }) {
  const edaOps = useMemo(() => {
    const t = eda?.telemetry || {}
    return Object.keys(t).map(k => ({ op: k, count: t[k]?.count ?? 0, errors: t[k]?.errors ?? 0 }))
  }, [eda])

  return (
    <div className="processes-container">
      <div className="processes-header">
        <Cpu className="processes-icon" />
        <h2>Background Processes</h2>
      </div>
      <div className="processes-content">
        <div className="process-card">
          <div className="process-header">
            <h4>Orchestrator</h4>
            <div className="process-status">
              <div className={`status-dot ${!!orchestrator?.healthy ? 'healthy' : 'unhealthy'}`} />
              <span className={!!orchestrator?.healthy ? 'status-healthy' : 'status-unhealthy'}>
                {!!orchestrator?.healthy ? 'healthy' : 'unhealthy'}
              </span>
            </div>
          </div>
          <div className="process-details">
            <div>Agents: {Object.keys(orchestrator?.agents || {}).length}</div>
            <div className="process-time">
              Started: {orchestrator?.timestamp ? new Date(orchestrator.timestamp).toLocaleTimeString() : 'N/A'}
            </div>
          </div>
        </div>

        <div className="process-card">
          <div className="process-header">
            <h4>Refinery</h4>
            <div className="process-status">
              <div className={`status-dot ${!!refinery?.healthy ? 'healthy' : 'unhealthy'}`} />
              <span className={!!refinery?.healthy ? 'status-healthy' : 'status-unhealthy'}>
                {!!refinery?.healthy ? 'healthy' : 'unhealthy'}
              </span>
            </div>
          </div>
          <div className="process-details">
            <div>FE Module: <span className={refinery?.fe_module_enabled ? 'status-enabled' : 'status-disabled'}>
              {refinery?.fe_module_enabled ? 'enabled' : 'disabled'}
            </span></div>
            <div>Available: <span className={refinery?.fe_module_available ? 'status-enabled' : 'status-disabled'}>
              {String(!!refinery?.fe_module_available)}
            </span></div>
          </div>
        </div>
      </div>

      <div className="telemetry-section">
        <div className="telemetry-header">
          <TrendingUp className="telemetry-icon" />
          <h3>EDA Telemetry</h3>
        </div>
        <div className="telemetry-table">
          <div className="telemetry-row telemetry-header-row">
            <div>Operation</div>
            <div>Count</div>
            <div>Errors</div>
          </div>
          {edaOps.map((o) => (
            <div key={o.op} className="telemetry-row">
              <div>{o.op}</div>
              <div className="telemetry-count">{o.count}</div>
              <div className={`telemetry-errors ${o.errors ? 'has-errors' : 'no-errors'}`}>{o.errors}</div>
            </div>
          ))}
          {!edaOps.length && (
            <div className="telemetry-empty">No telemetry data</div>
          )}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [orcHealth, setOrcHealth] = useState(null)
  const [edaHealth, setEdaHealth] = useState(null)
  const [refineryHealth, setRefineryHealth] = useState(null)
  const [mlHealth, setMlHealth] = useState(null)
  const [datasets, setDatasets] = useState(null)
  const [runs, setRuns] = useState(null)
  const [lastResult, setLastResult] = useState('')
  const [busy, setBusy] = useState(false)

  const poll = async () => {
    try {
      const [orc, eda, refi, ml, ds, rs] = await Promise.all([
        fetchJson(`${API.orchestratorBase}/health`).catch(() => null),
        fetchJson(`${API.edaBase}/health`).catch(() => null),
        fetchJson(`${API.refineryBase}/health`).catch(() => null),
        fetchJson(`${API.mlBase}/health`).catch(() => null),
        fetchJson(`${API.orchestratorBase}/datasets`).catch(() => ({ datasets: [] })),
        fetchJson(`${API.orchestratorBase}/runs`).catch(() => ({ runs: [] })),
      ])
      setOrcHealth(orc && { healthy: true, ...orc })
      setEdaHealth(eda && { healthy: true, ...eda })
      setRefineryHealth(refi && { healthy: true, ...refi })
      setMlHealth(ml && { healthy: true, ...ml })
      setDatasets(ds)
      setRuns(rs)
    } catch (e) { /* ignore */ }
  }

  useEffect(() => { poll() }, [])
  useInterval(poll, 2000)

  async function handleUpload(file, name) {
    const form = new FormData()
    form.append('file', file)
    form.append('name', name)
    await fetch(`${API.orchestratorBase}/datasets/upload`, { method: 'POST', body: form })
    setLastResult(`Uploaded ${name}`)
    await poll()
  }

  async function handleLoadIris() {
    const body = { path: '/app/iris.csv', name: 'iris', file_type: 'csv' }
    const res = await fetchJson(`${API.edaBase}/load_data`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    setLastResult(JSON.stringify(res, null, 2))
    await poll()
  }

  async function handleStartBasicInfo() {
    const body = { run_name: 'EDA Basic Info', tasks: [{ agent: 'eda_agent', action: 'basic_info', args: { name: 'iris' } }] }
    const res = await fetchJson(`${API.orchestratorBase}/workflows/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    setLastResult(JSON.stringify(res, null, 2))
    await poll()
  }
  function planFromPrompt(prompt) {
    const p = (prompt || '').toLowerCase()
    const tasks = []
    if (p.includes('load iris')) {
      tasks.push({ agent: 'eda_agent', action: 'load_data', args: { path: '/app/iris.csv', name: 'iris', file_type: 'csv' } })
    }
    if (p.includes('basic') || p.includes('summary') || p.includes('info')) {
      tasks.push({ agent: 'eda_agent', action: 'basic_info', args: { name: 'iris' } })
    }
    if (p.includes('quality') || p.includes('missing') || p.includes('drift')) {
      tasks.push({ agent: 'refinery_agent', action: 'check_missing_values', args: { dataset: 'iris' } })
    }
    if (tasks.length === 0) {
      tasks.push({ agent: 'eda_agent', action: 'basic_info', args: { name: 'iris' } })
    }
    return tasks
  }
  async function handleSubmitPrompt(prompt) {
    setBusy(true)
    try {
      if ((prompt || '').toLowerCase().includes('load iris')) {
        await handleLoadIris()
      }
      const tasks = planFromPrompt(prompt)
      const body = { run_name: `NL Run – ${new Date().toLocaleTimeString()}`, tasks }
      const res = await fetchJson(`${API.orchestratorBase}/workflows/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      setLastResult(JSON.stringify(res, null, 2))
      await poll()
    } catch (e) {
      setLastResult(`Error: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <Header orchestrator={orcHealth} eda={edaHealth} refinery={refineryHealth} ml={mlHealth} />
      <div className="main-layout">
        <div className="main-content">
          <ConsolePanel onSubmitPrompt={handleSubmitPrompt} lastResult={lastResult} busy={busy} />
          <WorkflowsPanel runs={runs} startBasicInfo={handleStartBasicInfo} />
        </div>
        <div className="sidebar">
          <ProcessesPanel orchestrator={orcHealth} eda={edaHealth} refinery={refineryHealth} />
          <DatasetsPanel datasets={datasets} onUpload={handleUpload} />
        </div>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)


