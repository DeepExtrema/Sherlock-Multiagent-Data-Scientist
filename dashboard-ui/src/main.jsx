import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
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

function StatusPill({ label, ok }) {
  return (
    <div className={`pill ${ok ? 'ok' : 'bad'}`}>
      <span className="dot" />
      <span>{label}</span>
    </div>
  )
}

function TopBar({ orchestrator, eda, refinery, ml }) {
  return (
    <div className="topbar">
      <div className="topbar-inner">
        <div className="brand">DEEPLINE</div>
        <div className="status">
          <StatusPill label="Orchestrator" ok={!!orchestrator?.healthy} />
          <StatusPill label="EDA" ok={!!eda?.healthy} />
          <StatusPill label="Refinery" ok={!!refinery?.healthy} />
          <StatusPill label="ML" ok={!!ml?.healthy} />
        </div>
      </div>
    </div>
  )
}

function ConsolePanel({ onSubmitPrompt, lastResult, busy }) {
  const [nlPrompt, setNlPrompt] = useState('Analyze iris dataset basic stats')
  return (
    <section className="panel">
      <div className="panel-header">Console</div>
      <div className="console">
        <textarea value={nlPrompt} onChange={e => setNlPrompt(e.target.value)} className="prompt" placeholder="Ask Deepline to analyze your data..." />
        <div className="actions">
          <button className="btn neon" disabled={busy} onClick={() => onSubmitPrompt(nlPrompt)}>{busy ? 'Running...' : 'Run'}</button>
        </div>
        <div className="result mono">
          {lastResult ? <pre>{lastResult}</pre> : <span className="muted">Awaiting action…</span>}
        </div>
      </div>
    </section>
  )
}

function DatasetsPanel({ datasets, onUpload }) {
  const [name, setName] = useState('uploaded')
  const [file, setFile] = useState(null)
  return (
    <section className="panel">
      <div className="panel-header">Datasets</div>
      <div className="vspace">
        <div className="upload">
          <input type="file" onChange={e => setFile(e.target.files?.[0] ?? null)} />
          <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="dataset name" />
          <button className="btn" onClick={() => file && onUpload(file, name)}>Upload</button>
        </div>
        <div className="table">
          <div className="trow thead"><div>Name</div><div>Rows×Cols</div><div>Mem</div></div>
          {datasets?.datasets?.map((d, i) => (
            <div key={i} className="trow">
              <div>{d.name}</div>
              <div>{d.shape?.[0]} × {d.shape?.[1]}</div>
              <div>{d.memory_usage}</div>
            </div>
          ))}
          {!datasets?.datasets?.length && <div className="muted">No datasets</div>}
        </div>
      </div>
    </section>
  )
}

function WorkflowsPanel({ runs, startBasicInfo }) {
  const [starting, setStarting] = useState(false)
  const onStart = async () => {
    setStarting(true)
    try { await startBasicInfo() } finally { setStarting(false) }
  }
  return (
    <section className="panel">
      <div className="panel-header">Workflows</div>
      <div className="vspace">
        <button className="btn" disabled={starting} onClick={onStart}>Quick-Start: Basic Info (iris)</button>
        <div className="table">
          <div className="trow thead"><div>Run ID</div><div>Status</div><div>Progress</div></div>
          {(runs?.runs ?? []).slice().reverse().slice(0, 10).map((r) => (
            <div key={r.run_id} className={`trow ${r.status}`}>
              <div className="mono small">{r.run_id}</div>
              <div>{r.status}</div>
              <div>{Math.round(r.progress)}%</div>
            </div>
          ))}
          {!runs?.runs?.length && <div className="muted">No runs</div>}
        </div>
      </div>
    </section>
  )
}

function ProcessesPanel({ orchestrator, eda, refinery }) {
  const edaOps = useMemo(() => {
    const t = eda?.telemetry || {}
    return Object.keys(t).map(k => ({ op: k, count: t[k]?.count ?? 0, errors: t[k]?.errors ?? 0 }))
  }, [eda])
  return (
    <section className="panel">
      <div className="panel-header">Background Processes</div>
      <div className="grid two">
        <div>
          <div className="subhead">Orchestrator</div>
          <div className="kv"><span>healthy</span><span>{String(!!orchestrator?.healthy)}</span></div>
          <div className="kv"><span>time</span><span>{orchestrator?.timestamp}</span></div>
          <div className="kv"><span>agents</span><span>{Object.keys(orchestrator?.agents || {}).length}</span></div>
        </div>
        <div>
          <div className="subhead">Refinery</div>
          <div className="kv"><span>healthy</span><span>{String(!!refinery?.healthy)}</span></div>
          <div className="kv"><span>fe_module_available</span><span>{String(!!refinery?.fe_module_available)}</span></div>
          <div className="kv"><span>fe_module_enabled</span><span>{String(!!refinery?.fe_module_enabled)}</span></div>
        </div>
        <div className="span2">
          <div className="subhead">EDA Telemetry</div>
          <div className="table">
            <div className="trow thead"><div>Operation</div><div>Count</div><div>Errors</div></div>
            {edaOps.map((o) => (
              <div key={o.op} className="trow">
                <div>{o.op}</div>
                <div>{o.count}</div>
                <div className={o.errors ? 'bad' : ''}>{o.errors}</div>
              </div>
            ))}
            {!edaOps.length && <div className="muted">No telemetry yet</div>}
          </div>
        </div>
      </div>
    </section>
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
      <TopBar orchestrator={orcHealth} eda={edaHealth} refinery={refineryHealth} ml={mlHealth} />
      <div className="container">
        <div className="grid main simple">
          <div className="col">
            <ConsolePanel onSubmitPrompt={handleSubmitPrompt} lastResult={lastResult} busy={busy} />
            <WorkflowsPanel runs={runs} startBasicInfo={handleStartBasicInfo} />
          </div>
          <div className="col">
            <ProcessesPanel orchestrator={orcHealth} eda={edaHealth} refinery={refineryHealth} />
            <DatasetsPanel datasets={datasets} onUpload={handleUpload} />
          </div>
        </div>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)


