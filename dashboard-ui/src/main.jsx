import React, { useEffect, useRef, useState, useMemo } from 'react'
import { createRoot } from 'react-dom/client'
import { Play, Upload, Activity, Database, Cpu, Terminal, TrendingUp } from 'lucide-react'
import './index.css'

const API = {
  orchestratorBase: 'http://localhost:8000',
  edaBase: 'http://localhost:8001',
  refineryBase: 'http://localhost:8005',
}

function useInterval(callback, delayMs) {
  const saved = useRef(callback)
  useEffect(() => { saved.current = callback }, [callback])
  useEffect(() => {
    if (delayMs == null) return
    const id = setInterval(() => saved.current(), delayMs)
    return () => clearInterval(id)
  }, [delayMs])
}

async function fetchJson(url, options) {
  const res = await fetch(url, options)
  if (!res.ok) throw new Error(`${res.status}`)
  return res.json()
}

function Header({ orc, eda, refi }) {
  const Pill = ({ ok, label }) => (
    <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs border ${ok ? 'border-emerald-400 text-emerald-300' : 'border-rose-400 text-rose-300'} bg-white/5`}>
      <span className={`w-2 h-2 rounded-full ${ok ? 'bg-emerald-400' : 'bg-rose-400'}`} /> {label}
    </span>
  )
  return (
    <header className="sticky top-0 z-10 bg-gray-900/80 backdrop-blur border-b border-gray-800">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <h1 className="tracking-[0.2em] text-sm font-semibold text-cyan-300">DEEPLINE</h1>
        <div className="flex gap-2">
          <Pill ok={!!orc?.healthy} label="Orchestrator" />
          <Pill ok={!!eda?.healthy} label="EDA" />
          <Pill ok={!!refi?.healthy} label="Refinery" />
        </div>
      </div>
    </header>
  )
}

function Console({ onSubmit, logs, busy }) {
  const [text, setText] = useState('Analyze iris dataset basic stats')
  return (
    <section className="bg-gray-900/60 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3 text-cyan-300">
        <Terminal className="w-4 h-4" /><h2 className="font-medium">Console</h2>
      </div>
      <textarea className="w-full h-24 rounded-md bg-black/30 border border-gray-700 px-3 py-2 text-sm font-mono text-cyan-100 focus:outline-none focus:border-cyan-500" value={text} onChange={e=>setText(e.target.value)} />
      <div className="mt-2 flex justify-end">
        <button disabled={busy} onClick={()=>onSubmit(text)} className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-cyan-500 hover:bg-cyan-600 text-gray-900 text-sm disabled:opacity-50">
          <Play className="w-4 h-4" /> {busy? 'Running…' : 'Run'}
        </button>
      </div>
      <div className="mt-3 h-28 overflow-auto rounded-md bg-black/30 border border-gray-800 p-2 text-xs font-mono text-cyan-100">
        {logs?.length ? logs.map((l,i)=>(
          <div key={i} className={l.type==='error'?'text-rose-300':l.type==='success'?'text-emerald-300':'text-cyan-200'}>{l.message}</div>
        )) : <div className="text-gray-400">Awaiting action…</div>}
      </div>
    </section>
  )
}

function Workflows({ runs, onQuickStart }) {
  const color = (s)=> s==='COMPLETED'?'text-emerald-400': s==='RUNNING' ? 'text-cyan-400':'text-amber-400'
  return (
    <section className="bg-gray-900/60 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-cyan-300"><Activity className="w-4 h-4"/><h2 className="font-medium">Workflows</h2></div>
        <button onClick={onQuickStart} className="px-3 py-1.5 rounded-md border border-gray-700 hover:border-cyan-500 text-xs">Quick-Start: Basic Info</button>
      </div>
      <div className="space-y-2">
        {(runs?.runs ?? []).slice().reverse().slice(0,8).map(r=> (
          <div key={r.run_id} className="bg-black/30 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <div className="text-gray-300 font-mono text-xs truncate mr-3">{r.run_id}</div>
              <div className={`text-xs font-semibold ${color(r.status)}`}>{r.status}</div>
            </div>
            <div className="mt-2 w-full bg-gray-800 h-1.5 rounded-full">
              <div className="h-1.5 rounded-full bg-cyan-500" style={{width:`${Math.round(r.progress)}%`}} />
            </div>
          </div>
        ))}
        {!(runs?.runs||[]).length && <div className="text-gray-400 text-sm">No runs</div>}
      </div>
    </section>
  )
}

function Processes({ orc, eda, refi }) {
  const edaOps = useMemo(()=>{
    const t = eda?.telemetry || {}
    return Object.keys(t).map(k=>({op:k, count:t[k]?.count??0, errors:t[k]?.errors??0}))
  },[eda])
  return (
    <section className="bg-gray-900/60 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center gap-2 text-cyan-300 mb-3"><Cpu className="w-4 h-4"/><h2 className="font-medium">Background Processes</h2></div>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-black/30 border border-gray-800 rounded-lg p-3 text-sm">
          <div className="text-cyan-300 font-medium mb-1">Orchestrator</div>
          <div className="flex justify-between text-gray-300"><span>healthy</span><span>{String(!!orc?.healthy)}</span></div>
          <div className="flex justify-between text-gray-300"><span>time</span><span className="truncate ml-3">{orc?.timestamp}</span></div>
          <div className="flex justify-between text-gray-300"><span>agents</span><span>{Object.keys(orc?.agents||{}).length}</span></div>
        </div>
        <div className="bg-black/30 border border-gray-800 rounded-lg p-3 text-sm">
          <div className="text-cyan-300 font-medium mb-1">Refinery</div>
          <div className="flex justify-between text-gray-300"><span>healthy</span><span>{String(!!refi?.healthy)}</span></div>
          <div className="flex justify-between text-gray-300"><span>fe_module_available</span><span>{String(!!refi?.fe_module_available)}</span></div>
          <div className="flex justify-between text-gray-300"><span>fe_module_enabled</span><span>{String(!!refi?.fe_module_enabled)}</span></div>
        </div>
      </div>
      <div className="mt-3">
        <div className="flex items-center gap-2 text-cyan-300 mb-2"><TrendingUp className="w-4 h-4"/><h3 className="text-sm font-medium">EDA Telemetry</h3></div>
        <div className="border border-gray-800 rounded-lg overflow-hidden">
          <div className="grid grid-cols-3 text-xs bg-black/40 text-gray-300">
            <div className="px-2 py-1 border-r border-gray-800">Operation</div>
            <div className="px-2 py-1 border-r border-gray-800">Count</div>
            <div className="px-2 py-1">Errors</div>
          </div>
          {edaOps.length ? edaOps.map(o=> (
            <div key={o.op} className="grid grid-cols-3 text-sm text-gray-200 border-t border-gray-800">
              <div className="px-2 py-1">{o.op}</div>
              <div className="px-2 py-1">{o.count}</div>
              <div className={`px-2 py-1 ${o.errors?'text-rose-300':'text-emerald-300'}`}>{o.errors}</div>
            </div>
          )) : <div className="px-2 py-2 text-gray-400 text-sm">No telemetry</div>}
        </div>
      </div>
    </section>
  )
}

function Datasets({ datasets, onUpload }) {
  const [file, setFile] = useState(null)
  const [name, setName] = useState('uploaded')
  return (
    <section className="bg-gray-900/60 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center gap-2 text-cyan-300 mb-3"><Database className="w-4 h-4"/><h2 className="font-medium">Datasets</h2></div>
      <div className="flex gap-2 mb-3">
        <input type="file" onChange={e=>setFile(e.target.files?.[0]??null)} className="text-sm text-gray-300" />
        <input value={name} onChange={e=>setName(e.target.value)} className="flex-1 bg-black/30 border border-gray-700 rounded-md px-2 text-sm" placeholder="dataset name" />
        <button onClick={()=>file && onUpload(file,name)} className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-700 hover:border-cyan-500 text-xs">
          <Upload className="w-4 h-4"/> Upload
        </button>
      </div>
      <div className="border border-gray-800 rounded-lg overflow-hidden">
        <div className="grid grid-cols-3 text-xs bg-black/40 text-gray-300">
          <div className="px-2 py-1 border-r border-gray-800">Name</div>
          <div className="px-2 py-1 border-r border-gray-800">Rows×Cols</div>
          <div className="px-2 py-1">Mem</div>
        </div>
        {datasets?.datasets?.length ? datasets.datasets.map((d,i)=> (
          <div key={i} className="grid grid-cols-3 text-sm text-gray-200 border-t border-gray-800">
            <div className="px-2 py-1">{d.name}</div>
            <div className="px-2 py-1">{d.shape?.[0]} × {d.shape?.[1]}</div>
            <div className="px-2 py-1">{d.memory_usage}</div>
          </div>
        )) : <div className="px-2 py-2 text-gray-400 text-sm">No datasets</div>}
      </div>
    </section>
  )
}

function App(){
  const [orc,setOrc]=useState(null)
  const [eda,setEda]=useState(null)
  const [refi,setRefi]=useState(null)
  const [datasets,setDatasets]=useState(null)
  const [runs,setRuns]=useState(null)
  const [logs,setLogs]=useState([])
  const [busy,setBusy]=useState(false)

  const poll = async ()=>{
    const [o,e,r,ds,rs] = await Promise.all([
      fetchJson(`${API.orchestratorBase}/health`).catch(()=>null),
      fetchJson(`${API.edaBase}/health`).catch(()=>null),
      fetchJson(`${API.refineryBase}/health`).catch(()=>null),
      fetchJson(`${API.orchestratorBase}/datasets`).catch(()=>({datasets:[]})),
      fetchJson(`${API.orchestratorBase}/runs`).catch(()=>({runs:[]})),
    ])
    setOrc(o && {healthy:true,...o})
    setEda(e && {healthy:true,...e})
    setRefi(r && {healthy:true,...r})
    setDatasets(ds)
    setRuns(rs)
  }
  useEffect(()=>{ poll() },[])
  useInterval(poll,2000)

  function plan(prompt){
    const p=(prompt||'').toLowerCase(); const tasks=[]
    if(p.includes('load iris')) tasks.push({agent:'eda_agent',action:'load_data',args:{path:'/app/iris.csv',name:'iris',file_type:'csv'}})
    if(p.includes('basic')||p.includes('summary')||p.includes('info')) tasks.push({agent:'eda_agent',action:'basic_info',args:{name:'iris'}})
    if(!tasks.length) tasks.push({agent:'eda_agent',action:'basic_info',args:{name:'iris'}})
    return tasks
  }

  async function onSubmit(text){
    setBusy(true)
    setLogs(l=>[...l,{type:'command',message:`> ${text}` }])
    try{
      if((text||'').toLowerCase().includes('load iris')){
        await fetchJson(`${API.edaBase}/load_data`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:'/app/iris.csv',name:'iris',file_type:'csv'})}).catch(()=>null)
      }
      const res = await fetchJson(`${API.orchestratorBase}/workflows/start`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_name:`NL Run - ${new Date().toLocaleTimeString()}`,tasks:plan(text)})})
      setLogs(l=>[...l,{type:'success',message:`Workflow ${res.run_id} started` }])
      await poll()
    }catch(e){
      setLogs(l=>[...l,{type:'error',message:`Error: ${e.message}`}])
    }finally{ setBusy(false) }
  }

  async function quickStart(){
    await onSubmit('basic info iris')
  }

  async function upload(file,name){
    const form=new FormData(); form.append('file',file); form.append('name',name)
    await fetch(`${API.orchestratorBase}/datasets/upload`,{method:'POST',body:form}).catch(()=>null)
    await poll()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-900 to-gray-950 text-gray-100">
      <Header orc={orc} eda={eda} refi={refi} />
      <main className="max-w-6xl mx-auto p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-4">
          <Console onSubmit={onSubmit} logs={logs} busy={busy} />
          <Workflows runs={runs} onQuickStart={quickStart} />
        </div>
        <div className="space-y-4">
          <Processes orc={orc} eda={eda} refi={refi} />
          <Datasets datasets={datasets} onUpload={upload} />
        </div>
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)


