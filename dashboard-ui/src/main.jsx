import React from 'react'
import { createRoot } from 'react-dom/client'

function App() {
  return (
    <div style={{color:'#00eaff', background:'#0b0f14', minHeight:'100vh', fontFamily:'Inter, ui-sans-serif'}}> 
      <div style={{padding:'16px', borderBottom:'1px solid #122'}}>
        <h1 style={{margin:0, letterSpacing:'1px'}}>Deepline Dashboard (MVP)</h1>
        <small>Cyberpunk UI shell. Next: wire agents, workflows, datasets, monitors.</small>
      </div>
      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'12px', padding:'16px'}}>
        <section style={{border:'1px solid #122', borderRadius:8, padding:12}}>
          <h3 style={{marginTop:0}}>Console</h3>
          <p>Prompt input and NL→DSL translate (to be wired).</p>
        </section>
        <section style={{border:'1px solid #122', borderRadius:8, padding:12}}>
          <h3 style={{marginTop:0}}>Datasets</h3>
          <p>Upload/list datasets from orchestrator.</p>
        </section>
        <section style={{border:'1px solid #122', borderRadius:8, padding:12}}>
          <h3 style={{marginTop:0}}>Workflows</h3>
          <p>Runs, DAGs, artifacts.</p>
        </section>
        <section style={{border:'1px solid #122', borderRadius:8, padding:12}}>
          <h3 style={{marginTop:0}}>Agents</h3>
          <ul>
            <li>EDA (8001)</li>
            <li>Refinery (8005)</li>
            <li>FE submodule (via Refinery)</li>
            <li>ML (8002)</li>
          </ul>
        </section>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)


