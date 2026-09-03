import { useEffect, useRef, useState } from 'react'
import { registerWebMcpTools, type DatasetSnapshot, type Visualization } from './webmcp'
import './App.css'

type Analysis = DatasetSnapshot & { warnings: string[] }
type Answer = { type: string; value?: number; explanation?: string; data?: Record<string, unknown>[] }
const API_URL = import.meta.env.VITE_API_URL ?? ''

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [pending, setPending] = useState<Analysis | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [question, setQuestion] = useState('')
  const [conversation, setConversation] = useState<{ question: string; answer: Answer }[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [selectedChart, setSelectedChart] = useState<Visualization | null>(null)
  const [webMcpReady, setWebMcpReady] = useState(false)
  const analysisRef = useRef<Analysis | null>(null)
  const fileRef = useRef<File | null>(null)
  analysisRef.current = analysis
  fileRef.current = file

  const snapshot = (): DatasetSnapshot | null => analysisRef.current
  const askQuestion = async (text: string) => {
    if (!fileRef.current) return { error: 'No dataset is loaded.' }
    const body = new FormData()
    body.append('file', fileRef.current)
    body.append('question', text)
    const response = await fetch(`${API_URL}/api/ask`, { method: 'POST', body })
    const result = await response.json()
    if (!response.ok) throw new Error(result.detail ?? 'Could not answer the question.')
    setConversation((items) => [...items, { question: text, answer: result }])
    return result
  }

  useEffect(() => { setWebMcpReady(registerWebMcpTools(snapshot, askQuestion)) }, [])

  async function handleFile(selected: File | undefined) {
    if (!selected) return
    setFile(selected)
    setPending(null)
    setAnalysis(null)
    setConversation([])
    setSelectedChart(null)
    setBusy(true)
    setError('')
    try {
      const body = new FormData()
      body.append('file', selected)
      const response = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail ?? 'Could not analyze the CSV.')
      setPending(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not analyze the CSV.')
    } finally { setBusy(false) }
  }

  function confirmUpload() {
    if (!pending) return
    setAnalysis(pending)
    setPending(null)
  }

  async function submitQuestion() {
    const text = question.trim()
    if (!text) return
    setBusy(true)
    setError('')
    try { await askQuestion(text); setQuestion('') } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not answer the question.')
    } finally { setBusy(false) }
  }

  return <main className="shell">
    <header className="topbar"><div className="brand"><span className="mark">DL</span><span>DataLens</span></div><span className={webMcpReady ? 'status live' : 'status'}>{webMcpReady ? 'WebMCP ready' : 'WebMCP unavailable'}</span></header>
    <section className="intro"><p className="kicker">CSV intelligence workspace</p><h1>See the signal<br /><em>inside your data.</em></h1><p className="lede">Upload a dataset, get a clean profile in seconds, and ask grounded questions without handing your data to arbitrary generated code.</p></section>
    <section className="workspace">
      <aside className="rail"><div className="upload"><p className="eyebrow">01 / Load</p><label htmlFor="csv">Choose a CSV</label><input id="csv" type="file" accept=".csv,text/csv" onChange={(event) => void handleFile(event.target.files?.[0])} /><p className="hint">Your file stays in this browser session.</p></div>{pending && <div className="review"><p className="eyebrow">Review upload</p><strong>{pending.filename}</strong><span>{pending.profile.n_rows as number} rows · {pending.profile.n_columns as number} columns</span><button type="button" onClick={confirmUpload}>Confirm dataset <span>✓</span></button><button className="text-button" type="button" onClick={() => { setPending(null); setFile(null) }}>Choose another file</button></div>}{analysis && <div className="file-meta"><span>Confirmed dataset</span><strong>{analysis.filename}</strong><small>{analysis.profile.n_rows as number} rows · {analysis.profile.n_columns as number} columns</small></div>}</aside>
      <div className="content">{pending && !analysis ? <div className="confirm-state"><span className="empty-icon">✓</span><h2>Ready to explore</h2><p>Review the detected structure, then confirm this dataset to generate visual insights.</p><div className="mini-stats"><span><strong>{pending.profile.n_rows as number}</strong> rows</span><span><strong>{pending.profile.n_columns as number}</strong> columns</span><span><strong>{Object.keys(pending.profile.columns as object).length}</strong> detected</span></div></div> : !analysis ? <div className="empty"><span className="empty-icon">↗</span><h2>Drop in a CSV to begin</h2><p>DataLens will detect messy numbers, dates, missing values, and useful structure automatically.</p></div> : <Dashboard analysis={analysis} conversation={conversation} question={question} busy={busy} onQuestion={setQuestion} onSubmit={() => void submitQuestion()} onChartSelect={setSelectedChart} />}{busy && <p className="notice">Working through the data...</p>}{error && <p className="error">{error}</p>}</div>
    </section>
    {selectedChart && <ChartModal chart={selectedChart} onClose={() => setSelectedChart(null)} />}
  </main>
}

function Dashboard({ analysis, conversation, question, busy, onQuestion, onSubmit, onChartSelect }: { analysis: Analysis; conversation: { question: string; answer: Answer }[]; question: string; busy: boolean; onQuestion: (value: string) => void; onSubmit: () => void; onChartSelect: (chart: Visualization) => void }) {
  const columns = analysis.profile.columns as Record<string, { type: string; missing_count: number; missing_pct: number }>
  const missing = Object.values(columns).reduce((sum, column) => sum + column.missing_count, 0)
  return <>
    <div className="metrics"><div><span>Rows</span><strong>{analysis.profile.n_rows as number}</strong></div><div><span>Columns</span><strong>{analysis.profile.n_columns as number}</strong></div><div><span>Missing values</span><strong>{missing}</strong></div></div>
    <div className="panel"><div className="panel-head"><div><p className="eyebrow">02 / Inspect</p><h2>Automatic visual insights</h2></div><span className="badge">Click to expand</span></div><div className="charts">{analysis.visualizations?.map((chart) => <Chart key={chart.title} chart={chart} onOpen={() => onChartSelect(chart)} />)}</div></div>
    <div className="panel"><div className="panel-head"><div><p className="eyebrow">03 / Profile</p><h2>Column profile</h2></div></div><div className="table-wrap"><table><thead><tr><th>Column</th><th>Type</th><th>Missing</th></tr></thead><tbody>{Object.entries(columns).map(([name, info]) => <tr key={name}><td>{name}</td><td><span className="type">{info.type}</span></td><td>{info.missing_count} <small>({info.missing_pct}%)</small></td></tr>)}</tbody></table></div></div>
    <div className="panel ask-panel"><p className="eyebrow">04 / Ask</p><h2>Talk to the dataset</h2><div className="ask-row"><input value={question} onChange={(event) => onQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') onSubmit() }} placeholder="What is the average sales by category?" /><button type="button" onClick={onSubmit} disabled={busy}>Ask <span>↗</span></button></div>{conversation.map((item, index) => <div className="chat-turn" key={`${item.question}-${index}`}><div className="bubble user-bubble">{item.question}</div><div className="bubble assistant-bubble"><strong>{item.answer.type === 'scalar' ? item.answer.value : 'Here is what I found'}</strong><p>{item.answer.explanation ?? 'I analyzed the confirmed dataset using validated operations.'}</p>{item.answer.data && <div className="result-table">{item.answer.data.slice(0, 8).map((row, rowIndex) => <div className="result-row" key={rowIndex}>{Object.entries(row).map(([key, value]) => <span key={key}><small>{key}</small>{String(value)}</span>)}</div>)}</div>}</div></div>)}</div>
  </>
}

function Chart({ chart, onOpen }: { chart: Visualization; onOpen: () => void }) {
  const max = Math.max(...chart.values, 1)
  return <button className="chart-card" type="button" onClick={onOpen} aria-label={`Expand ${chart.title}`}><h3>{chart.title}<span className="expand-icon">↗</span></h3>{chart.kind === 'donut' ? <div className="donut" style={{ background: `conic-gradient(#1d7759 0 ${chart.values[0] / chart.values.reduce((sum, value) => sum + value, 0) * 360}deg, #e75b38 0 220deg, #e2b84b 0 290deg, #bfc8bc 0)` }}><span>{chart.values.reduce((sum, value) => sum + value, 0)}</span></div> : chart.kind === 'line' ? <div className="line-chart">{chart.values.map((value, index) => <span key={index} style={{ left: `${(index / Math.max(chart.values.length - 1, 1)) * 100}%`, bottom: `${Math.max((value / max) * 80, 8)}%` }} />)}</div> : chart.kind === 'scatter' ? <div className="scatter-chart">{chart.values.map((value, index) => <span key={index} style={{ left: `${(index / Math.max(chart.values.length - 1, 1)) * 92 + 4}%`, bottom: `${Math.max((value / max) * 82, 5)}%` }} />)}</div> : <div className="bars">{chart.values.map((value, index) => <div className="bar-item" key={`${chart.labels[index]}-${index}`}><div className="bar-track"><div className={`bar-fill ${chart.kind === 'histogram' ? 'orange' : ''}`} style={{ height: `${Math.max((value / max) * 100, 5)}%` }} /></div><span title={chart.labels[index]}>{chart.labels[index]}</span><b>{value}</b></div>)}</div>}</button>
}

function ChartModal({ chart, onClose }: { chart: Visualization; onClose: () => void }) {
  const max = Math.max(...chart.values, 1)
  return <div className="modal-backdrop" role="presentation" onClick={onClose}><section className="chart-modal" role="dialog" aria-modal="true" aria-label={chart.title} onClick={(event) => event.stopPropagation()}><div className="modal-head"><div><p className="eyebrow">Expanded insight</p><h2>{chart.title}</h2></div><button className="close-button" type="button" onClick={onClose} aria-label="Close expanded chart">×</button></div><div className="large-chart">{chart.kind === 'donut' ? <div className="donut large" style={{ background: `conic-gradient(#1d7759 0 ${chart.values[0] / chart.values.reduce((sum, value) => sum + value, 0) * 360}deg, #e75b38 0 220deg, #e2b84b 0 290deg, #bfc8bc 0)` }}><span>{chart.values.reduce((sum, value) => sum + value, 0)}</span></div> : chart.kind === 'scatter' ? <div className="large-scatter">{chart.values.map((value, index) => <span key={index} title={`${chart.labels[index]} / ${value}`} style={{ left: `${(index / Math.max(chart.values.length - 1, 1)) * 94 + 3}%`, bottom: `${Math.max((value / max) * 88, 4)}%` }} />)}</div> : chart.kind === 'line' ? <div className="large-line">{chart.values.map((value, index) => <span key={index} title={`${chart.labels[index]} / ${value}`} style={{ left: `${(index / Math.max(chart.values.length - 1, 1)) * 94 + 3}%`, bottom: `${Math.max((value / max) * 88, 4)}%` }} />)}</div> : <div className="large-bars">
    {chart.values.map((value, index) => <div className="large-bar-item" key={`${chart.labels[index]}-${index}`}><div className="large-track"><div className="bar-fill" style={{ height: `${Math.max((value / max) * 100, 4)}%` }} /></div><b>{value}</b><span>{chart.labels[index]}</span></div>)}
  </div>}</div><div className="full-values">{chart.values.map((value, index) => <div key={`${chart.labels[index]}-value`}><span>{chart.labels[index]}</span><strong>{value}</strong></div>)}</div></section></div>
}

export default App
