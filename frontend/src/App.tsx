import { useEffect, useRef, useState } from 'react'
import { registerWebMcpTools, type DatasetSnapshot } from './webmcp'
import './App.css'

type Analysis = DatasetSnapshot & { warnings: string[] }
type Answer = { type: string; value?: number; explanation?: string; data?: Record<string, unknown>[] }

const API_URL = import.meta.env.VITE_API_URL ?? ''

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
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
    setAnswer(result)
    return result
  }

  useEffect(() => {
    setWebMcpReady(registerWebMcpTools(snapshot, askQuestion))
  }, [])

  async function handleFile(selected: File | undefined) {
    if (!selected) return
    setFile(selected)
    setBusy(true)
    setError('')
    try {
      const body = new FormData()
      body.append('file', selected)
      const response = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail ?? 'Could not analyze the CSV.')
      setAnalysis(result)
      setAnswer(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not analyze the CSV.')
      setAnalysis(null)
    } finally {
      setBusy(false)
    }
  }

  async function submitQuestion() {
    if (!question.trim()) return
    setBusy(true)
    setError('')
    try { await askQuestion(question.trim()) } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not answer the question.')
    } finally { setBusy(false) }
  }

  return (
    <main className="shell">
      <header className="topbar"><div className="brand"><span className="mark">DL</span><span>DataLens</span></div><span className={webMcpReady ? 'status live' : 'status'}>{webMcpReady ? 'WebMCP ready' : 'WebMCP unavailable'}</span></header>
      <section className="intro"><p className="kicker">CSV intelligence workspace</p><h1>See the signal<br /><em>inside your data.</em></h1><p className="lede">Upload a dataset, get a clean profile in seconds, and ask grounded questions without handing your data to arbitrary generated code.</p></section>
      <section className="workspace">
        <aside className="rail"><div className="upload"><p className="eyebrow">01 / Load</p><label htmlFor="csv">Choose a CSV</label><input id="csv" type="file" accept=".csv,text/csv" onChange={(event) => void handleFile(event.target.files?.[0])} /><p className="hint">Your file stays in this browser session.</p></div>{analysis && <div className="file-meta"><span>Loaded file</span><strong>{analysis.filename}</strong><small>{analysis.profile.n_rows as number} rows · {analysis.profile.n_columns as number} columns</small></div>}</aside>
        <div className="content">{!analysis ? <div className="empty"><span className="empty-icon">↗</span><h2>Drop in a CSV to begin</h2><p>DataLens will detect messy numbers, dates, missing values, and useful structure automatically.</p></div> : <><div className="metrics"><div><span>Rows</span><strong>{analysis.profile.n_rows as number}</strong></div><div><span>Columns</span><strong>{analysis.profile.n_columns as number}</strong></div><div><span>Missing values</span><strong>{Object.values(analysis.profile.columns as Record<string, { missing_count: number }>).reduce((sum, column) => sum + column.missing_count, 0)}</strong></div></div><div className="panel"><div className="panel-head"><div><p className="eyebrow">02 / Inspect</p><h2>Column profile</h2></div><span className="badge">{webMcpReady ? 'Agent accessible' : 'Local only'}</span></div><div className="table-wrap"><table><thead><tr><th>Column</th><th>Type</th><th>Missing</th></tr></thead><tbody>{Object.entries(analysis.profile.columns as Record<string, { type: string; missing_count: number; missing_pct: number }>).map(([name, info]) => <tr key={name}><td>{name}</td><td><span className="type">{info.type}</span></td><td>{info.missing_count} <small>({info.missing_pct}%)</small></td></tr>)}</tbody></table></div></div><div className="panel ask-panel"><p className="eyebrow">03 / Ask</p><h2>Talk to the dataset</h2><div className="ask-row"><input value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void submitQuestion() }} placeholder="What is the average sales by category?" /><button type="button" onClick={() => void submitQuestion()} disabled={busy}>Ask <span>↗</span></button></div>{answer && <div className="answer"><strong>{answer.type === 'scalar' ? answer.value : 'Answer ready'}</strong><p>{answer.explanation ?? 'The result is shown below.'}</p>{answer.data && <pre>{JSON.stringify(answer.data.slice(0, 8), null, 2)}</pre>}</div>}</div></>}{busy && <p className="notice">Working through the data...</p>}{error && <p className="error">{error}</p>}</div>
      </section>
    </main>
  )
}

export default App
