import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  BookOpen,
  Bot,
  CheckCircle,
  Coffee,
  Copy,
  Database,
  Droplets,
  Gauge,
  Images,
  Milk,
  RefreshCw,
  Send,
  Settings,
  X,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { askManual, checkHealth as checkApiHealth, imageUrl, ingestDefaultManual } from './services/api'
import './App.css'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  images?: string[]
}

const SUGGESTED_QUESTIONS = [
  'How do I descale my Saeco GranAroma?',
  'How can I adjust the coffee strength?',
  'What is the best way to clean the milk carafe?',
  'How do I change the water filter?',
]

const CAPABILITIES = [
  {
    label: 'Brewing & drink settings',
    question: 'How do I adjust brewing and drink settings?',
    Icon: Coffee,
  },
  {
    label: 'Descaling & water filters',
    question: 'What is the process for descaling and replacing water filters?',
    Icon: Droplets,
  },
  {
    label: 'Milk carafe cleaning',
    question: 'How do I clean the milk carafe?',
    Icon: Milk,
  },
  {
    label: 'Grinder & bean settings',
    question: 'How do I adjust the grinder and bean settings?',
    Icon: Settings,
  },
  {
    label: 'Manual figures & diagrams',
    question: 'Show me manual figures and diagrams.',
    Icon: Images,
  },
  {
    label: 'Maintenance schedules',
    question: 'What is the recommended maintenance schedule?',
    Icon: Gauge,
  },
]

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content:
        'I have the **GranAroma 6850 manual** ready. Ask about cleaning, descaling, drink settings, water filters, or any maintenance step and I will answer from the manual.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [backendStatus, setBackendStatus] = useState<'online' | 'offline' | 'checking'>('checking')
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [ingestStatus, setIngestStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const chatAreaRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const chat = chatAreaRef.current
    if (chat) chat.scrollTop = chat.scrollHeight
  }, [messages, isLoading])

  useEffect(() => {
    const checkHealth = async () => {
      try {
        setBackendStatus((await checkApiHealth()) ? 'online' : 'offline')
      } catch {
        setBackendStatus('offline')
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSelectedImage(null)
    }

    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [])

  const handleSend = async (text: string = input) => {
    const query = text.trim()
    if (!query || isLoading) return

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }

    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    try {
      const data = await askManual(query)

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer || "I could not process that response.",
        images: data.context_images || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }
      setMessages((prev) => [...prev, assistantMsg])
    } catch (error) {
      console.error(error)
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: error instanceof Error
            ? error.message
            : 'The manual service could not answer. Please try again.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleIngest = async () => {
    setIngestStatus('running')

    try {
      const message = await ingestDefaultManual()
      setIngestStatus('done')
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `**Index updated.** ${message}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ])
    } catch (err) {
      console.error(err)
      setIngestStatus('error')
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `**Index update failed.** ${err instanceof Error ? err.message : 'The backend did not accept the ingestion request.'}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ])
    }
  }

  const copyToClipboard = async (message: Message) => {
    await navigator.clipboard.writeText(message.content)
    setCopiedId(message.id)
    window.setTimeout(() => setCopiedId(null), 1600)
  }

  const handleInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') handleSend()
  }

  const renderSuggestions = (className = '') => (
    <div className={`suggested-questions ${className}`}>
      {SUGGESTED_QUESTIONS.map((question) => (
        <button
          key={question}
          className="suggestion-btn"
          onClick={() => handleSend(question)}
          disabled={isLoading}
        >
          {question}
        </button>
      ))}
    </div>
  )

  const backendOnline = backendStatus === 'online'
  const statusLabel = backendStatus === 'checking' ? 'Checking' : backendOnline ? 'Service ready' : 'Service offline'
  const assistantState = backendOnline ? 'Manual service connected' : 'Manual service unavailable'
  const ingestLabel =
    ingestStatus === 'running'
      ? 'Updating index'
      : ingestStatus === 'done'
        ? 'Index refreshed'
        : ingestStatus === 'error'
          ? 'Retry index'
          : 'Update index'

  const statusIcon = backendOnline ? <CheckCircle size={16} aria-hidden="true" /> : <AlertCircle size={16} aria-hidden="true" />
  const ingestIcon =
    ingestStatus === 'running'
      ? <RefreshCw size={16} aria-hidden="true" className="spin-icon" />
      : <Database size={16} aria-hidden="true" />

  return (
    <div className="app-container">
      <aside className="sidebar">
        <header className="sidebar-header">
          <div className="logo-section">
            <div className="logo-container">
              <Coffee size={22} aria-hidden="true" />
            </div>
            <div>
              <p className="eyebrow">Saeco GranAroma</p>
              <h1>Coffee Maestro</h1>
            </div>
          </div>
          <p className="subtitle">A polished companion for cleaning, settings, parts, and maintenance.</p>
        </header>

        <nav className="sidebar-section">
          <h2>Manual service</h2>
          <div className="status-card">
            <div className={`status-item ${backendOnline ? 'online' : 'offline'}`}>
              <span className="status-copy">
                {statusIcon}
                Backend API
              </span>
              <span className="status-pill">{statusLabel}</span>
            </div>
            <div className="status-note">Answers are grounded in indexed manual text and reference figures.</div>
          </div>
        </nav>

        <section className="sidebar-section">
          <h2>Quick questions</h2>
          {renderSuggestions()}
        </section>

        <div className="sidebar-section sidebar-footer">
          <p className="fine-print">Uses the local GranAroma PDF and extracted figures.</p>
        </div>
      </aside>

      <main className="chat-main">
        <header className="header">
          <div className="assistant-identity">
            <div className="bot-icon-container">
              <Bot size={20} aria-hidden="true" />
            </div>
            <div>
              <span className="assistant-title">Saeco Assistant</span>
              <span className={`assistant-status ${backendOnline ? 'online' : 'offline'}`}>{assistantState}</span>
            </div>
          </div>
          <div className="header-actions">
            <span className="header-badge">
              <BookOpen size={16} aria-hidden="true" />
              Manual indexed
            </span>
            <button className="utility-btn compact" onClick={handleIngest} disabled={ingestStatus === 'running'}>
              {ingestIcon}
              <span>{ingestLabel}</span>
            </button>
          </div>
        </header>

        <section className="brand-panel" aria-labelledby="brand-panel-title">
          <div className="brand-identity">
            <div className="brand-identity-copy">
              <p className="eyebrow">Saeco GranAroma 6850</p>
              <h2 id="brand-panel-title">Your machine's manual, on call.</h2>
              <p className="brand-sub">Ask Saeco Assistant anything from your GranAroma 6850 manual. Answers are pulled directly from the indexed PDF, with manual figures attached when a step needs a visual.</p>
            </div>
            <div className="brand-caps" aria-label="What Saeco Assistant can help with" role="group">
              {CAPABILITIES.map(({ label, question, Icon }) => (
                <button type="button" className="cap-btn" key={label} onClick={() => handleSend(question)} disabled={isLoading}>
                  <span className="cap-icon"><Icon size={18} aria-hidden="true" /></span>
                  <span className="cap-label">{label}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="mobile-suggestions" aria-label="Suggested questions">
          {renderSuggestions('horizontal')}
        </section>

        <div className="chat-area" ref={chatAreaRef}>
          {messages.map((msg) => (
            <div key={msg.id} className={`message-wrapper ${msg.role}`}>
              <div className="message-bubble">
                {msg.role === 'assistant' && <span className="message-kicker">Saeco Assistant</span>}
                <div className="markdown-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                </div>

                {msg.images && msg.images.length > 0 && (
                  <div className="context-images">
                    {msg.images.map((img, idx) => {
                      const referenceImageUrl = imageUrl(img)
                      return (
                        <button
                          key={img}
                          className="image-card"
                          onClick={() => setSelectedImage(referenceImageUrl)}
                          aria-label={`Open manual reference figure ${idx + 1}`}
                        >
                          <img src={referenceImageUrl} alt={`Manual reference figure ${idx + 1}`} />
                          <span className="image-overlay">Open figure {idx + 1}</span>
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
              <div className="message-info">
                <span>{msg.timestamp}</span>
                {msg.role === 'assistant' && (
                  <button className="copy-btn" onClick={() => copyToClipboard(msg)}>
                    <Copy size={13} aria-hidden="true" />
                    <span>{copiedId === msg.id ? 'Copied' : 'Copy'}</span>
                  </button>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message-wrapper assistant">
              <div className="message-bubble loading-bubble" role="status" aria-live="polite" aria-atomic="true">
                <span className="loading-label">Checking the manual…</span>
                <div className="typing-indicator" aria-hidden="true">
                  <div className="dot dot-1"></div>
                  <div className="dot dot-2"></div>
                  <div className="dot"></div>
                </div>
              </div>
            </div>
          )}
        </div>

        <section className="input-section">
          <div className="input-container">
            <Coffee size={20} aria-hidden="true" />
            <label className="sr-only" htmlFor="manual-question">
              Ask a question about the GranAroma manual
            </label>
            <input
              id="manual-question"
              type="text"
              className="chat-input"
              placeholder="Ask about descaling, filters, milk cleaning..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleInputKeyDown}
              disabled={isLoading}
            />
            <button className="action-btn send-btn" onClick={() => handleSend()} disabled={!input.trim() || isLoading} aria-label="Send question">
              <Send size={18} aria-hidden="true" />
            </button>
          </div>
        </section>
      </main>

      {selectedImage && (
        <div className="lightbox-overlay" onClick={() => setSelectedImage(null)} role="dialog" aria-modal="true" aria-label="Manual reference figure">
          <div className="lightbox-content" onClick={(event) => event.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setSelectedImage(null)} aria-label="Close reference figure">
              <X size={22} aria-hidden="true" />
            </button>
            <img src={selectedImage} alt="Zoomed manual reference" className="lightbox-image" />
            <div className="lightbox-caption">Manual reference figure</div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
