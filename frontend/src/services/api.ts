export interface AskResponse {
  answer: string
  context_images: string[]
  citations?: Array<Record<string, unknown>>
  quality_warnings?: string[]
  agent_trace?: string[]
}

const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || ''

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`
}

function headers() {
  return {
    'Content-Type': 'application/json',
  }
}

async function readJson(response: Response) {
  const text = await response.text()
  if (!text) return {}
  try {
    return JSON.parse(text) as unknown
  } catch {
    throw new Error(text)
  }
}

function getErrorMessage(payload: unknown, fallback: string) {
  if (payload && typeof payload === 'object') {
    const detail = 'detail' in payload ? payload.detail : undefined
    const message = 'message' in payload ? payload.message : undefined
    if (typeof detail === 'string') return detail
    if (typeof message === 'string') return message
  }
  return fallback
}

export async function checkHealth() {
  const response = await fetch(apiUrl('/health'))
  return response.ok
}

export async function askManual(query: string): Promise<AskResponse> {
  const response = await fetch(apiUrl('/api/ask'), {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ query }),
  })
  const payload = await readJson(response)
  if (!response.ok) {
    throw new Error(getErrorMessage(payload, 'The manual service could not answer.'))
  }
  if (!payload || typeof payload !== 'object' || !('answer' in payload)) {
    throw new Error('The manual service returned an invalid response.')
  }

  const data = payload as Partial<AskResponse>
  return {
    answer: typeof data.answer === 'string' ? data.answer : '',
    context_images: Array.isArray(data.context_images) ? data.context_images.filter((item): item is string => typeof item === 'string') : [],
    citations: data.citations,
    quality_warnings: data.quality_warnings,
    agent_trace: data.agent_trace,
  }
}

export async function ingestDefaultManual() {
  const response = await fetch(apiUrl('/api/ingest'), {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({}),
  })
  const payload = await readJson(response)
  if (!response.ok) {
    throw new Error(getErrorMessage(payload, 'Please check the backend logs.'))
  }
  if (payload && typeof payload === 'object' && 'message' in payload && typeof payload.message === 'string') {
    return payload.message
  }
  return 'Index updated.'
}

export function imageUrl(imagePath: string) {
  const normalized = imagePath.replaceAll('\\', '/')
  const segments = normalized.split('/').filter(Boolean)
  const legacyAbsolutePath = normalized.startsWith('/') || /^[A-Za-z]:\//.test(normalized)
  const publicSegments = legacyAbsolutePath ? segments.slice(-1) : segments
  return apiUrl(`/images/${publicSegments.map(encodeURIComponent).join('/')}`)
}
