export interface PredictionResult {
  rank: number
  class_name: string
  confidence: number
}

export interface PredictionResponse {
  top_predictions: PredictionResult[]
  predicted_class: string
  confidence: number
  processing_time_ms: number
}

export interface BatikInfo {
  class_name: string
  origin: string
  region: string
  description: string
  characteristics: string[]
}

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

export async function predictBatik(file: File): Promise<PredictionResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE_URL}/predict`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<PredictionResponse>
}

export async function getBatikInfo(className: string): Promise<BatikInfo> {
  const res = await fetch(`${BASE_URL}/batik/${encodeURIComponent(className)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<BatikInfo>
}

export async function getAllClasses(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/classes`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<string[]>
}
