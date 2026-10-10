export type Resource = { id: string; label: string }
export type DocumentType = Resource & { mappings: Resource[] }
export type Field = { field_id: string; label: string; data_type: string; bbox: number[] }
export type Writing = {
  model: string; handwriting: boolean; text_handwriting: boolean; intensity: number
  writer: number | null; bias: number; color: string; width: number; pressure: number
  dropout: number; text_policy: 'normalize' | 'strict'
}
export type Options = {
  document_type: string; mapping_id: string; fields: string[]; count: number
  seed: number; writing: Writing; filter_preset: string
}
export type Capabilities = {
  models: (Resource & { available: boolean; reason: string | null })[]
  filters: string[]; max_count: number; retention_hours: number
}
export type Preset = Resource & { writing: Writing | null; filter_preset: string; available: boolean; reason: string | null }
export type Job = {
  id: string; status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'expired'
  completed: number; total: number; error: string | null; expires_at: number; cancel_requested: boolean
}
export type Access = { id: string; token: string }
export type Artifact = { id: string; name: string; media_type: string; size: number }

export async function request<T>(path: string, init: RequestInit = {}, token?: string): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    ...init,
    headers: { ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}), ...init.headers },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.error?.message ?? 'Não foi possível concluir a solicitação')
  }
  return response.json() as Promise<T>
}

export async function fetchArtifact(access: Access, artifact: Artifact): Promise<string> {
  const response = await fetch(`/api/v1/jobs/${access.id}/artifacts/${artifact.id}?download=false`, {
    headers: { Authorization: `Bearer ${access.token}` },
  })
  if (!response.ok) throw new Error('Não foi possível abrir o arquivo. Verifique se o job expirou.')
  return URL.createObjectURL(await response.blob())
}
