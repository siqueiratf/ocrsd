import { FormEvent, useEffect, useState } from 'react'
import { Access, Artifact, Capabilities, DocumentType, Field, Job, Options, Preset, Writing, fetchArtifact, request } from './api'

const defaultWriting: Writing = { model: 'font', handwriting: true, text_handwriting: true, intensity: 0.65, writer: null, bias: 0.75, color: '#193787', width: 1.5, pressure: 0.25, dropout: 0, text_policy: 'normalize' }
const statuses: Record<Job['status'], string> = { queued: 'Na fila', running: 'Gerando documentos', succeeded: 'Dataset concluído', failed: 'Geração interrompida', cancelled: 'Job cancelado', expired: 'Resultados expirados' }
const fieldTypes: Record<string, string> = { text: 'Texto', integer: 'Inteiro', float: 'Decimal', date: 'Data', datetime: 'Data e hora', time: 'Hora', person_name: 'Nome de pessoa', checkbox: 'Marcação', digit_sequence: 'Sequência numérica', text_sequence: 'Sequência de texto' }
const filters: Record<string, string> = { none: 'Sem filtros', light: 'Leve', medium: 'Moderado', strong: 'Intenso' }

function savedAccess(): Access | null {
  try {
    const value = JSON.parse(sessionStorage.getItem('ocrsd-job') ?? 'null')
    return typeof value?.id === 'string' && typeof value?.token === 'string' ? value : null
  } catch { return null }
}

function rememberAccess(access: Access | null) {
  try {
    if (access) sessionStorage.setItem('ocrsd-job', JSON.stringify(access))
    else sessionStorage.removeItem('ocrsd-job')
  } catch { return }
}

function initialAccess(): Access | null {
  try {
    const parameters = new URLSearchParams(window.location.hash.slice(1))
    const id = parameters.get('job')
    const token = parameters.get('token')
    if (id && token) {
      const access = { id, token }
      rememberAccess(access)
      window.history.replaceState(null, '', window.location.pathname + window.location.search)
      return access
    }
  } catch { return savedAccess() }
  return savedAccess()
}

async function copyText(value: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value)
    return
  }
  const input = document.createElement('textarea')
  input.value = value
  input.style.position = 'fixed'
  input.style.opacity = '0'
  document.body.append(input)
  input.select()
  const copied = document.execCommand('copy')
  input.remove()
  if (!copied) throw new Error('Cópia indisponível neste navegador')
}

export default function App() {
  const [types, setTypes] = useState<DocumentType[]>([])
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [presets, setPresets] = useState<Preset[]>([])
  const [fields, setFields] = useState<Field[]>([])
  const [options, setOptions] = useState<Options>({ document_type: '', mapping_id: '', fields: [], count: 1, seed: 42, writing: defaultWriting, filter_preset: 'none' })
  const [access, setAccess] = useState<Access | null>(initialAccess)
  const [job, setJob] = useState<Job | null>(null)
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [preview, setPreview] = useState('')
  const [error, setError] = useState('')
  const [shareStatus, setShareStatus] = useState('')
  const [busy, setBusy] = useState(false)
  const [mappingBusy, setMappingBusy] = useState(false)
  const [workerReady, setWorkerReady] = useState(true)
  const active = Boolean(access && (!job || ['queued', 'running'].includes(job.status)))

  useEffect(() => {
    let live = true
    Promise.all([request<DocumentType[]>('/document-types'), request<Capabilities>('/capabilities'), request<Preset[]>('/writing-presets'), request<{ worker_ready: boolean }>('/health')])
      .then(([documents, caps, styles, health]) => {
        if (!live) return
        setTypes(documents); setCapabilities(caps); setPresets(styles); setWorkerReady(health.worker_ready)
        if (documents[0]) setOptions(previous => ({ ...previous, document_type: documents[0].id, mapping_id: documents[0].mappings[0]?.id ?? '' }))
      }).catch(reason => { if (live) setError(String(reason.message)) })
    return () => { live = false }
  }, [])

  useEffect(() => {
    if (!options.document_type || !options.mapping_id) return
    let live = true
    setMappingBusy(true); setFields([])
    request<{ fields: Field[] }>(`/document-types/${options.document_type}/mappings/${options.mapping_id}`)
      .then(data => {
        if (!live) return
        setFields(data.fields); setOptions(previous => ({ ...previous, fields: data.fields.map(field => field.field_id) }))
      }).catch(reason => { if (live) setError(reason.message) })
      .finally(() => { if (live) setMappingBusy(false) })
    return () => { live = false }
  }, [options.document_type, options.mapping_id])

  useEffect(() => {
    if (!access) return
    let live = true
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const current = await request<Job>(`/jobs/${access.id}`, {}, access.token)
        if (!live) return
        setJob(current)
        if (current.status === 'succeeded') {
          const result = await request<{ artifacts: Artifact[] }>(`/jobs/${access.id}/results`, {}, access.token)
          if (live) setArtifacts(result.artifacts)
        } else if (['queued', 'running'].includes(current.status)) timer = setTimeout(poll, 1500)
      } catch (reason) {
        if (live) { setError((reason as Error).message); timer = setTimeout(poll, 5000) }
      }
    }
    void poll()
    return () => { live = false; clearTimeout(timer) }
  }, [access])

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  function writing(update: Partial<Writing>) { setOptions(previous => ({ ...previous, writing: { ...previous.writing, ...update } })) }

  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError('')
    try {
      await request<Options>('/generation/validate', { method: 'POST', body: JSON.stringify(options) })
      const created = await request<Job & { token: string }>('/jobs', { method: 'POST', body: JSON.stringify(options) })
      const key = { id: created.id, token: created.token }
      rememberAccess(key)
      setArtifacts([]); setPreview(''); setJob(created); setAccess(key)
    } catch (reason) { setError((reason as Error).message) } finally { setBusy(false) }
  }

  async function openArtifact(artifact: Artifact, download: boolean) {
    if (!access) return
    try {
      const url = await fetchArtifact(access, artifact)
      if (download) {
        const link = document.createElement('a'); link.href = url; link.download = artifact.name; link.click()
        setTimeout(() => URL.revokeObjectURL(url), 30000)
      } else setPreview(url)
    } catch (reason) { setError((reason as Error).message) }
  }

  async function cancel() {
    if (!access) return
    try { setJob(await request<Job>(`/jobs/${access.id}/cancel`, { method: 'POST', body: '{}' }, access.token)) }
    catch (reason) { setError((reason as Error).message) }
  }

  async function share() {
    if (!access) return
    const link = `${window.location.origin}${window.location.pathname}${window.location.search}#job=${encodeURIComponent(access.id)}&token=${encodeURIComponent(access.token)}`
    try {
      await copyText(link)
      setShareStatus('Link copiado. Quem receber poderá acessar os resultados até a expiração do job.')
    } catch (reason) { setError((reason as Error).message) }
  }

  const currentType = types.find(type => type.id === options.document_type)
  const archive = artifacts.find(artifact => artifact.media_type === 'application/zip')

  return <>
    <header className="masthead"><a href="/" className="brand">OCRSD<span>Laboratório documental</span></a><span className="badge">Pesquisa · Educação · Desenvolvimento</span></header>
    <main>
      <section className="intro"><p className="eyebrow">GERAÇÃO DE DATASETS</p><h1>Documentos sintéticos.<br /><span>Possibilidades reais de pesquisa.</span></h1><p>Configure documentos de registro civil para testar reconhecimento de texto e estudar processamento documental.</p></section>
      <div className="notice"><strong>Uso exclusivamente educacional e de pesquisa.</strong> Todos os documentos gerados recebem a marca “SINTÉTICO — SEM VALIDADE LEGAL”.</div>
      {error && <div role="alert" className="error">{error}<button type="button" onClick={() => setError('')} aria-label="Fechar mensagem">×</button></div>}
      {!workerReady && <p role="status" className="notice">O worker está indisponível. Jobs enviados ficarão na fila até a retomada do serviço.</p>}
      <div className="workspace">
        <form onSubmit={submit} className="configuration">
          <fieldset disabled={busy || active}>
            <section className="card"><h2><span>01</span> Documento e campos</h2>
              <label>Tipo de documento<select value={options.document_type} onChange={event => {
                const type = types.find(item => item.id === event.target.value)
                setOptions(previous => ({ ...previous, document_type: event.target.value, mapping_id: type?.mappings[0]?.id ?? '', fields: [] }))
              }}>{types.map(type => <option key={type.id} value={type.id}>{type.label}</option>)}</select></label>
              <label>Mapeamento<select value={options.mapping_id} onChange={event => setOptions(previous => ({ ...previous, mapping_id: event.target.value, fields: [] }))}>{currentType?.mappings.map(mapping => <option key={mapping.id} value={mapping.id}>{mapping.label}</option>)}</select></label>
              <div className="field-toolbar"><strong>Campos ({options.fields.length})</strong><button type="button" onClick={() => setOptions(previous => ({ ...previous, fields: fields.map(field => field.field_id) }))}>Todos</button><button type="button" onClick={() => setOptions(previous => ({ ...previous, fields: [] }))}>Limpar</button></div>
              <div className="field-list" aria-label="Campos disponíveis" aria-busy={mappingBusy}>{fields.map(field => <label key={field.field_id} className="checkbox"><input type="checkbox" checked={options.fields.includes(field.field_id)} onChange={event => setOptions(previous => ({ ...previous, fields: event.target.checked ? [...previous.fields, field.field_id] : previous.fields.filter(id => id !== field.field_id) }))} /><span>{field.label}<small>{fieldTypes[field.data_type] ?? field.data_type}</small></span></label>)}</div>
            </section>
            <section className="card"><h2><span>02</span> Escrita e aparência</h2>
              <label>Preset de escrita<select defaultValue="" onChange={event => { const preset = presets.find(item => item.id === event.target.value); if (preset?.writing) setOptions(previous => ({ ...previous, writing: preset.writing!, filter_preset: preset.filter_preset })) }}><option value="">Configuração personalizada</option>{presets.map(preset => <option key={preset.id} value={preset.id} disabled={!preset.available}>{preset.label}{!preset.available ? ` — ${preset.reason}` : ''}</option>)}</select></label>
              <label>Modelo de escrita<select value={options.writing.model} onChange={event => writing({ model: event.target.value, writer: null, bias: event.target.value === 'sjvasquez' ? 0.75 : 2.1 })}>{capabilities?.models.map(model => <option key={model.id} value={model.id} disabled={!model.available}>{model.label}{!model.available ? ' — indisponível' : ''}</option>)}</select></label>
              <div className="pair"><label className="checkbox"><input type="checkbox" checked={options.writing.handwriting} onChange={event => writing({ handwriting: event.target.checked })} />Números manuscritos</label><label className="checkbox"><input type="checkbox" checked={options.writing.text_handwriting} onChange={event => writing({ text_handwriting: event.target.checked })} />Texto manuscrito</label></div>
              <label>Variação da escrita: {Math.round(options.writing.intensity * 100)}%<input type="range" min="0" max="1" step="0.05" value={options.writing.intensity} onChange={event => writing({ intensity: Number(event.target.value) })} /></label>
              {options.writing.model !== 'font' && <details><summary>Ajustes da caneta neural</summary><div className="pair"><label>Estilo (vazio = aleatório)<input type="number" min="0" max={options.writing.model === 'sjvasquez' ? 12 : 511} value={options.writing.writer ?? ''} onChange={event => writing({ writer: event.target.value === '' ? null : Number(event.target.value) })} /></label><label>Regularidade<input type="number" min="0" max="3" step="0.05" value={options.writing.bias} onChange={event => writing({ bias: Number(event.target.value) })} /></label><label>Cor<input type="color" value={options.writing.color} onChange={event => writing({ color: event.target.value })} /></label><label>Espessura<input type="number" min="0.1" max="40" step="0.1" value={options.writing.width} onChange={event => writing({ width: Number(event.target.value) })} /></label><label>Pressão<input type="number" min="0" max="1" step="0.05" value={options.writing.pressure} onChange={event => writing({ pressure: Number(event.target.value) })} /></label><label>Falha de tinta<input type="number" min="0" max="1" step="0.05" value={options.writing.dropout} onChange={event => writing({ dropout: Number(event.target.value) })} /></label></div><label>Caracteres<select value={options.writing.text_policy} onChange={event => writing({ text_policy: event.target.value as Writing['text_policy'] })}><option value="normalize">Normalizar acentos e símbolos</option><option value="strict">Exigir alfabeto compatível</option></select></label></details>}
              <label>Filtros de digitalização<select value={options.filter_preset} onChange={event => setOptions(previous => ({ ...previous, filter_preset: event.target.value }))}>{capabilities?.filters.map(filter => <option key={filter} value={filter}>{filters[filter]}</option>)}</select></label>
            </section>
            <section className="card"><h2><span>03</span> Tamanho do dataset</h2><div className="pair"><label>Quantidade<input required type="number" min="1" max={capabilities?.max_count ?? 10} value={options.count} onChange={event => setOptions(previous => ({ ...previous, count: Number(event.target.value) }))} /></label><label>Semente aleatória<input required type="number" min="0" max="4294967295" value={options.seed} onChange={event => setOptions(previous => ({ ...previous, seed: Number(event.target.value) }))} /></label></div><p className="hint">A mesma semente e configuração permitem repetir a geração. Os resultados ficam disponíveis por {capabilities?.retention_hours ?? 24} horas após a conclusão.</p><button className="primary" disabled={!capabilities || mappingBusy || !options.fields.length} type="submit">{busy ? 'Enviando…' : active ? 'Geração em andamento' : 'Gerar dataset'}</button></section>
          </fieldset>
        </form>
        <aside className="results"><section className="card progress-card"><p className="eyebrow">ÁREA DE TRABALHO</p><h2>{job ? statuses[job.status] : 'Seu próximo dataset começa aqui'}</h2>{job ? <><p role="status" aria-live="polite">{job.completed} de {job.total} documentos gerados{job.status === 'running' && job.completed === job.total ? ' · preparando arquivos' : ''}</p><progress aria-label="Progresso da geração" value={job.completed} max={job.total} /><p className="job-id">Job: {job.id}</p>{job.error && <p role="alert">{job.error}</p>}{active && <button type="button" disabled={job.cancel_requested} onClick={() => void cancel()}>{job.cancel_requested ? 'Cancelamento solicitado' : 'Cancelar geração'}</button>}{archive && <><button className="primary" type="button" onClick={() => void openArtifact(archive, true)}>Baixar dataset completo (.zip)</button><button type="button" onClick={() => void share()}>Copiar link temporário</button><p className="hint">PDFs, valores esperados em JSON e manifesto. Disponível até {new Date(job.expires_at * 1000).toLocaleString('pt-BR')}.</p>{shareStatus && <p role="status" className="hint">{shareStatus}</p>}</>}</> : <p>Selecione os campos e personalize a escrita. Acompanhe a geração e inspecione os documentos antes de baixar.</p>}
          {access && <button type="button" className="quiet" onClick={() => { rememberAccess(null); setAccess(null); setJob(null); setArtifacts([]); setPreview('') }}>Esquecer acesso nesta aba</button>}
        </section>
        <section className="card preview-card"><h2>Prévia dos documentos</h2>{artifacts.length ? <><label>Documento gerado<select defaultValue="" onChange={event => { const item = artifacts.find(artifact => artifact.id === event.target.value); if (item) void openArtifact(item, false) }}><option value="">Selecione um PDF</option>{artifacts.filter(item => item.media_type === 'application/pdf').map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>{preview ? <iframe title="Prévia do PDF sintético" src={`${preview}#toolbar=0&navpanes=0&view=FitH`} /> : <div className="empty-preview">Escolha um documento para visualizar.</div>}<div className="artifact-list">{artifacts.filter(item => item.media_type === 'application/json').map(item => <button type="button" key={item.id} onClick={() => void openArtifact(item, true)}>{item.name} ↓</button>)}</div></> : <div className="empty-preview"><div className="document-symbol" aria-hidden="true">Aa<br /><span>────<br />────<br />───</span></div><p>Os PDFs aparecerão após a conclusão do job.</p></div>}</section>
        </aside>
      </div>
    </main><footer>OCRSD · Documentos artificiais para pesquisa responsável. Sem upload de documentos pessoais nesta interface.</footer>
  </>
}
