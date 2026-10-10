import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

const job = { id: 'job1', status: 'succeeded', completed: 1, total: 1, expires_at: 1791734400, cancel_requested: false, error: null }
const files = [{ id: 'pdf', name: 'synthetic_0001.pdf', media_type: 'application/pdf', size: 100 }, { id: 'zip', name: 'dataset.zip', media_type: 'application/zip', size: 100 }]
let submitted: Record<string, unknown>[]
let requests: { path: string; init: RequestInit }[]
let failed: boolean
let currentStatus: string

beforeEach(() => {
  window.history.replaceState(null, '', '/')
  sessionStorage.clear()
  submitted = []; requests = []; failed = false; currentStatus = 'succeeded'
  vi.stubGlobal('fetch', vi.fn(async (input: string, init: RequestInit = {}) => {
    const path = input.replace('/api/v1', '')
    requests.push({ path, init })
    let data: unknown
    let status = 200
    if (path === '/document-types') data = [{ id: 'dnv', label: 'DNV', mappings: [{ id: 'default', label: 'Padrão' }] }]
    else if (path === '/capabilities') data = { models: [{ id: 'font', label: 'Fonte local', available: true }, { id: 'sjvasquez', label: 'sjvasquez', available: false }], filters: ['none', 'light'], max_count: 10, retention_hours: 24 }
    else if (path === '/writing-presets') data = []
    else if (path === '/health') data = { worker_ready: true }
    else if (path.includes('/mappings/')) data = { fields: [{ field_id: '1', label: 'Nome', data_type: 'person_name' }, { field_id: '2', label: 'Data', data_type: 'date' }] }
    else if (path === '/generation/validate') { data = failed ? { error: { code: 'quota', message: 'Limite de documentos atingido' } } : JSON.parse(String(init.body)); status = failed ? 422 : 200 }
    else if (path === '/jobs') { submitted.push(JSON.parse(String(init.body))); data = { ...job, status: 'queued', token: 'private-token' }; status = 202 }
    else if (path === '/jobs/job1') data = { ...job, status: currentStatus }
    else if (path === '/jobs/job1/results') data = { artifacts: files }
    else if (path === '/jobs/job1/cancel') { currentStatus = 'cancelled'; data = { ...job, status: 'cancelled' } }
    else if (path.includes('/artifacts/')) return new Response(new Blob(['PDF'], { type: 'application/pdf' }))
    else throw new Error(`Rota inesperada: ${path}`)
    return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } })
  }))
  vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:preview'), revokeObjectURL: vi.fn() })
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: vi.fn(async () => {}) } })
})

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('Geração de datasets', () => {
  it('seleciona campos, envia opções, acompanha, mostra PDF e baixa ZIP com autenticação', async () => {
    const user = userEvent.setup()
    const copy = vi.spyOn(navigator.clipboard, 'writeText')
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    render(<App />)
    await screen.findByLabelText('NomeNome de pessoa')
    await user.click(screen.getByRole('button', { name: 'Limpar' }))
    await user.click(screen.getByLabelText('NomeNome de pessoa'))
    fireEvent.change(screen.getByLabelText('Semente aleatória'), { target: { value: '123' } })
    await user.selectOptions(screen.getByLabelText('Filtros de digitalização'), 'light')
    await user.click(screen.getByRole('button', { name: 'Gerar dataset' }))
    await screen.findByRole('heading', { name: 'Dataset concluído' })
    expect(submitted[0]).toMatchObject({ fields: ['1'], seed: 123, filter_preset: 'light', count: 1 })
    expect(submitted[0]).not.toHaveProperty('output')
    await user.selectOptions(screen.getByLabelText('Documento gerado'), 'pdf')
    await waitFor(() => expect(screen.getByTitle('Prévia do PDF sintético').getAttribute('src')).toContain('blob:preview'))
    await user.click(screen.getByRole('button', { name: 'Baixar dataset completo (.zip)' }))
    await waitFor(() => expect(click).toHaveBeenCalled())
    await user.click(screen.getByRole('button', { name: 'Copiar link temporário' }))
    expect(copy).toHaveBeenCalledWith(expect.stringContaining('#job=job1&token=private-token'))
    await screen.findByText(/Link copiado/)
    expect(requests.filter(item => item.path.includes('/artifacts/')).every(item => (item.init.headers as Record<string, string>).Authorization === 'Bearer private-token')).toBe(true)
    expect(JSON.parse(sessionStorage.getItem('ocrsd-job')!)).toEqual({ id: 'job1', token: 'private-token' })
    expect((screen.getByRole('option', { name: 'sjvasquez — indisponível' }) as HTMLOptionElement).disabled).toBe(true)
  })

  it('mostra erro de validação e permite corrigir sem criar um job', async () => {
    failed = true
    const user = userEvent.setup()
    render(<App />)
    await screen.findByLabelText('NomeNome de pessoa')
    await user.click(screen.getByRole('button', { name: 'Gerar dataset' }))
    expect((await screen.findByRole('alert')).textContent).toContain('Limite de documentos atingido')
    expect(submitted).toHaveLength(0)
    expect((screen.getByRole('button', { name: 'Gerar dataset' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('acompanha o job mesmo quando o navegador bloqueia armazenamento da aba', async () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('Armazenamento bloqueado') })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByLabelText('NomeNome de pessoa')
    await user.click(screen.getByRole('button', { name: 'Gerar dataset' }))
    await screen.findByRole('heading', { name: 'Dataset concluído' })
    expect(submitted).toHaveLength(1)
  })

  it('recupera o acesso da aba e cancela um job em execução', async () => {
    currentStatus = 'running'
    sessionStorage.setItem('ocrsd-job', JSON.stringify({ id: 'job1', token: 'private-token' }))
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: 'Cancelar geração' }))
    await screen.findByRole('heading', { name: 'Job cancelado' })
    expect(requests.find(item => item.path.endsWith('/cancel'))?.init.method).toBe('POST')
    await user.click(screen.getByRole('button', { name: 'Esquecer acesso nesta aba' }))
    expect(sessionStorage.getItem('ocrsd-job')).toBeNull()
  })

  it('abre um link compartilhado sem manter a chave na URL', async () => {
    window.location.hash = '#job=job1&token=shared-token'
    render(<App />)
    await screen.findByRole('heading', { name: 'Dataset concluído' })
    expect(window.location.hash).toBe('')
    expect(JSON.parse(sessionStorage.getItem('ocrsd-job')!)).toEqual({ id: 'job1', token: 'shared-token' })
    const statusRequest = requests.find(item => item.path === '/jobs/job1')
    expect((statusRequest?.init.headers as Record<string, string>).Authorization).toBe('Bearer shared-token')
  })
})
