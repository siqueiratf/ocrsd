(() => {
  const legacy = new URLSearchParams(location.hash.slice(1))
  if (legacy.get('share') && legacy.get('token') && /^[A-Za-z0-9_-]{1,100}$/.test(legacy.get('share'))) {
    location.replace(`/compartilhado/${legacy.get('share')}#token=${encodeURIComponent(legacy.get('token'))}`)
    return
  }

  const modal = () => document.getElementById('modal')
  const stack = () => document.getElementById('avisos')
  const NOTICE_STATE = 'ocrsd-notices'
  const MAX_ERRORS = 3
  let memoryState = {}
  const readState = () => {
    try { return JSON.parse(localStorage.getItem(NOTICE_STATE) || '{}') } catch { return memoryState }
  }
  const writeState = state => {
    const limit = Date.now() - 86400000
    for (const [key, entry] of Object.entries(state)) if ((entry.first || 0) < limit) delete state[key]
    memoryState = state
    try { localStorage.setItem(NOTICE_STATE, JSON.stringify(state)) } catch {}
  }
  const remember = (key, change) => {
    const state = readState()
    state[key] = { first: Date.now(), ...state[key], ...change }
    writeState(state)
    return state[key]
  }
  const timers = new WeakMap()
  const lifetimeOf = element => Number(element.dataset.noticeLifetime ?? 15000)

  function closeNotice(element) {
    if (!element?.isConnected || element.classList.contains('is-leaving')) return
    clearInterval(timers.get(element))
    if (element.dataset.noticeKey) remember(element.dataset.noticeKey, { closed: true })
    element.classList.add('is-leaving')
    setTimeout(() => element.remove(), 180)
  }

  function armNotice(element) {
    if (timers.has(element)) return
    const key = element.dataset.noticeKey
    const entry = key ? readState()[key] : null
    if (entry?.closed) { element.remove(); return }
    const first = key ? (entry?.first || remember(key, {}).first) : Date.now()
    const lifetime = lifetimeOf(element)
    if (!lifetime) {
      timers.set(element, 0)
      const errors = [...stack().querySelectorAll('.toast[data-notice-lifetime="0"]:not(.is-leaving)')]
      errors.slice(0, Math.max(0, errors.length - MAX_ERRORS)).forEach(closeNotice)
      return
    }
    const timer = element.querySelector('[data-notice-timer]')
    const tick = () => {
      const remaining = Math.max(0, first + lifetime - Date.now())
      timer?.style.setProperty('stroke-dashoffset', String(100 - remaining / lifetime * 100))
      if (!remaining) closeNotice(element)
    }
    timers.set(element, setInterval(tick, 100))
    tick()
  }

  function showNotice(message, kind = 'sucesso') {
    const template = document.getElementById(kind === 'erro' ? 'modelo-aviso-erro' : 'modelo-aviso')
    const element = template?.content.firstElementChild?.cloneNode(true)
    if (!element || !stack()) return
    element.querySelector('[data-notice-title]').textContent = message
    stack().append(element)
  }
  window.OCRSDNotices = { show: showNotice }

  function datasetOpened(job, finished = true) {
    if (!job) return
    const key = `job:${job}:finished`
    stack()?.querySelectorAll(`[data-notice-key="${key}"]`).forEach(closeNotice)
    if (finished) remember(key, { closed: true })
  }

  let polling = null
  async function pollActivity() {
    const source = stack()?.dataset.activitySource
    if (!source || document.hidden) return
    let response
    try {
      response = await fetch(source, { credentials: 'same-origin', headers: { 'HX-Request': 'true' } })
    } catch { return }
    if (response.status === 401) { clearInterval(polling); return }
    if (!response.ok) return
    const template = document.createElement('template')
    template.innerHTML = await response.text()
    const incoming = [...template.content.querySelectorAll('[data-notice-key]')]
    const keys = new Set(incoming.map(item => item.dataset.noticeKey))
    stack().querySelectorAll('[data-notice-key$=":running"]').forEach(item => { if (!keys.has(item.dataset.noticeKey)) closeNotice(item) })
    const viewing = document.querySelector('.result-page[data-job-id]')?.dataset.jobId
    const state = readState()
    for (const item of incoming) {
      const key = item.dataset.noticeKey
      if (state[key]?.closed) continue
      if (viewing && item.dataset.noticeJob === viewing && key.endsWith(':finished')) { remember(key, { closed: true }); continue }
      const shown = stack().querySelector(`[data-notice-key="${CSS.escape(key)}"]`)
      if (shown) shown.querySelector('[data-notice-body]')?.replaceWith(item.querySelector('[data-notice-body]'))
      else stack().append(item)
    }
  }

  function post(url, values) {
    return fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'HX-Request': 'true', 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams(values),
    })
  }

  function adoptWorkspace() {
    if (!document.body.hasAttribute('data-sessao-nova')) return
    let token = null
    try { token = localStorage.getItem('ocrsd-workspace') } catch { return }
    if (!token) return
    post('/sessao', { token }).then(response => {
      if (response.ok || response.status === 401) {
        try { localStorage.removeItem('ocrsd-workspace') } catch {}
      }
      if (response.ok) location.reload()
    }).catch(() => {})
  }

  function openShared() {
    const token = new URLSearchParams(location.hash.slice(1)).get('token')
    if (token) history.replaceState(null, '', location.pathname + location.search)
    const pending = document.querySelector('[data-compartilhamento], [data-convite]')
    if (!pending) return
    const fail = async response => {
      pending.querySelector('[data-abrindo]').hidden = true
      pending.querySelector('[data-sem-token]').hidden = false
      if (response) document.getElementById('avisos').insertAdjacentHTML('beforeend', await response.text())
    }
    if (!token) return fail()
    post(pending.dataset.convite || `/compartilhado/${pending.dataset.compartilhamento}/acesso`, { token })
      .then(response => {
        if (!response.ok) return fail(response)
        if (pending.dataset.destino) location.replace(pending.dataset.destino)
        else location.reload()
      })
      .catch(() => fail())
  }

  function copy(button) {
    const input = document.querySelector(button.dataset.copiar)
    if (!input) return
    input.select()
    const label = button.querySelector('span') || button
    navigator.clipboard?.writeText(input.value)
      .then(() => { label.textContent = 'Copiado' })
      .catch(() => { label.textContent = 'Use Ctrl+C' })
  }

  document.addEventListener('htmx:afterSwap', event => {
    const dialog = modal()
    if (event.detail.target === dialog && dialog.childElementCount && !dialog.open) dialog.showModal()
  })

  document.addEventListener('htmx:load', event => {
    const auto = event.detail.elt.querySelector?.('[data-copiar-auto]')
    if (auto) {
      auto.select()
      navigator.clipboard?.writeText(auto.value).catch(() => {})
    }
  })

  document.addEventListener('fechar-modal', () => modal()?.close())

  document.addEventListener('toggle', event => {
    if (!event.target.matches?.('details.menu') || !event.target.open) return
    for (const menu of document.querySelectorAll('details.menu[open]')) if (menu !== event.target) menu.open = false
  }, true)

  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return
    for (const menu of document.querySelectorAll('details.menu[open]')) {
      menu.open = false
      menu.querySelector('summary').focus()
    }
  })

  document.addEventListener('click', event => {
    for (const menu of document.querySelectorAll('details.menu[open]')) {
      if (!menu.contains(event.target) || event.target.closest('.menu-list a, .menu-list button')) menu.open = false
    }
    const dialog = modal()
    if (event.target === dialog || event.target.closest('[data-fechar]')) dialog.close()
    const dismiss = event.target.closest('[data-fechar-aviso]')
    if (dismiss) closeNotice(dismiss.closest('.toast'))
    const notice = event.target.closest('.toast[data-notice-href]')
    if (notice && !dismiss && !event.target.closest('a')) {
      datasetOpened(notice.dataset.noticeJob)
      location.href = notice.dataset.noticeHref
    }
    const resultLink = event.target.closest('a[href^="/resultados/"]')
    const job = resultLink?.getAttribute('href').match(/^\/resultados\/([0-9a-f]{32})(?:[?#]|$)/)
    if (job) datasetOpened(job[1], !resultLink.closest('[data-notice-key$=":running"]'))
    const button = event.target.closest('[data-copiar]')
    if (button) copy(button)
  })

  document.addEventListener('input', event => {
    const range = event.target.closest?.('input[type=range][data-saida]')
    if (range) document.getElementById(range.dataset.saida).textContent = `${Math.round(range.value * 100)}%`
  })

  document.addEventListener('DOMContentLoaded', () => {
    modal()?.addEventListener('close', () => modal().replaceChildren())
    adoptWorkspace()
    openShared()
    const container = stack()
    if (container) {
      container.querySelectorAll('.toast').forEach(armNotice)
      new MutationObserver(records => records.forEach(record => record.addedNodes.forEach(node => {
        if (node.nodeType !== 1) return
        if (node.matches('.toast')) armNotice(node)
        node.querySelectorAll?.('.toast').forEach(armNotice)
      }))).observe(container, { childList: true, subtree: true })
      if (container.dataset.activitySource) {
        pollActivity()
        polling = setInterval(pollActivity, 3000)
      }
    }
  })
  document.addEventListener('visibilitychange', () => { if (!document.hidden) pollActivity() })
  document.addEventListener('ocrsd:dataset-opened', event => datasetOpened(event.detail.job, event.detail.finished !== false))
  window.addEventListener('storage', event => {
    if (event.key !== NOTICE_STATE) return
    const state = readState()
    stack()?.querySelectorAll('[data-notice-key]').forEach(item => {
      if (state[item.dataset.noticeKey]?.closed) closeNotice(item)
    })
  })
})()
