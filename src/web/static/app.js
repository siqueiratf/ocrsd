(() => {
  const legacy = new URLSearchParams(location.hash.slice(1))
  if (legacy.get('share') && legacy.get('token') && /^[A-Za-z0-9_-]{1,100}$/.test(legacy.get('share'))) {
    location.replace(`/compartilhado/${legacy.get('share')}#token=${encodeURIComponent(legacy.get('token'))}`)
    return
  }

  const modal = () => document.getElementById('modal')

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
    const pending = document.querySelector('[data-compartilhamento]')
    if (!pending) return
    const fail = async response => {
      pending.querySelector('[data-abrindo]').hidden = true
      pending.querySelector('[data-sem-token]').hidden = false
      if (response) document.getElementById('avisos').insertAdjacentHTML('beforeend', await response.text())
    }
    if (!token) return fail()
    post(`/compartilhado/${pending.dataset.compartilhamento}/acesso`, { token })
      .then(response => response.ok ? location.reload() : fail(response))
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
    for (const toast of event.detail.elt.querySelectorAll?.('.toast') ?? []) {
      setTimeout(() => toast.remove(), toast.classList.contains('toast-erro') ? 9000 : 4500)
    }
    if (event.detail.elt.classList?.contains('toast')) {
      const toast = event.detail.elt
      setTimeout(() => toast.remove(), toast.classList.contains('toast-erro') ? 9000 : 4500)
    }
    const auto = event.detail.elt.querySelector?.('[data-copiar-auto]')
    if (auto) {
      auto.select()
      navigator.clipboard?.writeText(auto.value).catch(() => {})
    }
  })

  document.addEventListener('fechar-modal', () => modal()?.close())

  document.addEventListener('click', event => {
    const dialog = modal()
    if (event.target === dialog || event.target.closest('[data-fechar]')) dialog.close()
    const dismiss = event.target.closest('[data-fechar-aviso]')
    if (dismiss) dismiss.closest('.toast').remove()
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
  })
})()
