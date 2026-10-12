(() => {
  const form = document.querySelector('[data-preset-form]')
  if (!form) return
  const isNew = form.dataset.modo === 'novo'
  const stateLabel = form.querySelector('[data-estado]')
  const stateText = form.querySelector('[data-estado-texto]')
  const stale = form.querySelector('[data-previa-desatualizada]')
  const ignored = new Set(['variacoes'])
  const previewIgnored = new Set(['variacoes', 'preset_name', 'description', 'id', 'retorno'])
  const serialize = skip => JSON.stringify([...new FormData(form)].filter(([key]) => !skip.has(key)))
  let baseline = serialize(ignored)
  let previewed = null
  let leaving = false

  const dirty = () => serialize(ignored) !== baseline

  function refresh() {
    const changed = dirty()
    stateLabel.dataset.estado = changed || isNew ? 'pendente' : 'salvo'
    stateText.textContent = isNew ? 'Ainda não salvo' : changed ? 'Alterações não salvas' : 'Sem alterações'
    stale.hidden = previewed === null || serialize(previewIgnored) === previewed
  }

  function paint() {
    for (const swatch of form.querySelectorAll('[data-cor-fundo]')) swatch.style.background = swatch.dataset.corFundo
    const mode = form.querySelector('[name="writer_mode"]:checked')
    form.querySelector('[data-escritor]')?.classList.toggle('is-muted', mode?.value === 'aleatorio')
  }

  form.addEventListener('input', event => {
    const input = event.target
    if (input.dataset.espelho) {
      const mirror = document.getElementById(input.dataset.espelho)
      if (mirror && input.value !== '') mirror.value = input.value
    }
    if (input.dataset.valor) document.getElementById(input.dataset.valor).textContent = input.value
    if (input.matches('[data-cor]')) form.querySelector('[data-cor-hex]').value = input.value
    if (input.matches('[data-cor-hex]') && /^#[0-9a-fA-F]{6}$/.test(input.value)) form.querySelector('[data-cor]').value = input.value.toLowerCase()
    if (input.matches('[data-escritor]')) {
      const fixed = form.querySelector('[name="writer_mode"][value="fixo"]')
      if (fixed) fixed.checked = true
    }
    if (input.matches('[data-nome-preset]')) form.querySelector('[data-preset-titulo]').textContent = input.value || 'Novo preset'
    paint()
    refresh()
  })

  form.addEventListener('change', () => { paint(); refresh() })

  form.addEventListener('click', event => {
    const swatch = event.target.closest('[data-amostra-cor]')
    if (!swatch) return
    const color = form.querySelector('[data-cor]')
    color.value = swatch.dataset.amostraCor
    color.dispatchEvent(new Event('input', { bubbles: true }))
  })

  document.addEventListener('click', event => {
    if (event.target.closest('button[data-sair]')) {
      leaving = true
      return
    }
    const link = event.target.closest('a[data-sair], a[data-usar]')
    if (!link || !dirty()) return
    if (window.confirm('Sair sem salvar as alterações deste preset?')) leaving = true
    else event.preventDefault()
  })

  document.body.addEventListener('htmx:afterRequest', event => {
    if (!event.detail.successful) leaving = false
  })

  document.body.addEventListener('htmx:afterSwap', event => {
    if (event.detail.target.id === 'playground') previewed = serialize(previewIgnored)
    paint()
    refresh()
  })

  document.body.addEventListener('preset-salvo', event => {
    baseline = serialize(ignored)
    const version = form.querySelector('[data-versao]')
    if (version) version.textContent = `versão ${event.detail.version}`
    refresh()
  })

  window.addEventListener('beforeunload', event => {
    if (dirty() && !leaving) event.preventDefault()
  })

  paint()
  refresh()
})()
