(() => {
  const storageKey = job => `ocrsd-result-view-${job}`
  const narrow = () => matchMedia('(max-width: 650px)').matches
  const readStorage = key => { try { return sessionStorage.getItem(key) } catch { return null } }
  const writeStorage = (key, value) => { try { sessionStorage.setItem(key, value) } catch {} }

  const formatDuration = seconds => {
    const total = Math.max(0, Math.floor(seconds))
    const hours = Math.floor(total / 3600)
    const minutes = Math.floor(total % 3600 / 60)
    const remainder = total % 60
    const pad = value => String(value).padStart(2, '0')
    return hours ? `${pad(hours)}:${pad(minutes)}:${pad(remainder)}` : `${pad(minutes)}:${pad(remainder)}`
  }
  const prepareDurations = root => root.querySelectorAll?.('[data-duration-elapsed]').forEach(cell => {
    cell.dataset.durationOrigin = String(Date.now() / 1000 - Number(cell.dataset.durationElapsed))
  })
  const refreshDurations = () => document.querySelectorAll('[data-duration-origin]').forEach(cell => {
    cell.textContent = formatDuration(Date.now() / 1000 - Number(cell.dataset.durationOrigin))
  })
  setInterval(refreshDurations, 1000)

  const tooltip = document.createElement('div')
  tooltip.className = 'result-status-tooltip'
  tooltip.setAttribute('role', 'tooltip')
  tooltip.id = 'result-status-tooltip'
  tooltip.hidden = true
  let tooltipTarget = null
  const placeTooltip = target => {
    if (tooltipTarget && tooltipTarget !== target) tooltipTarget.removeAttribute('aria-describedby')
    tooltipTarget = target
    if (!target) { tooltip.hidden = true; return }
    if (!tooltip.isConnected) document.body.append(tooltip)
    tooltip.textContent = target.dataset.statusTooltip
    tooltip.hidden = false
    target.setAttribute('aria-describedby', tooltip.id)
    const box = target.getBoundingClientRect()
    const width = tooltip.offsetWidth
    const height = tooltip.offsetHeight
    const below = box.bottom + 6 + height <= document.documentElement.clientHeight - 8
    tooltip.style.translate = `${Math.round(Math.max(8, Math.min(box.left + box.width / 2 - width / 2, document.documentElement.clientWidth - width - 8)))}px ${Math.round(below ? box.bottom + 6 : Math.max(8, box.top - height - 6))}px`
  }
  document.addEventListener('mouseover', event => { if (event.target.matches?.('[data-status-tooltip]')) placeTooltip(event.target) })
  document.addEventListener('mouseout', event => { if (event.target.matches?.('[data-status-tooltip]') && event.target === tooltipTarget) placeTooltip(null) })
  document.addEventListener('focusin', event => { if (event.target.matches?.('[data-status-tooltip]')) placeTooltip(event.target) })
  document.addEventListener('focusout', event => { if (event.target === tooltipTarget) placeTooltip(null) })
  document.addEventListener('scroll', () => placeTooltip(null), true)
  window.addEventListener('resize', () => placeTooltip(null))

  function positionMenu(menu) {
    const trigger = menu.querySelector('summary').getBoundingClientRect()
    const popup = menu.querySelector('.menu-list')
    const width = popup.offsetWidth
    const height = popup.offsetHeight
    popup.style.left = `${Math.max(8, Math.min(trigger.right - width, innerWidth - width - 8))}px`
    popup.style.top = `${trigger.bottom + height + 8 > innerHeight ? Math.max(8, trigger.top - height - 6) : trigger.bottom + 6}px`
  }
  document.addEventListener('toggle', event => {
    if (event.target.matches?.('.results-heading-combined .menu, .result-tree-row .menu') && event.target.open) positionMenu(event.target)
  }, true)

  let viewer = null
  function updateTreeToggle(workspace) {
    const button = workspace.querySelector('[data-tree-toggle]')
    if (!button) return
    const open = narrow() ? workspace.classList.contains('is-tree-open') : !workspace.classList.contains('is-tree-collapsed')
    const label = open ? 'Ocultar lista de documentos' : 'Mostrar lista de documentos'
    button.setAttribute('aria-expanded', String(open))
    button.setAttribute('aria-label', label)
    button.dataset.tooltip = label
    button.classList.toggle('is-active', open)
  }
  function filterTree(workspace) {
    const needle = workspace.querySelector('[data-document-search]')?.value.toLocaleLowerCase('pt-BR') || ''
    workspace.querySelectorAll('[data-document-label]').forEach(item => { item.hidden = Boolean(needle) && !item.dataset.documentLabel.includes(needle) })
  }
  function initialize() {
    const workspace = document.getElementById('result-workspace')
    const stage = workspace?.querySelector('[data-reader-stage]')
    if (!stage || workspace.dataset.initialized) return
    workspace.dataset.initialized = 'true'
    viewer?.destroy()
    const images = () => [...stage.querySelectorAll('img')]
    let view = { mode: 'page', scale: 1 }
    try { view = { ...view, ...JSON.parse(readStorage(storageKey(workspace.dataset.jobId)) || '{}') } } catch {}
    if (readStorage('ocrsd-result-tree') === 'collapsed' && !narrow()) workspace.classList.add('is-tree-collapsed')
    viewer = OCRSDPdfView.createViewer({
      stage, root: workspace, view, shortcuts: document,
      output: workspace.querySelector('[data-pdf-zoom]'),
      natural: () => {
        const first = images()[0]
        return first?.naturalWidth ? [first.naturalWidth, first.naturalHeight] : null
      },
      render: scale => images().forEach(image => { if (image.naturalWidth) image.style.width = `${Math.round(image.naturalWidth * scale)}px` }),
      onChange: state => writeStorage(storageKey(workspace.dataset.jobId), JSON.stringify(state)),
    })
    updateTreeToggle(workspace)
    workspace.querySelector('[data-tree-toggle]')?.addEventListener('click', () => {
      if (narrow()) workspace.classList.toggle('is-tree-open')
      else {
        workspace.classList.toggle('is-tree-collapsed')
        writeStorage('ocrsd-result-tree', workspace.classList.contains('is-tree-collapsed') ? 'collapsed' : 'open')
      }
      updateTreeToggle(workspace)
    })
    workspace.querySelector('[data-document-search]')?.addEventListener('input', () => filterTree(workspace))
    workspace.addEventListener('htmx:beforeCleanupElement', event => {
      if (event.target === workspace) viewer?.destroy()
    })
    workspace.querySelector('.result-tree-row.is-active')?.scrollIntoView({ block: 'nearest' })
    document.dispatchEvent(new CustomEvent('ocrsd:dataset-opened', { detail: { job: workspace.dataset.jobId, finished: !['queued', 'running'].includes(workspace.dataset.jobStatus) } }))
  }

  function refreshLiveState() {
    const workspace = document.getElementById('result-workspace')
    const live = document.getElementById('result-live-update')
    const count = Number(live?.dataset.documentCount)
    const current = Number(workspace?.dataset.documentNumber)
    if (!workspace || !count || !current) return
    const counter = workspace.querySelector('[data-reader-counter]')
    if (counter) counter.textContent = `${current} de ${count}`
    const next = workspace.querySelector('[data-document-step="next"]')
    if (next && current < count) {
      next.classList.remove('is-disabled')
      next.removeAttribute('aria-disabled')
      next.removeAttribute('tabindex')
    }
    workspace.querySelector('[data-document-search]')?.closest('label')?.removeAttribute('hidden')
    filterTree(workspace)
  }

  document.addEventListener('DOMContentLoaded', () => { initialize(); prepareDurations(document); refreshDurations() })
  document.addEventListener('htmx:load', event => { prepareDurations(event.detail.elt.parentNode || event.detail.elt); refreshDurations() })
  document.addEventListener('htmx:afterSettle', event => {
    if (event.detail.target.id === 'result-workspace' || event.detail.elt?.id === 'result-workspace') initialize()
    if (event.detail.target.id === 'result-live-update') refreshLiveState()
    const page = document.querySelector('.result-page[data-back-url]')
    page?.querySelector('[data-back-link]')?.setAttribute('href', page.dataset.backUrl)
  })

  const editing = () => document.querySelector('[data-result-name-cell][data-editing]')
  document.addEventListener('htmx:beforeRequest', event => {
    if (event.detail.elt.matches?.('[data-results-polling]') && (editing() || document.querySelector('#tabela-jobs details.menu[open]') || tooltipTarget)) event.preventDefault()
  })

  function startRename(cell) {
    if (!cell || cell.dataset.editing) return
    const other = editing()
    if (other) cancelRename(other)
    cell.dataset.editing = 'true'
    const form = cell.querySelector('[data-rename-form]')
    const input = form.querySelector('input')
    input.value = input.defaultValue
    cell.querySelector('[data-result-name]').hidden = true
    cell.querySelector('[data-rename-start]').hidden = true
    form.hidden = false
    input.focus()
    input.select()
  }
  function cancelRename(cell, restoreFocus = false) {
    if (!cell?.dataset.editing || cell.dataset.saving) return
    delete cell.dataset.editing
    const form = cell.querySelector('[data-rename-form]')
    const input = form.querySelector('input')
    form.hidden = true
    input.value = input.defaultValue
    input.setCustomValidity('')
    cell.querySelector('[data-result-name]').hidden = false
    cell.querySelector('[data-rename-start]').hidden = false
    if (restoreFocus) cell.querySelector('[data-result-name]').focus()
  }
  let swallowClick = false
  document.addEventListener('pointerdown', event => {
    const cell = editing()
    if (!cell || cell.contains(event.target)) return
    cancelRename(cell)
    swallowClick = true
  }, true)
  document.addEventListener('pointerup', () => { if (swallowClick) setTimeout(() => { swallowClick = false }, 0) }, true)
  document.addEventListener('click', event => {
    if (!swallowClick) return
    swallowClick = false
    event.preventDefault()
    event.stopImmediatePropagation()
  }, true)
  document.addEventListener('focusout', event => {
    const cell = event.target.closest?.('[data-result-name-cell][data-editing]')
    if (cell && event.target.matches('[data-rename-form] input') && !cell.contains(event.relatedTarget)) cancelRename(cell)
  })
  document.addEventListener('dblclick', event => startRename(event.target.closest?.('[data-result-name]')?.closest('[data-result-name-cell]')))
  document.addEventListener('submit', event => {
    const form = event.target.closest('[data-rename-form]')
    if (!form) return
    const cell = form.closest('[data-result-name-cell]')
    const input = form.querySelector('input')
    const value = input.value.trim()
    if (cell.dataset.saving) { event.preventDefault(); return }
    if (!value) {
      event.preventDefault()
      input.setCustomValidity('Informe um nome para o dataset')
      input.reportValidity()
      return
    }
    if (value === input.defaultValue.trim()) {
      event.preventDefault()
      cancelRename(cell, true)
      return
    }
    input.value = value
    cell.dataset.saving = 'true'
    input.readOnly = true
  }, true)
  document.addEventListener('htmx:afterSwap', event => {
    if (event.detail.target.matches?.('[data-result-name-cell]') || event.detail.elt.matches?.('[data-result-name-cell]')) {
      const job = event.detail.requestConfig?.elt?.closest?.('[data-job-id]')?.dataset.jobId
      document.querySelector(`[data-result-name-cell][data-job-id="${job}"] [data-result-name]`)?.focus()
    }
  })
  document.addEventListener('htmx:responseError', event => {
    const cell = event.detail.elt.closest?.('[data-result-name-cell]')
    if (!cell || !event.detail.elt.matches('[data-rename-form]')) return
    delete cell.dataset.saving
    cell.querySelector('[data-rename-form] input').readOnly = false
    cancelRename(cell, true)
  })
  document.addEventListener('htmx:sendError', event => {
    const cell = event.detail.elt.closest?.('[data-result-name-cell]')
    if (!cell) return
    delete cell.dataset.saving
    cell.querySelector('[data-rename-form] input').readOnly = false
    cancelRename(cell, true)
  })
  document.addEventListener('input', event => { if (event.target.matches('[data-rename-form] input')) event.target.setCustomValidity('') })
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && tooltipTarget) placeTooltip(null)
    if (event.target.matches?.('[data-result-name]') && ['Enter', 'F2'].includes(event.key)) {
      event.preventDefault()
      startRename(event.target.closest('[data-result-name-cell]'))
    }
    if (event.key === 'Escape' && event.target.closest?.('[data-rename-form]')) {
      event.preventDefault()
      event.stopPropagation()
      cancelRename(event.target.closest('[data-result-name-cell]'), true)
    }
  })

  const submitFilters = () => document.querySelector('#filtros input[name="status"]')?.dispatchEvent(new Event('change', { bubbles: true }))
  document.addEventListener('change', event => {
    if (!event.target.matches('[data-model-choice]')) return
    document.querySelector('#filtros input[name="modelo"]').value = event.target.value
    submitFilters()
  }, true)
  document.addEventListener('click', event => {
    startRename(event.target.closest('[data-rename-start]')?.closest('[data-result-name-cell]'))
    const sort = event.target.closest('[data-sort]')
    if (sort) {
      const input = document.querySelector('#filtros input[name="ordem"]')
      const current = input.value
      input.value = current === `${sort.dataset.sort}_asc` ? `${sort.dataset.sort}_desc` : `${sort.dataset.sort}_asc`
      if (sort.dataset.sort === 'data' && current === 'recentes') input.value = 'data_asc'
      submitFilters()
    }
    if (event.target.closest('[data-status-clear]')) {
      document.querySelectorAll('[data-status-choice]').forEach(box => { box.checked = false })
      document.querySelector('#filtros input[name="status"]').value = ''
      submitFilters()
    }
    if (event.target.closest('[data-status-apply]')) {
      document.querySelector('#filtros input[name="status"]').value = [...document.querySelectorAll('[data-status-choice]:checked')].map(item => item.value).join(',')
      submitFilters()
    }
  })
  document.addEventListener('scroll', event => {
    document.querySelectorAll('.results-heading-combined .menu[open], .result-tree-row .menu[open]').forEach(menu => {
      if (!menu.contains(event.target)) menu.open = false
    })
  }, true)
  document.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest?.('input, textarea, select, [contenteditable], dialog')) return
    const step = event.key === 'ArrowLeft' ? 'previous' : event.key === 'ArrowRight' ? 'next' : ''
    const link = step && document.querySelector(`#result-workspace [data-document-step="${step}"]`)
    if (link && !link.hasAttribute('aria-disabled')) { event.preventDefault(); link.click() }
  })
  document.addEventListener('submit', event => {
    const form = event.target.closest('[data-export-form]')
    if (!form) return
    if (!form.querySelector('[name="pdfs"]').checked && !form.querySelector('[name="jsons"]').checked) {
      event.preventDefault()
      form.querySelector('[name="pdfs"]').setCustomValidity('Selecione ao menos um formato')
      form.querySelector('[name="pdfs"]').reportValidity()
      return
    }
    setTimeout(() => form.closest('dialog')?.close(), 0)
  })
  document.addEventListener('change', event => {
    if (event.target.matches('[data-export-form] input[type="checkbox"]')) event.target.closest('form').querySelector('[name="pdfs"]').setCustomValidity('')
  })
})()
