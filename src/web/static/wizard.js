(() => {
  const positionOverlays = () => document.querySelectorAll('.wizard-box[data-box-x]').forEach(box => {
    box.style.left = `${box.dataset.boxX}%`
    box.style.top = `${box.dataset.boxY}%`
    box.style.width = `${box.dataset.boxWidth}%`
    box.style.height = `${box.dataset.boxHeight}%`
  })
  positionOverlays()

  const preview = document.querySelector('.wizard-preview:has([data-pdf-stage])')
  const stage = preview?.querySelector('[data-pdf-stage]')
  let viewer = null
  if (preview) {
    const page = () => stage.querySelector('[data-pdf-page]')
    viewer = OCRSDPdfView.createViewer({
      stage, root: preview, shortcuts: preview, view: { mode: 'width', scale: 1 },
      output: preview.querySelector('[data-pdf-zoom]'),
      natural: () => {
        const image = page()?.querySelector('img')
        return image?.naturalWidth ? [image.naturalWidth, image.naturalHeight] : null
      },
      render: scale => {
        const element = page()
        const image = element?.querySelector('img')
        if (element && image?.naturalWidth) element.style.width = `${Math.round(image.naturalWidth * scale)}px`
      },
    })
  }

  const overlaysFor = id => [...document.querySelectorAll('[data-overlay-field]')].filter(box => box.dataset.overlayField === id)
  function reveal(id, pulse = false) {
    const boxes = overlaysFor(id)
    if (!stage || !boxes.length) return
    const view = stage.getBoundingClientRect()
    const rects = boxes.map(box => box.getBoundingClientRect())
    const left = Math.min(...rects.map(rect => rect.left))
    const top = Math.min(...rects.map(rect => rect.top))
    const right = Math.max(...rects.map(rect => rect.right))
    const bottom = Math.max(...rects.map(rect => rect.bottom))
    const margin = 24
    const visible = left >= view.left + margin && right <= view.right - margin && top >= view.top + margin && bottom <= view.bottom - margin
    if (!visible) {
      stage.scrollTo({
        left: stage.scrollLeft + (left + right) / 2 - (view.left + view.right) / 2,
        top: stage.scrollTop + (top + bottom) / 2 - (view.top + view.bottom) / 2,
        behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
      })
    }
    if (pulse) {
      boxes.forEach(box => {
        box.classList.remove('is-located')
        void box.offsetWidth
        box.classList.add('is-located')
      })
    }
  }
  const hover = (id, active) => overlaysFor(id).forEach(box => box.classList.toggle('is-hovered', active))

  const form = document.querySelector('[data-field-form]')
  if (form) {
    const boxes = [...form.querySelectorAll('input[name="fields"]')]
    const entries = [...form.querySelectorAll('[data-field-id]')]
    const list = form.querySelector('.wizard-field-scroll')
    const count = document.getElementById('wizard-field-count')
    const search = form.querySelector('[data-field-search]')
    const selectedOnly = form.querySelector('[data-selected-only]')
    const normalize = value => value.normalize('NFD').replace(/[̀-ͯ]/g, '').toLocaleLowerCase('pt-BR')
    const filter = () => {
      const needle = normalize(search.value.trim())
      entries.forEach(entry => {
        const checked = entry.querySelector('input').checked
        entry.hidden = Boolean((selectedOnly.checked && !checked) || (needle && !normalize(entry.dataset.fieldSearchText).includes(needle)))
      })
      form.querySelectorAll('[data-field-section]').forEach(section => {
        const visible = [...section.querySelectorAll('[data-field-id]')].some(entry => !entry.hidden)
        section.hidden = !visible
        if (needle && visible) section.open = true
      })
    }
    const sync = () => {
      const selected = new Set(boxes.filter(box => box.checked).map(box => box.value))
      count.textContent = `${selected.size} de ${boxes.length} campos selecionados`
      document.querySelectorAll('[data-overlay-field]').forEach(box => {
        const active = selected.has(box.dataset.overlayField)
        box.classList.toggle('is-selected', active)
        box.setAttribute('aria-pressed', String(active))
      })
      entries.forEach(entry => entry.classList.toggle('is-selected', selected.has(entry.dataset.fieldId)))
      form.querySelectorAll('[data-field-section]').forEach(section => {
        const group = [...section.querySelectorAll('input[name="fields"]')]
        const chosen = group.filter(box => box.checked).length
        const toggle = section.querySelector('[data-field-group-checkbox]')
        toggle.checked = chosen === group.length
        toggle.indeterminate = chosen > 0 && chosen < group.length
        section.querySelector('[data-group-count]').textContent = `${chosen} de ${group.length}`
      })
      filter()
    }
    const showEntry = entry => {
      entry.closest('details').open = true
      const box = entry.getBoundingClientRect()
      const view = list.getBoundingClientRect()
      if (box.top < view.top || box.bottom > view.bottom) list.scrollBy({ top: (box.top + box.bottom) / 2 - (view.top + view.bottom) / 2 })
    }
    form.addEventListener('change', event => {
      if (event.target.matches('input[name="fields"]')) {
        sync()
        if (event.target.checked) reveal(event.target.value, true)
      }
      if (event.target.matches('[data-selected-only]')) filter()
      if (event.target.matches('[data-field-group-checkbox]')) {
        event.target.closest('[data-field-section]').querySelectorAll('input[name="fields"]').forEach(box => { box.checked = event.target.checked })
        sync()
      }
    })
    search.addEventListener('input', filter)
    form.addEventListener('click', event => {
      const action = event.target.closest('[data-field-action]')
      if (!action) return
      boxes.forEach(box => { box.checked = action.dataset.fieldAction === 'all' })
      sync()
    })
    let dwell = null
    entries.forEach(entry => {
      entry.addEventListener('mouseenter', () => {
        hover(entry.dataset.fieldId, true)
        clearTimeout(dwell)
        dwell = setTimeout(() => reveal(entry.dataset.fieldId), 350)
      })
      entry.addEventListener('mouseleave', () => {
        hover(entry.dataset.fieldId, false)
        clearTimeout(dwell)
      })
      const input = entry.querySelector('input')
      input.addEventListener('focus', () => { hover(entry.dataset.fieldId, true); reveal(entry.dataset.fieldId) })
      input.addEventListener('blur', () => hover(entry.dataset.fieldId, false))
    })
    preview?.addEventListener('click', event => {
      const overlay = event.target.closest('[data-overlay-field]')
      if (!overlay) return
      const box = boxes.find(item => item.value === overlay.dataset.overlayField)
      if (!box) return
      box.checked = !box.checked
      const entry = box.closest('[data-field-id]')
      if (selectedOnly.checked && !box.checked) selectedOnly.checked = false
      if (search.value && !normalize(entry.dataset.fieldSearchText).includes(normalize(search.value))) search.value = ''
      sync()
      showEntry(entry)
      entry.classList.remove('is-located')
      void entry.offsetWidth
      entry.classList.add('is-located')
    })
    preview?.addEventListener('mouseover', event => {
      const overlay = event.target.closest('[data-overlay-field]')
      if (overlay) entries.find(entry => entry.dataset.fieldId === overlay.dataset.overlayField)?.classList.add('is-hovered')
    })
    preview?.addEventListener('mouseout', event => {
      const overlay = event.target.closest('[data-overlay-field]')
      if (overlay) entries.find(entry => entry.dataset.fieldId === overlay.dataset.overlayField)?.classList.remove('is-hovered')
    })
    window.addEventListener('pageshow', sync)
    sync()
  }

  document.addEventListener('htmx:afterSwap', event => {
    if (event.detail.target.id !== 'wizard-mapping-preview') return
    positionOverlays()
    viewer?.apply(true)
  })
  document.querySelectorAll('[data-quantity]').forEach(button => button.addEventListener('click', () => {
    const input = document.querySelector('input[name="count"]')
    input.value = button.dataset.quantity
    input.focus()
  }))
})()
