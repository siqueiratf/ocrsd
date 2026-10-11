(() => {
  const SVG = 'http://www.w3.org/2000/svg'
  const data = JSON.parse(document.getElementById('dados-editor').textContent)
  const root = document.getElementById('editor')
  const svg = root.querySelector('[data-tela]')
  const container = root.querySelector('[data-tela-container]')
  const list = root.querySelector('[data-lista]')
  const total = root.querySelector('[data-total]')
  const search = root.querySelector('[data-busca]')
  const inspector = root.querySelector('[data-inspetor]')
  const zoomLabel = root.querySelector('[data-zoom]')
  const form = document.getElementById('form-salvar')
  const saveButton = document.querySelector('[data-salvar]')
  const saveLabel = document.querySelector('[data-salvar-rotulo]')

  let definition = data.definition
  let fields = structuredClone(definition.fields)
  let size = definition.page_size_px
  let selected = fields[0]?.field_id ?? ''
  let tool = 'select'
  let zoom = 1
  let past = []
  let future = []
  let drag = null
  let dirty = false
  let leaving = false
  let editing = ''
  let savedTimer = 0

  const image = document.createElementNS(SVG, 'image')
  image.setAttribute('href', data.preview)
  image.setAttribute('preserveAspectRatio', 'none')
  const layer = document.createElementNS(SVG, 'g')
  svg.append(image, layer)

  const current = () => fields.find(field => field.field_id === selected)

  function svgNode(tag, attributes) {
    const node = document.createElementNS(SVG, tag)
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value)
    return node
  }

  function normalized(box) {
    return [Math.min(box[0], box[2]), Math.min(box[1], box[3]), Math.max(box[0], box[2]), Math.max(box[1], box[3])]
  }

  function bounded(box) {
    const result = normalized(box)
    const width = Math.min(size[0], result[2] - result[0])
    const height = Math.min(size[1], result[3] - result[1])
    const x1 = Math.max(0, Math.min(size[0] - width, result[0]))
    const y1 = Math.max(0, Math.min(size[1] - height, result[1]))
    return [x1, y1, x1 + width, y1 + height].map(Math.round)
  }

  function uniqueId(base) {
    const used = new Set(fields.map(field => field.field_id))
    if (!used.has(base)) return base
    let suffix = 2
    while (used.has(`${base}_${suffix}`)) suffix += 1
    return `${base}_${suffix}`
  }

  function markDirty() {
    dirty = true
    clearTimeout(savedTimer)
    saveButton.disabled = false
    saveLabel.textContent = 'Salvar mapeamento'
  }

  function remember(snapshot = structuredClone(fields)) {
    past = [...past.slice(-49), snapshot]
    future = []
    markDirty()
  }

  function commit(next, select = selected) {
    remember()
    fields = next
    selected = select
    editing = ''
    render()
  }

  function point(event) {
    const matrix = svg.getScreenCTM()
    if (!matrix) return [0, 0]
    const transformed = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    return [Math.round(transformed.x), Math.round(transformed.y)]
  }

  function renderList() {
    const needle = search.value.trim().toLowerCase()
    total.textContent = fields.length
    list.replaceChildren(...fields
      .filter(field => `${field.label} ${field.field_id}`.toLowerCase().includes(needle))
      .map(field => {
        const button = document.createElement('button')
        button.type = 'button'
        button.dataset.campo = field.field_id
        if (field.field_id === selected) button.setAttribute('aria-current', 'true')
        const label = document.createElement('b')
        label.textContent = field.label || field.field_id
        const meta = document.createElement('small')
        meta.textContent = `${field.field_id} · ${field.data_type ?? 'text'}`
        button.append(label, meta)
        return button
      }))
    if (!list.childElementCount) {
      const empty = document.createElement('p')
      empty.className = 'empty-hint'
      empty.textContent = fields.length ? 'Nenhum campo encontrado.' : 'Use a ferramenta Campo para desenhar a primeira área.'
      list.append(empty)
    }
  }

  function renderCanvas() {
    svg.setAttribute('viewBox', `0 0 ${size[0]} ${size[1]}`)
    svg.setAttribute('width', Math.round(size[0] * zoom))
    svg.setAttribute('height', Math.round(size[1] * zoom))
    image.setAttribute('width', size[0])
    image.setAttribute('height', size[1])
    layer.replaceChildren(...fields.map(field => {
      const [x1, y1, x2, y2] = field.bbox
      const active = field.field_id === selected
      const group = svgNode('g', { class: active ? 'box is-selected' : 'box' })
      const area = { x: x1, y: y1, width: Math.max(0, x2 - x1), height: Math.max(0, y2 - y1) }
      group.append(svgNode('rect', { ...area, class: 'box-hit', 'data-campo': field.field_id }))
      group.append(svgNode('rect', { ...area, class: 'box-outline' }))
      if (active) {
        [[x1, y1], [x2, y1], [x1, y2], [x2, y2]].forEach(([cx, cy], corner) => {
          group.append(svgNode('circle', { cx, cy, r: Math.max(5, 7 / zoom), class: 'box-handle', 'data-campo': field.field_id, 'data-canto': corner }))
        })
      }
      return group
    }))
  }

  function syncBox() {
    const field = current()
    if (!field) return
    const values = { x: field.bbox[0], y: field.bbox[1], w: field.bbox[2] - field.bbox[0], h: field.bbox[3] - field.bbox[1] }
    for (const [name, value] of Object.entries(values)) {
      const input = inspector.querySelector(`[data-prop="${name}"]`)
      if (input && document.activeElement !== input) input.value = value
    }
  }

  function renderInspector() {
    const field = current()
    if (!field) {
      inspector.replaceChildren(document.getElementById('modelo-vazio').content.cloneNode(true))
      return
    }
    const panel = document.getElementById('modelo-inspetor').content.cloneNode(true)
    const set = (name, value) => { panel.querySelector(`[data-prop="${name}"]`).value = value }
    panel.querySelector('[data-texto="tipo"]').textContent = `${field.data_type ?? 'text'} · selecionado`
    set('field_id', field.field_id)
    set('label', field.label ?? '')
    set('description', field.description ?? '')
    const type = panel.querySelector('[data-prop="data_type"]')
    if (field.data_type && ![...type.options].some(option => option.value === field.data_type)) type.add(new Option(field.data_type, field.data_type))
    type.value = field.data_type ?? 'text'
    set('selection_mode', field.selection_mode ?? 'single')
    set('validation', JSON.stringify(field.validation ?? {}, null, 2))
    panel.querySelector('[data-prop="render_as_printed"]').checked = Boolean(field.render_as_printed)
    panel.querySelector('[data-prop="always_redact"]').checked = Boolean(field.always_redact)
    if (field.elements?.length) {
      const hint = panel.querySelector('[data-texto="elementos"]')
      hint.hidden = false
      hint.textContent = `${field.elements.length} subelementos preservados. IDs, caixas, índices e valores continuam no JSON.`
    }
    inspector.replaceChildren(panel)
    syncBox()
  }

  function renderTools() {
    root.dataset.tool = tool
    for (const button of root.querySelectorAll('[data-ferramenta]')) button.setAttribute('aria-pressed', String(button.dataset.ferramenta === tool))
    root.querySelector('[data-acao="duplicar"]').disabled = !current()
    root.querySelector('[data-acao="excluir"]').disabled = !current()
    root.querySelector('[data-acao="desfazer"]').disabled = !past.length
    root.querySelector('[data-acao="refazer"]').disabled = !future.length
    zoomLabel.textContent = `${Math.round(zoom * 100)}%`
  }

  function render() {
    renderList()
    renderCanvas()
    renderInspector()
    renderTools()
  }

  function select(identifier) {
    selected = identifier
    editing = ''
    render()
    list.querySelector('[aria-current]')?.scrollIntoView({ block: 'nearest' })
  }

  function setZoom(value) {
    zoom = Math.max(0.2, Math.min(2.5, value))
    renderCanvas()
    renderTools()
  }

  function fit() {
    setZoom(Math.min((container.clientWidth - 40) / size[0], (container.clientHeight - 40) / size[1]))
  }

  function undo() {
    const previous = past.at(-1)
    if (!previous) return
    future = [structuredClone(fields), ...future]
    past = past.slice(0, -1)
    fields = previous
    if (!current()) selected = fields[0]?.field_id ?? ''
    editing = ''
    markDirty()
    render()
  }

  function redo() {
    const next = future[0]
    if (!next) return
    past = [...past, structuredClone(fields)]
    future = future.slice(1)
    fields = next
    if (!current()) selected = fields[0]?.field_id ?? ''
    editing = ''
    markDirty()
    render()
  }

  function duplicate() {
    const field = current()
    if (!field) return
    const identifier = uniqueId(`${field.field_id}_copia`)
    commit([...fields, { ...structuredClone(field), field_id: identifier, label: `${field.label} — cópia`, bbox: bounded(field.bbox.map(value => value + 18)) }], identifier)
  }

  function remove() {
    const field = current()
    if (field) commit(fields.filter(item => item !== field), '')
  }

  async function reload() {
    if (dirty && !window.confirm('Descartar as alterações ainda não salvas?')) return
    const response = await fetch(data.source, { credentials: 'same-origin' })
    if (!response.ok) return
    definition = await response.json()
    fields = structuredClone(definition.fields)
    size = definition.page_size_px
    selected = fields[0]?.field_id ?? ''
    past = []
    future = []
    dirty = false
    saveButton.disabled = true
    render()
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify({ ...definition, fields }, null, 2)], { type: 'application/json' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `${data.mapping_id}.json`
    link.click()
    setTimeout(() => URL.revokeObjectURL(link.href), 30000)
  }

  function setTool(next) {
    tool = next
    renderTools()
  }

  svg.addEventListener('pointerdown', event => {
    const handle = event.target.closest('[data-campo]')
    if (tool === 'select' && handle) {
      const field = fields.find(item => item.field_id === handle.dataset.campo)
      selected = field.field_id
      drag = { kind: handle.dataset.canto ? 'resize' : 'move', corner: Number(handle.dataset.canto ?? 3), start: point(event), original: [...field.bbox], before: structuredClone(fields), moved: false }
      svg.setPointerCapture(event.pointerId)
      editing = ''
      render()
      return
    }
    if (tool !== 'draw') return
    const start = point(event)
    const identifier = uniqueId(`campo_${fields.length + 1}`)
    commit([...fields, { field_id: identifier, label: 'Novo campo', data_type: 'text', description: '', bbox: [...start, ...start] }], identifier)
    drag = { kind: 'draw', start }
    svg.setPointerCapture(event.pointerId)
  })

  svg.addEventListener('pointermove', event => {
    const field = current()
    if (!drag || !field) return
    const position = point(event)
    if (drag.kind === 'draw') {
      field.bbox = bounded([...drag.start, ...position])
    } else if (drag.kind === 'move') {
      const [dx, dy] = [position[0] - drag.start[0], position[1] - drag.start[1]]
      const box = drag.original
      field.bbox = bounded([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy])
    } else {
      const box = [...drag.original]
      const [horizontal, vertical] = [[0, 2, 0, 2][drag.corner], [1, 1, 3, 3][drag.corner]]
      box[horizontal] = position[0]
      box[vertical] = position[1]
      field.bbox = bounded(box)
    }
    if (drag.before && !drag.moved) {
      remember(drag.before)
      drag.moved = true
    }
    renderCanvas()
    syncBox()
  })

  function endDrag() {
    if (!drag) return
    const field = current()
    if (drag.kind === 'draw' && field) {
      const [x1, y1, x2, y2] = field.bbox
      if (x2 - x1 < 4 || y2 - y1 < 4) field.bbox = bounded([x1, y1, x1 + 240, y1 + 48])
    }
    drag = null
    render()
  }

  svg.addEventListener('pointerup', endDrag)
  svg.addEventListener('pointercancel', endDrag)

  list.addEventListener('click', event => {
    const button = event.target.closest('[data-campo]')
    if (!button) return
    setTool('select')
    select(button.dataset.campo)
  })

  search.addEventListener('input', renderList)

  function updateField(input) {
    const field = current()
    const name = input.dataset.prop
    if (['x', 'y', 'w', 'h'].includes(name)) {
      if (input.value === '' || !Number.isFinite(Number(input.value))) return false
      const value = Math.round(Number(input.value))
      const box = [...field.bbox]
      if (name === 'x' || name === 'y') {
        const index = name === 'x' ? 0 : 1
        box[index + 2] += value - box[index]
        box[index] = value
      } else {
        const index = name === 'w' ? 0 : 1
        box[index + 2] = box[index] + Math.max(1, value)
      }
      field.bbox = bounded(box)
      return true
    }
    if (input.type === 'checkbox') field[name] = input.checked
    else field[name] = input.value
    if (name === 'field_id') selected = input.value
    if (name === 'data_type') inspector.querySelector('[data-texto="tipo"]').textContent = `${input.value} · selecionado`
    return true
  }

  inspector.addEventListener('input', event => {
    const input = event.target.closest('[data-prop]')
    if (!input || !current()) return
    if (input.dataset.prop === 'validation') {
      input.removeAttribute('aria-invalid')
      inspector.querySelector('[data-erro]').hidden = true
      return
    }
    const key = `${fields.indexOf(current())}:${input.dataset.prop}`
    const snapshot = structuredClone(fields)
    if (!updateField(input)) return
    if (editing !== key || input.type === 'checkbox' || input.tagName === 'SELECT') remember(snapshot)
    editing = input.type === 'checkbox' || input.tagName === 'SELECT' ? '' : key
    renderList()
    renderCanvas()
    renderTools()
    if (['x', 'y', 'w', 'h'].includes(input.dataset.prop)) syncBox()
  })

  inspector.addEventListener('change', event => {
    const input = event.target.closest('[data-prop]')
    editing = ''
    if (!input || input.dataset.prop !== 'validation' || !current()) return
    try {
      const parsed = JSON.parse(input.value)
      if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') throw new Error()
      remember()
      current().validation = parsed
      renderTools()
    } catch {
      input.setAttribute('aria-invalid', 'true')
      inspector.querySelector('[data-erro]').hidden = false
    }
  })

  inspector.addEventListener('focusout', event => {
    if (['x', 'y', 'w', 'h'].includes(event.target.dataset?.prop)) syncBox()
  })

  document.addEventListener('click', event => {
    const toolButton = event.target.closest('[data-ferramenta]')
    if (toolButton) setTool(toolButton.dataset.ferramenta)
    const action = event.target.closest('[data-acao]')?.dataset.acao
    if (action === 'duplicar') duplicate()
    if (action === 'excluir') remove()
    if (action === 'desfazer') undo()
    if (action === 'refazer') redo()
    if (action === 'recarregar') reload()
    if (action === 'afastar') setZoom(zoom - 0.1)
    if (action === 'aproximar') setZoom(zoom + 0.1)
    if (action === 'ajustar') fit()
    if (action === 'exportar') exportJson()
    const back = event.target.closest('[data-voltar]')
    if (back) {
      if (dirty && !window.confirm('Sair sem salvar as alterações do mapeamento?')) event.preventDefault()
      else leaving = true
    }
  })

  form.addEventListener('input', markDirty)

  form.addEventListener('htmx:configRequest', event => {
    event.detail.parameters.definition = JSON.stringify({ ...definition, fields })
  })

  form.addEventListener('htmx:beforeRequest', () => {
    saveButton.disabled = true
    saveLabel.textContent = 'Salvando…'
  })

  form.addEventListener('htmx:afterRequest', event => {
    if (!event.detail.successful) {
      saveButton.disabled = false
      saveLabel.textContent = 'Salvar mapeamento'
      return
    }
    past = []
    future = []
    dirty = false
    editing = ''
    saveLabel.textContent = 'Salvo'
    savedTimer = setTimeout(() => { saveLabel.textContent = 'Salvar mapeamento' }, 2500)
    renderTools()
  })

  document.addEventListener('htmx:beforeRequest', event => {
    if (event.target.closest('[data-saida]')) leaving = true
  })

  window.addEventListener('keydown', event => {
    const target = event.target
    const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable
    const command = event.ctrlKey || event.metaKey
    if (document.getElementById('modal')?.open) return
    if (command && ['+', '=', '-', '0'].includes(event.key)) {
      event.preventDefault()
      if (event.key === '0') fit()
      else setZoom(zoom + (event.key === '-' ? -0.1 : 0.1))
      return
    }
    if (command && event.key.toLowerCase() === 's') {
      event.preventDefault()
      if (dirty) form.requestSubmit()
      return
    }
    if (typing) return
    if (command && event.key.toLowerCase() === 'z') {
      event.preventDefault()
      if (event.shiftKey) redo()
      else undo()
    } else if (command && event.key.toLowerCase() === 'y') {
      event.preventDefault()
      redo()
    } else if (command && event.key.toLowerCase() === 'd') {
      event.preventDefault()
      duplicate()
    } else if (!command && (event.key === 'Delete' || event.key === 'Backspace')) {
      event.preventDefault()
      remove()
    } else if (!command && event.key.toLowerCase() === 'v') {
      setTool('select')
    } else if (!command && event.key.toLowerCase() === 'n') {
      setTool('draw')
    } else if (event.key === 'Escape' && drag) {
      endDrag()
    }
  }, { capture: true })

  window.addEventListener('beforeunload', event => {
    if (dirty && !leaving) event.preventDefault()
  })

  render()
  requestAnimationFrame(fit)
})()
