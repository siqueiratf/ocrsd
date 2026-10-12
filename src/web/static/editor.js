(() => {
  const SVG = 'http://www.w3.org/2000/svg'
  const data = JSON.parse(document.getElementById('dados-editor').textContent)
  const app = document.getElementById('editor')
  const shell = app.querySelector('[data-shell]')
  const svg = app.querySelector('[data-tela]')
  const canvas = app.querySelector('[data-tela-container]')
  const inspector = app.querySelector('[data-inspetor]')
  const documentPanel = app.querySelector('[data-documento]')
  const tree = app.querySelector('[data-arvore]')
  const form = document.getElementById('form-salvar')
  const saveButton = app.querySelector('[data-salvar]')
  const saveState = app.querySelector('[data-estado]')
  const banner = app.querySelector('[data-banner]')
  const narrow = window.matchMedia('(max-width: 1180px)')
  const labels = data.type_labels
  const categories = data.type_category

  let definition = data.definition
  let fields = structuredClone(definition.fields)
  let layout = data.layout ? structuredClone(data.layout) : null
  const size = definition.page_size_px
  const dpi = definition.dpi || 300
  let revision = data.revision
  let saved = snapshot()
  let selection = null
  let tool = 'select'
  let zoom = 1
  let past = []
  let future = []
  let drag = null
  let placing = null
  let editing = ''
  let leaving = false
  let spaceDown = false
  let saving = false

  const clone = value => structuredClone(value)
  const normalize = text => String(text ?? '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase()
  const field = () => selection?.kind === 'field' ? fields[selection.index] : null
  const element = () => field() && selection.element != null ? field().elements?.[selection.element] : null
  const layoutItem = () => selection?.kind === 'layout' ? layout.elements[selection.index] : null
  const target = () => element() || field() || layoutItem()
  const metadata = () => Object.fromEntries(['label', 'description'].map(name => [name, form.elements[name]?.value ?? '']))

  function snapshot() {
    return {
      state: JSON.stringify({ fields, layout }),
      meta: JSON.stringify(Object.fromEntries(['label', 'description'].map(name => [name, form.elements[name]?.value ?? '']))),
      fields: new Map(fields.map(item => [item.field_id, JSON.stringify(item)])),
      layout: new Map((layout?.elements ?? []).map(item => [item.id, JSON.stringify(item)])),
    }
  }

  const dirty = () => saved.state !== JSON.stringify({ fields, layout }) || saved.meta !== JSON.stringify(metadata())

  function h(tag, attributes = {}, ...children) {
    const node = document.createElement(tag)
    for (const [name, value] of Object.entries(attributes)) {
      if (value == null || value === false) continue
      if (name === 'text') node.textContent = value
      else if (name === 'class') node.className = value
      else if (name.startsWith('on')) node.addEventListener(name.slice(2), value)
      else node.setAttribute(name, value === true ? '' : value)
    }
    node.append(...children.flat().filter(child => child != null && child !== false))
    return node
  }

  function s(tag, attributes = {}) {
    const node = document.createElementNS(SVG, tag)
    for (const [name, value] of Object.entries(attributes)) if (value != null) node.setAttribute(name, value)
    return node
  }

  function icon(name) {
    const source = document.getElementById('icones').content.querySelector(`[data-icone="${name}"] svg`)
    return source ? source.cloneNode(true) : h('span')
  }

  function toast(message, kind = 'erro') {
    window.OCRSDNotices?.show(message, kind)
  }

  function bounded(box, minimum = 4) {
    const [x1, y1, x2, y2] = [Math.min(box[0], box[2]), Math.min(box[1], box[3]), Math.max(box[0], box[2]), Math.max(box[1], box[3])]
    const width = Math.max(minimum, Math.min(size[0], x2 - x1))
    const height = Math.max(minimum, Math.min(size[1], y2 - y1))
    const left = Math.max(0, Math.min(size[0] - width, x1))
    const top = Math.max(0, Math.min(size[1] - height, y1))
    return [left, top, left + width, top + height].map(Math.round)
  }

  function uniqueId(base, used) {
    const slug = normalize(base).replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'campo'
    if (!used.has(slug)) return slug
    let suffix = 2
    while (used.has(`${slug}_${suffix}`)) suffix += 1
    return `${slug}_${suffix}`
  }

  function point(event) {
    const matrix = svg.getScreenCTM()
    if (!matrix) return [0, 0]
    const transformed = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    return [Math.round(transformed.x), Math.round(transformed.y)]
  }

  function remember(state = { fields: clone(fields), layout: clone(layout) }) {
    past = [...past.slice(-79), state]
    future = []
  }

  function changed() {
    renderCanvas()
    renderTree()
    renderHeader()
    renderDocument()
  }

  function commit(mutate, nextSelection = selection) {
    remember()
    mutate()
    selection = nextSelection
    editing = ''
    render()
  }

  function render() {
    renderCanvas()
    renderTree()
    renderInspector()
    renderHeader()
    renderDocument()
  }

  function fieldIssues(item, index) {
    const issues = []
    const id = String(item.field_id ?? '').trim()
    const rules = item.validation || {}
    if (!id) issues.push({ path: 'field_id', message: 'Informe o ID do campo', blocking: true })
    else if (fields.some((other, position) => position !== index && String(other.field_id).trim() === id)) issues.push({ path: 'field_id', message: 'Este ID já está em uso', blocking: true })
    const elementIds = (item.elements || []).map(entry => String(entry.element_id ?? '').trim())
    if (elementIds.some(value => !value) || new Set(elementIds).size !== elementIds.length) issues.push({ path: 'elements', message: 'Subelementos precisam de IDs únicos', blocking: true })
    for (const [low, high] of [['min_value', 'max_value'], ['min', 'max']]) {
      if (rules[low] != null && rules[high] != null && rules[low] > rules[high]) issues.push({ path: `validation.${low}`, message: 'O mínimo é maior que o máximo', blocking: true })
    }
    if (rules.mask != null && !/[#Aa*]/.test(rules.mask)) issues.push({ path: 'validation.mask', message: 'A máscara precisa usar #, A, a ou *', blocking: true })
    if (item.data_type === 'checkbox' && (item.elements || []).length < 2) issues.push({ path: 'elements', message: 'O checkbox precisa de ao menos duas opções' })
    if (['digit_sequence', 'text_sequence'].includes(item.data_type) && !(item.elements || []).length && !rules.digit_count) issues.push({ path: 'elements', message: 'A sequência precisa de caixas de caracteres' })
    if (item.data_type === 'enum' && !(rules.allowed_values || []).length) issues.push({ path: 'validation.allowed_values', message: 'Informe os valores permitidos' })
    if (splitDate(item) && (item.elements || []).map(child => child.part).sort().join() !== data.date_parts.map(([part]) => part).sort().join()) issues.push({ path: 'elements', message: 'A data separada precisa de uma área para o dia, uma para o mês e uma para o ano' })
    if (dnvBoxes(item) && (item.elements || []).length !== 11) issues.push({ path: 'elements', message: 'O número da DNV em caixas precisa de 11 caixas' })
    if (rules.semantic_key != null && !/^[A-Za-z0-9_-]{1,60}$/.test(rules.semantic_key)) issues.push({ path: 'validation.semantic_key', message: 'Use até 60 letras, números, “_” ou “-”', blocking: true })
    return issues
  }

  function layoutIssues(item, index) {
    const id = String(item.id ?? '').trim()
    if (!id) return [{ path: 'id', message: 'Informe o ID do elemento', blocking: true }]
    if (layout.elements.some((other, position) => position !== index && other.id === id)) return [{ path: 'id', message: 'Este ID já está em uso', blocking: true }]
    return []
  }

  const isUnsaved = (map, key, item) => map.get(key) !== JSON.stringify(item)

  function category(item) {
    return item.custom_type ? 'personalizados' : categories[item.data_type] || 'basicos'
  }

  function renderBox(group, box, reference, active, title) {
    const [x1, y1, x2, y2] = box
    const area = { x: x1, y: y1, width: Math.max(1, x2 - x1), height: Math.max(1, y2 - y1) }
    group.append(s('rect', { ...area, class: 'box-hit', 'data-ref': reference }), s('rect', { ...area, class: 'box-outline' }))
    const fontSize = 12 / zoom
    const labelWidth = Math.min(Math.max(30, title.length * fontSize * 0.6 + 12 / zoom), 600 / zoom)
    const labelHeight = fontSize * 1.6
    const labelY = y1 - labelHeight - 2 / zoom >= 0 ? y1 - labelHeight - 2 / zoom : y1 + 2 / zoom
    const label = s('g', { class: 'box-label' })
    label.append(s('rect', { x: x1, y: labelY, width: labelWidth, height: labelHeight, rx: 3 / zoom }))
    const text = s('text', { x: x1 + 6 / zoom, y: labelY + labelHeight * 0.7, 'font-size': fontSize })
    text.textContent = title
    label.append(text)
    group.append(label)
    if (active) {
      const handle = 11 / zoom
      const cx = (x1 + x2) / 2
      const cy = (y1 + y2) / 2
      for (const [direction, x, y] of [['nw', x1, y1], ['n', cx, y1], ['ne', x2, y1], ['e', x2, cy], ['se', x2, y2], ['s', cx, y2], ['sw', x1, y2], ['w', x1, cy]]) {
        const node = s('rect', { x: x - handle / 2, y: y - handle / 2, width: handle, height: handle, rx: 2 / zoom, class: 'box-handle', 'data-handle': direction })
        node.style.cursor = `${direction}-resize`
        group.append(node)
      }
    }
  }

  function renderLayoutVisual(item) {
    const [x1, y1, x2, y2] = item.bbox
    const scale = dpi / 72
    if (item.kind === 'text') {
      const fontSize = item.font_size * scale
      const anchor = { left: 'start', center: 'middle', right: 'end' }[item.align] || 'start'
      const x = { left: x1, center: (x1 + x2) / 2, right: x2 }[item.align] ?? x1
      const node = s('text', { class: 'layout-text', 'font-size': fontSize, 'font-weight': item.bold ? 700 : 400, 'text-anchor': anchor })
      String(item.text ?? '').split('\n').forEach((line, index) => {
        const span = s('tspan', { x, y: y1 + fontSize * 0.92 + index * fontSize * 1.2 })
        span.textContent = line || ' '
        node.append(span)
      })
      return node
    }
    const stroke = item.stroke * scale
    if (item.kind === 'line') {
      return x2 - x1 >= y2 - y1
        ? s('line', { class: 'layout-line', x1, x2, y1: (y1 + y2) / 2, y2: (y1 + y2) / 2, 'stroke-width': stroke })
        : s('line', { class: 'layout-line', x1: (x1 + x2) / 2, x2: (x1 + x2) / 2, y1, y2, 'stroke-width': stroke })
    }
    return s('rect', { class: `layout-box${item.fill ? ' is-filled' : ''}`, x: x1, y: y1, width: x2 - x1, height: y2 - y1, 'stroke-width': stroke })
  }

  function renderCanvas() {
    svg.setAttribute('viewBox', `0 0 ${size[0]} ${size[1]}`)
    svg.setAttribute('width', Math.round(size[0] * zoom))
    svg.setAttribute('height', Math.round(size[1] * zoom))
    const nodes = []
    if (data.preview) nodes.push(s('image', { href: data.preview, x: 0, y: 0, width: size[0], height: size[1], preserveAspectRatio: 'none' }))
    else nodes.push(s('rect', { x: 0, y: 0, width: size[0], height: size[1], fill: '#fff' }))
    for (const [index, item] of (layout?.elements ?? []).entries()) {
      nodes.push(renderLayoutVisual(item))
      const active = selection?.kind === 'layout' && selection.index === index
      const group = s('g', { class: `box cat-layout${active ? ' is-selected' : ''}${isUnsaved(saved.layout, item.id, item) ? ' is-unsaved' : ''}${layoutIssues(item, index).length ? ' is-invalid' : ''}` })
      renderBox(group, item.bbox, `l:${index}`, active, `${item.id} · ${{ text: 'Texto fixo', line: 'Linha', box: 'Caixa' }[item.kind]}`)
      nodes.push(group)
    }
    const selectedGroups = []
    for (const [index, item] of fields.entries()) {
      const active = selection?.kind === 'field' && selection.index === index
      const issues = fieldIssues(item, index)
      const group = s('g', { class: `box cat-${category(item)}${active && selection.element == null ? ' is-selected' : ''}${isUnsaved(saved.fields, item.field_id, item) ? ' is-unsaved' : ''}${issues.length ? ' is-invalid' : ''}`, 'data-campo': index })
      renderBox(group, item.bbox, `f:${index}`, active && selection.element == null, item.label || item.field_id || 'Campo sem ID')
      for (const [position, child] of (item.elements || []).entries()) {
        const current = active && selection.element === position
        const childGroup = s('g', { class: `box is-child cat-${category(item)}${current ? ' is-selected' : ''}` })
        renderBox(childGroup, child.bbox, `f:${index}:${position}`, current, child.label || child.element_id || '')
        if (current) selectedGroups.push(childGroup)
        else group.append(childGroup)
      }
      if (active) selectedGroups.unshift(group)
      else nodes.push(group)
    }
    nodes.push(...selectedGroups)
    if (drag?.kind === 'draw' && drag.box) {
      const [x1, y1, x2, y2] = drag.box
      nodes.push(s('rect', { x: x1, y: y1, width: x2 - x1, height: y2 - y1, class: 'draft-rect' }))
    }
    svg.replaceChildren(...nodes)
    app.querySelector('[data-zoom]').textContent = `${Math.round(zoom * 100)}%`
  }

  function treeRow(attributes, title, subtitle, flags, child = false) {
    return h('button', { type: 'button', role: 'treeitem', class: `tree-item${child ? ' is-child' : ''}`, ...attributes },
      h('span', { class: `dot cat-${attributes['data-categoria'] || 'basicos'}` }),
      h('span', { class: 'tree-text' }, h('b', { text: title }), h('small', { text: subtitle })),
      ...flags)
  }

  function renderTree() {
    const needle = normalize(app.querySelector('[data-busca-arvore]').value.trim())
    const matches = (...values) => !needle || values.some(value => normalize(value).includes(needle))
    const rows = []
    const groups = new Map()
    for (const [index, item] of fields.entries()) {
      const key = item.bloco || item.group || ''
      if (!groups.has(key)) groups.set(key, [])
      groups.get(key).push([index, item])
    }
    const total = fields.length + (layout?.elements.length ?? 0)
    app.querySelector('[data-total]').textContent = total
    for (const [group, entries] of groups) {
      const groupRows = []
      for (const [index, item] of entries) {
        const children = (item.elements || []).map((child, position) => [child, position]).filter(([child]) => matches(child.label, child.element_id))
        const visible = matches(item.label, item.field_id, item.description, labels[item.data_type]) || children.length
        if (!visible) continue
        const flags = []
        if (fieldIssues(item, index).length) flags.push(h('span', { class: 'tree-flag is-invalid', text: 'Revisar' }))
        else if (isUnsaved(saved.fields, item.field_id, item)) flags.push(h('span', { class: 'tree-flag is-unsaved', text: 'Não salvo' }))
        const selected = selection?.kind === 'field' && selection.index === index && selection.element == null
        groupRows.push(treeRow({ 'data-ref': `f:${index}`, 'aria-selected': String(selected), 'data-categoria': category(item) },
          item.label || item.field_id || 'Campo sem ID', `${item.field_id || '—'} · ${labels[item.data_type] || item.data_type || 'Texto'}${item.elements?.length ? ` · ${item.elements.length} subelementos` : ''}`, flags))
        const expanded = (selection?.kind === 'field' && selection.index === index) || needle
        if (expanded) {
          for (const [child, position] of (needle ? children : (item.elements || []).map((child, position) => [child, position]))) {
            const current = selection?.kind === 'field' && selection.index === index && selection.element === position
            groupRows.push(treeRow({ 'data-ref': `f:${index}:${position}`, 'aria-selected': String(current), 'data-categoria': category(item) },
              child.label || child.element_id, `${child.element_id}${child.value ? ` · ${child.value}` : ''}${child.index != null ? ` · posição ${child.index}` : ''}`, [], true))
          }
        }
      }
      if (groupRows.length) {
        if (group || groups.size > 1) rows.push(h('p', { class: 'tree-group', text: group || 'Sem grupo' }))
        rows.push(...groupRows)
      }
    }
    if (layout?.elements.length) {
      const layoutRows = layout.elements.map((item, index) => [item, index]).filter(([item]) => matches(item.id, item.text, item.kind))
      if (layoutRows.length) rows.push(h('p', { class: 'tree-group', text: 'Layout do documento' }))
      for (const [item, index] of layoutRows) {
        const selected = selection?.kind === 'layout' && selection.index === index
        rows.push(treeRow({ 'data-ref': `l:${index}`, 'aria-selected': String(selected), 'data-categoria': 'layout' },
          item.kind === 'text' ? (item.text || 'Texto vazio').split('\n')[0] : { line: 'Linha', box: 'Caixa' }[item.kind], item.id,
          isUnsaved(saved.layout, item.id, item) ? [h('span', { class: 'tree-flag is-unsaved', text: 'Não salvo' })] : []))
      }
    }
    if (!rows.length) rows.push(h('p', { class: 'palette-empty', text: total ? 'Nenhum item corresponde à busca.' : 'O documento ainda não tem campos. Use a aba Adicionar para inserir o primeiro.' }))
    tree.replaceChildren(...rows)
  }

  function getPath(object, path) {
    return path.split('.').reduce((value, key) => value?.[key], object)
  }

  function setPath(object, path, value) {
    const keys = path.split('.')
    let current = object
    for (const key of keys.slice(0, -1)) {
      if (!current[key] || typeof current[key] !== 'object') current[key] = {}
      current = current[key]
    }
    if (value === undefined) {
      delete current[keys.at(-1)]
      if (keys.length > 1 && !Object.keys(current).length) delete object[keys[0]]
    } else current[keys.at(-1)] = value
  }

  function readValue(input) {
    const kind = input.dataset.kind
    if (kind === 'bool') return input.checked ? true : undefined
    const raw = input.value
    if (kind === 'lines') {
      const values = [...new Set(raw.split('\n').map(item => item.trim()).filter(Boolean))]
      return values.length ? values : undefined
    }
    if (kind === 'number' || kind === 'int' || kind === 'percent') {
      if (raw.trim() === '') return undefined
      const number = Number(raw)
      if (!Number.isFinite(number)) return null
      if (kind === 'percent') return number ? Math.round(Math.min(95, Math.max(0, number)) * 100) / 10000 : undefined
      return kind === 'int' ? Math.trunc(number) : number
    }
    if (kind === 'required') return raw
    return raw.trim() === '' ? undefined : raw
  }

  function formatValue(value, kind) {
    if (value == null) return ''
    if (kind === 'lines') return Array.isArray(value) ? value.join('\n') : String(value)
    if (kind === 'percent') return String(Math.round(value * 100))
    return String(value)
  }

  function control(label, path, kind = 'text', options = {}) {
    const value = getPath(options.source || target(), path)
    let input
    if (kind === 'bool') {
      input = h('input', { type: 'checkbox', 'data-path': path, 'data-kind': kind })
      input.checked = Boolean(value)
      return h('label', { class: 'check' }, input, h('span', { text: label }), options.tooltip ? h('span', { class: 'hint', text: ` · ${options.tooltip}` }) : null)
    }
    if (options.choices) {
      input = h('select', { 'data-path': path, 'data-kind': kind })
      for (const [optionValue, optionLabel] of options.choices) input.add(new Option(optionLabel, optionValue))
      if (value != null && ![...input.options].some(option => option.value === String(value))) input.add(new Option(String(value), String(value)))
      input.value = value == null ? (options.fallback ?? '') : String(value)
    } else if (kind === 'lines' || options.multiline) {
      input = h('textarea', { rows: options.rows || 3, 'data-path': path, 'data-kind': kind, spellcheck: 'false' })
      input.value = formatValue(value, kind)
    } else {
      input = h('input', {
        type: ['number', 'int', 'percent'].includes(kind) ? 'number' : 'text', 'data-path': path, 'data-kind': kind,
        step: kind === 'number' ? 'any' : '1', min: kind === 'percent' ? 0 : options.min, max: kind === 'percent' ? 95 : options.max,
        placeholder: options.placeholder, spellcheck: 'false', autocomplete: 'off', class: options.mono ? 'mono' : null,
      })
      input.value = formatValue(value, kind)
    }
    return h('label', { class: 'field', 'data-field-path': path, title: options.tooltip }, h('span', { text: label }), input,
      options.hint ? h('small', { class: 'hint', text: options.hint }) : null, h('small', { class: 'field-message', 'data-message': path, hidden: true }))
  }

  function section(title, children, open = true) {
    return h('details', { class: 'inspector-section', open }, h('summary', { text: title }), h('div', { class: 'form-stack compact' }, ...children))
  }

  function typeChoices() {
    return Object.entries(labels).map(([key, label]) => [key, label])
  }

  function geometrySection(item, open = false) {
    const [x1, y1, x2, y2] = item.bbox
    const millimetres = value => (value / dpi * 25.4).toLocaleString('pt-BR', { maximumFractionDigits: 1 })
    return section('Geometria', [
      h('div', { class: 'grid-2' },
        control('X', 'bbox.0', 'int', { tooltip: 'Distância da borda esquerda em pixels a ' + dpi + ' DPI' }),
        control('Y', 'bbox.1', 'int', { tooltip: 'Distância da borda superior em pixels' }),
        h('label', { class: 'field', title: 'Largura em pixels' }, h('span', { text: 'Largura' }), h('input', { type: 'number', min: 1, 'data-geometry': 'w', value: x2 - x1 })),
        h('label', { class: 'field', title: 'Altura em pixels' }, h('span', { text: 'Altura' }), h('input', { type: 'number', min: 1, 'data-geometry': 'h', value: y2 - y1 }))),
      h('p', { class: 'mm-hint', 'data-mm': true, text: `≈ ${millimetres(x2 - x1)} × ${millimetres(y2 - y1)} mm · origem em ${millimetres(x1)}, ${millimetres(y1)} mm · ${dpi} DPI` }),
    ], open)
  }

  function dataSection(item) {
    const type = item.data_type || 'text'
    const controls = []
    const allowed = (label = 'Valores permitidos', hint = 'Um por linha. O gerador escolhe um destes valores.') => control(label, 'validation.allowed_values', 'lines', { hint, rows: 4 })
    if (type === 'text') controls.push(control('Máscara', 'validation.mask', 'text', { placeholder: '###.###-##', hint: '# dígito · A maiúscula · a minúscula · * letra ou dígito', tooltip: 'Gera valores no formato da máscara' }), allowed(), control('Tamanho máximo', 'validation.max_length', 'int', { min: 1 }))
    if (type === 'person_name') controls.push(control('Permitir números no nome', 'validation.allow_digits', 'bool'), control('Aplicar regras linguísticas pt-BR', 'validation.portuguese_rules', 'bool'))
    if (type === 'integer') controls.push(h('div', { class: 'grid-2' }, control('Valor mínimo', 'validation.min_value', 'number'), control('Valor máximo', 'validation.max_value', 'number')), control('Máscara', 'validation.mask', 'text', { placeholder: '####' }))
    if (type === 'decimal') controls.push(h('div', { class: 'grid-2' }, control('Valor mínimo', 'validation.min_value', 'number'), control('Valor máximo', 'validation.max_value', 'number')), control('Casas decimais', 'validation.decimals', 'int', { min: 0, max: 6 }))
    if (type === 'date') controls.push(control('Formato', 'validation.format', 'text', { choices: data.date_formats, fallback: 'DD/MM/AAAA', tooltip: 'Formato do valor gerado' }), splitDate(item) ? h('p', { class: 'hint', text: 'Uma única data é gerada e escrita em três áreas: dia, mês e ano.' }) : null)
    if (type === 'sex') controls.push(control('Representação', 'validation.format', 'text', { choices: data.sex_formats, fallback: 'completo' }))
    if (type === 'dnv_number') controls.push(control('Formato', 'validation.format', 'text', { choices: data.dnv_formats, fallback: 'agrupado' }), control('Prefixo', 'validation.prefix', 'text', { mono: true, placeholder: data.dnv_prefix, hint: `Número sintético de teste no formato XX-XXXXXXXX-X. O prefixo ${data.dnv_prefix} identifica numeração de teste; o último algarismo não é um dígito verificador oficial.` }))
    if (type === 'time') controls.push(h('p', { class: 'hint', text: 'Horários gerados no formato HH:MM.' }))
    if (type === 'enum') controls.push(allowed('Opções da lista', 'Obrigatório. Um valor por linha.'))
    if (['uf', 'municipality', 'address', 'occupation'].includes(type)) controls.push(allowed('Valores permitidos', 'Opcional. Sem valores, o gerador usa o catálogo do tipo.'))
    if (type === 'document') controls.push(control('Tipo de documento', 'validation.document_kind', 'text', { choices: Object.entries(data.document_kinds), fallback: 'cpf' }))
    if (['cep', 'cns', 'signature'].includes(type)) controls.push(h('p', { class: 'hint', text: { cep: 'CEP gerado no formato 00000-000.', cns: 'Número com 15 dígitos.', signature: 'Nome manuscrito com a escrita configurada na geração.' }[type] }))
    if (type === 'digit_sequence') controls.push(control('Conteúdo', 'validation.format', 'text', { choices: [['', 'Número'], ['DDMMYYYY', 'Data DDMMAAAA']], tooltip: 'Como os dígitos são preenchidos' }), h('div', { class: 'grid-2' }, control('Valor mínimo', 'validation.min_value', 'number'), control('Valor máximo', 'validation.max_value', 'number')))
    if (type === 'text_sequence') controls.push(control('Expressão regular', 'validation.pattern', 'text', { mono: true }))
    if (type === 'checkbox') controls.push(control('Seleção', 'selection_mode', 'text', { choices: [['single', 'Uma opção marcada'], ['multiple', 'Várias opções marcadas']], fallback: 'single' }))
    controls.push(control('Campo vazio em', 'validation.blank_probability', 'percent', { hint: '% dos documentos gerados. Zero torna o campo sempre preenchido.', tooltip: 'Simula campos opcionais não preenchidos' }))
    return section('Dados e geração', controls)
  }

  function partSelect(child, position) {
    const select = h('select', { 'data-element': position, 'data-element-key': 'part', 'aria-label': `Parte da data do subelemento ${position + 1}` })
    for (const [value, label] of data.date_parts) select.add(new Option(label, value))
    select.value = child.part ?? ''
    return select
  }

  function elementsSection(item) {
    const type = item.data_type
    const isCheckbox = type === 'checkbox'
    const isDate = splitDate(item)
    const rows = (item.elements || []).map((child, position) => h('div', { class: `element-row${selection.element === position ? ' is-active' : ''}` },
      h('input', { 'data-element': position, 'data-element-key': 'label', value: child.label ?? '', 'aria-label': `Rótulo do subelemento ${position + 1}` }),
      isCheckbox
        ? h('input', { 'data-element': position, 'data-element-key': 'value', value: child.value ?? '', placeholder: 'valor', 'aria-label': `Valor do subelemento ${position + 1}` })
        : isDate ? partSelect(child, position)
        : h('input', { type: 'number', 'data-element': position, 'data-element-key': 'index', value: child.index ?? '', 'aria-label': `Posição do subelemento ${position + 1}` }),
      h('button', { type: 'button', class: 'icon-btn', 'data-acao': 'selecionar-subelemento', 'data-indice': position, 'aria-label': 'Selecionar no documento', title: 'Selecionar no documento', text: '◎' })))
    return section(isCheckbox ? 'Opções do checkbox' : isDate ? 'Dia, mês e ano' : 'Caixas de caracteres', [
      rows.length ? h('div', { class: 'element-list' }, rows) : h('p', { class: 'hint', text: isCheckbox ? 'Desenhe ao menos duas opções marcáveis.' : isDate ? 'Divida a área em dia, mês e ano ou desenhe cada parte.' : 'Divida a área em caixas ou desenhe cada caixa.' }),
      h('div', { class: 'element-tools' },
        h('button', { type: 'button', class: 'btn btn-sm', 'data-acao': 'desenhar-subelemento', title: 'Desenhe a nova área no documento', text: isCheckbox ? 'Desenhar opção' : isDate ? 'Desenhar parte' : 'Desenhar caixa' }),
        isDate || dnvBoxes(item) ? null : h('label', { class: 'inline-number', title: 'Substitui os subelementos por áreas iguais' }, h('input', { type: 'number', min: 1, max: 60, value: (item.elements || []).length || (isCheckbox ? 2 : 8), 'data-divisoes': true, 'aria-label': 'Quantidade de divisões' })),
        h('button', { type: 'button', class: 'btn btn-sm', 'data-acao': 'dividir', text: isDate ? 'Dividir em dia, mês e ano' : dnvBoxes(item) ? 'Dividir em 11 caixas' : 'Dividir área' })),
    ])
  }

  function inspectorHead(title, subtitle, iconName, tone) {
    return h('div', { class: 'inspector-head' },
      h('span', { class: `element-icon cat-${tone}` }, icon(iconName)),
      h('div', {}, h('b', { text: title, 'data-titulo-inspetor': true }), h('small', { text: subtitle })))
  }

  function renderInspector() {
    const item = target()
    documentPanel.hidden = Boolean(item)
    if (!item) {
      inspector.replaceChildren()
      return
    }
    const parts = []
    if (selection.kind === 'layout') {
      const name = { text: 'Texto fixo', line: 'Linha', box: 'Caixa' }[item.kind]
      parts.push(inspectorHead(item.id, `${name} · elemento de layout`, { text: 'heading', line: 'line', box: 'frame' }[item.kind], 'layout'))
      const options = item.kind === 'text'
        ? [control('Conteúdo', 'text', 'required', { multiline: true, rows: 3 }), h('div', { class: 'grid-2' }, control('Fonte (pt)', 'font_size', 'number', { min: 4, max: 72 }), control('Alinhamento', 'align', 'text', { choices: [['left', 'Esquerda'], ['center', 'Centro'], ['right', 'Direita']] })), control('Negrito', 'bold', 'bool')]
        : [control('Espessura (pt)', 'stroke', 'number', { min: 0.25, max: 10 }), item.kind === 'box' ? control('Fundo cinza claro', 'fill', 'bool') : null]
      parts.push(section('Identificação', [control('ID do elemento', 'id', 'required', { mono: true })]), section('Aparência', options), geometrySection(item, true))
    } else if (selection.element != null) {
      const parent = field()
      parts.push(inspectorHead(item.label || item.element_id, `Subelemento de ${parent.label || parent.field_id}`, 'digits', category(parent)))
      parts.push(section('Identificação', [
        control('ID do subelemento', 'element_id', 'required', { mono: true }), control('Rótulo', 'label', 'required'),
        parent.data_type === 'checkbox' ? control('Valor quando marcado', 'value', 'text', { tooltip: 'Registrado no ground truth' })
          : splitDate(parent) ? control('Parte da data', 'part', 'text', { choices: data.date_parts })
          : control('Posição na sequência', 'index', 'int', { min: 0 }),
      ]), geometrySection(item, true))
      parts.push(h('button', { type: 'button', class: 'btn btn-sm', 'data-acao': 'voltar-campo', text: `Voltar para ${parent.label || parent.field_id}` }))
    } else {
      const type = item.data_type || 'text'
      parts.push(inspectorHead(item.label || item.field_id || 'Campo sem ID', `${labels[type] || type}${item.custom_type ? ` · tipo ${item.custom_type}` : ''}`, data.type_icons?.[type] || 'type-text', category(item)))
      const issues = fieldIssues(item, selection.index)
      if (issues.length) parts.push(h('ul', { class: 'issues inspector-issues' }, issues.map(issue => h('li', { text: issue.message }))))
      parts.push(section('Identificação', [
        control('ID do campo', 'field_id', 'required', { mono: true, tooltip: 'Identificador único usado no JSON e no ground truth' }),
        control('Rótulo', 'label', 'required'),
        control('Descrição', 'description', 'text', { multiline: true, rows: 2 }),
        control('Tipo', 'data_type', 'text', { choices: typeChoices(), fallback: 'text' }),
        control('Grupo', 'bloco', 'text', { hint: 'Organiza a aba Contextos.' }),
      ]))
      parts.push(dataSection(item))
      if (['checkbox', 'digit_sequence', 'text_sequence'].includes(type) || splitDate(item) || dnvBoxes(item)) parts.push(elementsSection(item))
      parts.push(geometrySection(item))
      const known = new Set(['field_id', 'label', 'description', 'data_type', 'bbox', 'validation', 'elements', 'selection_mode', 'render_as_printed', 'always_redact', 'notes', 'bloco', 'custom_type'])
      const preserved = Object.keys(item).filter(key => !known.has(key))
      parts.push(section('Avançado', [
        control('Exibir como texto digitado', 'render_as_printed', 'bool', { tooltip: 'Ignora a escrita manuscrita na geração' }),
        control('Sempre ocultar no processamento', 'always_redact', 'bool', { tooltip: 'Usado pela etapa de ocultação do pipeline' }),
        control('Distância máxima de correção', 'validation.max_edit_distance', 'int', { min: 0, tooltip: 'Tolerância usada na validação de OCR' }),
        ['date', 'sex', 'dnv_number'].includes(type) ? control('Vínculo semântico', 'validation.semantic_key', 'text', { mono: true, placeholder: { date: 'nascimento', sex: 'sexo', dnv_number: 'dnv' }[type], hint: 'Campos do mesmo tipo com o mesmo vínculo recebem o mesmo valor no documento.' }) : null,
        control('Notas', 'notes', 'text', { multiline: true, rows: 2 }),
        preserved.length ? h('p', { class: 'hint', text: `Metadados preservados: ${preserved.join(', ')}.` }) : null,
      ], false))
      parts.push(h('div', { class: 'inspector-actions' },
        h('button', { type: 'button', class: 'btn btn-sm', 'data-acao': 'mover-acima', title: 'Mover para cima na ordem', 'aria-label': 'Mover para cima', text: '↑' }),
        h('button', { type: 'button', class: 'btn btn-sm', 'data-acao': 'mover-abaixo', title: 'Mover para baixo na ordem', 'aria-label': 'Mover para baixo', text: '↓' }),
        h('button', { type: 'button', class: 'btn btn-sm', 'data-acao': 'salvar-tipo', disabled: ['checkbox', 'digit_sequence', 'text_sequence'].includes(type), title: 'Cria um tipo reutilizável com esta configuração', text: 'Salvar como tipo' })))
    }
    parts.push(h('div', { class: 'inspector-actions' },
      h('button', { type: 'button', class: 'btn btn-sm', 'data-acao': 'duplicar', title: 'Duplicar (Ctrl+D)', text: 'Duplicar' }),
      h('button', { type: 'button', class: 'btn btn-sm btn-danger', 'data-acao': 'excluir', title: 'Excluir (Delete)', text: 'Excluir' })))
    inspector.replaceChildren(h('div', { class: 'inspector' }, parts))
    markIssues()
  }

  function markIssues() {
    if (selection?.kind !== 'field' || selection.element != null) return
    const issues = fieldIssues(field(), selection.index)
    for (const label of inspector.querySelectorAll('[data-field-path]')) {
      const issue = issues.find(entry => entry.path === label.dataset.fieldPath)
      label.classList.toggle('field-invalid', Boolean(issue))
      const message = label.querySelector('[data-message]')
      message.hidden = !issue
      message.textContent = issue?.message || ''
    }
  }

  function syncGeometry() {
    const item = target()
    if (!item) return
    const [x1, y1, x2, y2] = item.bbox
    const values = { 'bbox.0': x1, 'bbox.1': y1 }
    for (const [path, value] of Object.entries(values)) {
      const input = inspector.querySelector(`[data-path="${path}"]`)
      if (input && document.activeElement !== input) input.value = value
    }
    for (const [key, value] of [['w', x2 - x1], ['h', y2 - y1]]) {
      const input = inspector.querySelector(`[data-geometry="${key}"]`)
      if (input && document.activeElement !== input) input.value = value
    }
    const hint = inspector.querySelector('[data-mm]')
    if (hint) {
      const millimetres = value => (value / dpi * 25.4).toLocaleString('pt-BR', { maximumFractionDigits: 1 })
      hint.textContent = `≈ ${millimetres(x2 - x1)} × ${millimetres(y2 - y1)} mm · origem em ${millimetres(x1)}, ${millimetres(y1)} mm · ${dpi} DPI`
    }
  }

  function renderHeader() {
    const isDirty = dirty()
    saveState.dataset.estado = saving ? 'salvando' : isDirty ? 'pendente' : saveState.dataset.estado === 'erro' ? 'erro' : 'salvo'
    app.querySelector('[data-estado-texto]').textContent = saving ? 'Salvando…' : isDirty ? 'Alterações não salvas' : saveState.dataset.estado === 'erro' ? 'Falha ao salvar' : 'Salvo'
    saveButton.disabled = saving || !isDirty
    const publish = app.querySelector('[data-publicar]')
    if (publish) {
      const pending = publicationIssues()
      publish.disabled = isDirty || pending.length > 0
      publish.dataset.tooltip = isDirty ? 'Salve as alterações antes de publicar' : pending.length ? `Pendência: ${pending[0]}` : 'Disponibiliza o modelo salvo na Geração'
    }
    app.querySelector('[data-acao="desfazer"]').disabled = !past.length
    app.querySelector('[data-acao="refazer"]').disabled = !future.length
    for (const button of app.querySelectorAll('[data-ferramenta]')) button.setAttribute('aria-pressed', String(button.dataset.ferramenta === tool))
    app.dataset.tool = tool
    app.classList.toggle('is-placing', Boolean(placing))
  }

  function publicationIssues() {
    const issues = fields.flatMap((item, index) => fieldIssues(item, index).map(issue => `${item.label || item.field_id || 'Campo sem ID'}: ${issue.message}`))
    if (!fields.length) issues.unshift('Adicione ao menos um campo ao documento')
    return issues
  }

  function renderDocument() {
    const issues = publicationIssues()
    const list = app.querySelector('[data-pendencias]')
    list.replaceChildren(...(issues.length ? issues.slice(0, 12).map(text => h('li', { text })) : [h('li', { class: 'is-ok', text: 'Nenhuma pendência. Salve e publique para usar na geração.' })]))
    if (issues.length > 12) list.append(h('li', { text: `E mais ${issues.length - 12} pendências.` }))
    const millimetres = value => (value / dpi * 25.4).toLocaleString('pt-BR', { maximumFractionDigits: 1 })
    const facts = [['Página', `${millimetres(size[0])} × ${millimetres(size[1])} mm`], ['Resolução', `${size[0]} × ${size[1]} px · ${dpi} DPI`], ['Campos', fields.length],
      ['Subelementos', fields.reduce((total, item) => total + (item.elements?.length || 0), 0)]]
    if (layout) facts.push(['Elementos de layout', layout.elements.length])
    app.querySelector('[data-fatos]').replaceChildren(...facts.flatMap(([term, value]) => [h('dt', { text: term }), h('dd', { text: String(value) })]))
  }

  function select(next, locate = false) {
    selection = next
    editing = ''
    if (narrow.matches && next) shell.dataset.direito = 'aberto'
    render()
    if (locate) locateSelection()
    tree.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' })
  }

  function locateSelection() {
    const item = target()
    if (!item) return
    const [x1, y1, x2, y2] = item.bbox
    const svgBox = svg.getBoundingClientRect()
    const canvasBox = canvas.getBoundingClientRect()
    canvas.scrollTo({
      left: canvas.scrollLeft + svgBox.left - canvasBox.left + (x1 + x2) / 2 * zoom - canvas.clientWidth / 2,
      top: canvas.scrollTop + svgBox.top - canvasBox.top + (y1 + y2) / 2 * zoom - canvas.clientHeight / 2,
      behavior: 'smooth',
    })
    const reference = selection.kind === 'layout' ? `l:${selection.index}` : selection.element != null ? `f:${selection.index}:${selection.element}` : `f:${selection.index}`
    const group = svg.querySelector(`[data-ref="${reference}"]`)?.parentNode
    group?.classList.add('is-located')
    setTimeout(() => group?.classList.remove('is-located'), 1900)
  }

  function parseReference(reference) {
    const [kind, index, child] = reference.split(':')
    return kind === 'l' ? { kind: 'layout', index: Number(index), element: null } : { kind: 'field', index: Number(index), element: child == null ? null : Number(child) }
  }

  function setZoom(value, anchor) {
    const next = OCRSDPdfView.clampZoom(value)
    const rect = canvas.getBoundingClientRect()
    const clientX = anchor?.clientX ?? rect.left + canvas.clientWidth / 2
    const clientY = anchor?.clientY ?? rect.top + canvas.clientHeight / 2
    const before = svg.getBoundingClientRect()
    const docX = (clientX - before.left) / zoom
    const docY = (clientY - before.top) / zoom
    zoom = next
    renderCanvas()
    const after = svg.getBoundingClientRect()
    canvas.scrollLeft += after.left + docX * zoom - clientX
    canvas.scrollTop += after.top + docY * zoom - clientY
  }

  function fit(mode = 'page') {
    zoom = OCRSDPdfView.fitScale(canvas.clientWidth, canvas.clientHeight, size[0], size[1], mode, zoom, 64)
    renderCanvas()
    canvas.scrollTo({ left: 0, top: 0 })
  }

  function cardConfig(card) {
    return JSON.parse(card.dataset.config)
  }

  function startPlacing(item, card = null) {
    placing = item
    tool = 'select'
    for (const node of app.querySelectorAll('.element-card[aria-pressed="true"]')) node.removeAttribute('aria-pressed')
    card?.setAttribute('aria-pressed', 'true')
    banner.hidden = false
    app.querySelector('[data-banner-texto]').textContent = item.kind === 'element'
      ? 'Desenhe a nova área no documento.'
      : `Clique ou arraste no documento para posicionar “${item.title}”. Shift mantém o elemento ativo.`
    renderHeader()
  }

  function stopPlacing() {
    placing = null
    banner.hidden = true
    for (const node of app.querySelectorAll('.element-card[aria-pressed="true"]')) node.removeAttribute('aria-pressed')
    renderHeader()
  }

  function splitBox(box, count, square) {
    const [x1, y1, x2, y2] = box
    const width = (x2 - x1) / count
    return Array.from({ length: count }, (_, index) => {
      if (!square) return bounded([x1 + index * width, y1, x1 + (index + 1) * width, y2], 2)
      const side = Math.max(4, Math.min(width, y2 - y1) * 0.7)
      const top = y1 + (y2 - y1 - side) / 2
      return bounded([x1 + index * width, top, x1 + index * width + side, top + side], 2)
    })
  }

  function makeElements(item, count) {
    const isCheckbox = item.data_type === 'checkbox'
    if (splitDate(item)) {
      const [x1, y1, x2, y2] = item.bbox
      const unit = (x2 - x1) / 8
      return data.date_parts.map(([part, label], index) => ({
        element_id: `${item.field_id}_${{ day: 'dia', month: 'mes', year: 'ano' }[part] || part}`, label, part,
        bbox: bounded([x1 + [0, 2, 4][index] * unit, y1, x1 + [2, 4, 8][index] * unit, y2], 2),
      }))
    }
    if (dnvBoxes(item)) count = 11
    return splitBox(item.bbox, count, isCheckbox).map((box, index) => isCheckbox
      ? { element_id: `${item.field_id}_${index}`, label: `Opção ${index + 1}`, bbox: box }
      : { element_id: `${item.field_id}_${index}`, label: `${item.label} ${index}`, bbox: box, index })
  }

  function create(item, box) {
    if (item.kind === 'element') {
      const parent = fields[item.fieldIndex]
      const pattern = new RegExp(`^${parent.field_id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}_(\\d+)$`)
      const suffix = Math.max(-1, ...(parent.elements || []).map(child => Number(pattern.exec(child.element_id)?.[1] ?? -1))) + 1
      const child = { element_id: `${parent.field_id}_${suffix}`, label: parent.data_type === 'checkbox' ? `Opção ${suffix + 1}` : `${parent.label} ${suffix}`, bbox: box }
      if (splitDate(parent)) {
        const free = data.date_parts.find(([part]) => !(parent.elements || []).some(other => other.part === part)) || data.date_parts[0]
        child.part = free[0]
        child.label = free[1]
      } else if (parent.data_type !== 'checkbox') child.index = suffix
      commit(() => { parent.elements = [...(parent.elements || []), child] }, { kind: 'field', index: item.fieldIndex, element: (parent.elements || []).length })
      return
    }
    if (item.layout) {
      const used = new Set(layout.elements.map(entry => entry.id))
      const entry = { id: uniqueId(item.title, used), ...clone(item.layout), bbox: box }
      commit(() => layout.elements.push(entry), { kind: 'layout', index: layout.elements.length, element: null })
      return
    }
    const config = clone(item.config)
    const title = config.label || item.title
    delete config.label
    const entry = { field_id: uniqueId(config.custom_type || item.key || title, new Set(fields.map(other => String(other.field_id)))), label: title, description: '', ...config, bbox: box }
    if (item.boxes || entry.data_type === 'checkbox') entry.elements = makeElements(entry, item.boxes || 2)
    commit(() => fields.push(entry), { kind: 'field', index: fields.length, element: null })
  }

  function defaultBox(item, center) {
    const [width, height] = item.size || [600, 70]
    return bounded([center[0] - width / 2, center[1] - height / 2, center[0] + width / 2, center[1] + height / 2])
  }

  function resizeBox(original, direction, dx, dy) {
    const box = [...original]
    if (direction.includes('w')) box[0] = original[0] + dx
    if (direction.includes('e')) box[2] = original[2] + dx
    if (direction.includes('n')) box[1] = original[1] + dy
    if (direction.includes('s')) box[3] = original[3] + dy
    return bounded(box)
  }

  function startPan(event) {
    drag = { kind: 'pan', x: event.clientX, y: event.clientY, left: canvas.scrollLeft, top: canvas.scrollTop }
    app.classList.add('is-panning-active')
    canvas.setPointerCapture(event.pointerId)
  }

  canvas.addEventListener('pointerdown', event => {
    if (event.button === 1 || (event.button === 0 && (tool === 'pan' || spaceDown) && !placing)) {
      event.preventDefault()
      startPan(event)
      return
    }
    if (event.button !== 0) return
    if (!svg.contains(event.target)) {
      if (!placing && !event.target.closest('[data-banner]')) select(null)
      return
    }
    const position = point(event)
    if (placing) {
      drag = { kind: 'draw', start: position, box: null, client: [event.clientX, event.clientY] }
      svg.setPointerCapture(event.pointerId)
      return
    }
    const handle = event.target.closest('[data-handle]')
    const item = target()
    if (handle && item) {
      drag = { kind: 'resize', direction: handle.dataset.handle, start: position, original: [...item.bbox], before: { fields: clone(fields), layout: clone(layout) }, moved: false }
      svg.setPointerCapture(event.pointerId)
      return
    }
    const hit = event.target.closest('[data-ref]')
    if (!hit) {
      select(null)
      return
    }
    const next = parseReference(hit.dataset.ref)
    const same = selection && next.kind === selection.kind && next.index === selection.index && next.element === selection.element
    if (!same) select(next)
    const current = target()
    drag = {
      kind: 'move', start: position, client: [event.clientX, event.clientY], original: [...current.bbox], before: { fields: clone(fields), layout: clone(layout) }, moved: false,
      children: selection.kind === 'field' && selection.element == null ? (current.elements || []).map(child => [...child.bbox]) : null,
    }
    svg.setPointerCapture(event.pointerId)
  })

  canvas.addEventListener('pointermove', event => {
    if (!drag) return
    if (drag.kind === 'pan') {
      canvas.scrollLeft = drag.left - (event.clientX - drag.x)
      canvas.scrollTop = drag.top - (event.clientY - drag.y)
      return
    }
    const position = point(event)
    if (drag.kind === 'draw') {
      drag.box = bounded([...drag.start, ...position], 1)
      renderCanvas()
      return
    }
    if (!drag.moved && Math.hypot(event.clientX - (drag.client?.[0] ?? event.clientX), event.clientY - (drag.client?.[1] ?? event.clientY)) < 3 && drag.kind === 'move') return
    const item = target()
    const dx = position[0] - drag.start[0]
    const dy = position[1] - drag.start[1]
    if (!drag.moved) {
      remember(drag.before)
      drag.moved = true
    }
    if (drag.kind === 'move') {
      const box = bounded([drag.original[0] + dx, drag.original[1] + dy, drag.original[2] + dx, drag.original[3] + dy])
      const [mx, my] = [box[0] - drag.original[0], box[1] - drag.original[1]]
      item.bbox = box
      if (drag.children) item.elements.forEach((child, position) => { child.bbox = bounded([drag.children[position][0] + mx, drag.children[position][1] + my, drag.children[position][2] + mx, drag.children[position][3] + my], 1) })
    } else {
      item.bbox = resizeBox(drag.original, drag.direction, dx, dy)
    }
    renderCanvas()
    syncGeometry()
  })

  function endDrag(event) {
    if (!drag) return
    const current = drag
    drag = null
    app.classList.remove('is-panning-active')
    if (current.kind === 'draw') {
      const tiny = !current.box || (current.box[2] - current.box[0] < 8 && current.box[3] - current.box[1] < 8)
      const item = placing
      if (!(event?.shiftKey && item.kind !== 'element')) stopPlacing()
      create(item, tiny ? (item.kind === 'element' ? bounded([current.start[0], current.start[1], current.start[0] + 60, current.start[1] + 60]) : defaultBox(item, current.start)) : bounded(current.box))
      return
    }
    if (current.moved) changed()
    renderInspector()
  }

  canvas.addEventListener('pointerup', endDrag)
  canvas.addEventListener('pointercancel', endDrag)
  canvas.addEventListener('wheel', event => {
    if (!event.ctrlKey && !event.metaKey) return
    event.preventDefault()
    setZoom(zoom * (event.deltaY < 0 ? OCRSDPdfView.WHEEL_STEP : 1 / OCRSDPdfView.WHEEL_STEP), event)
  }, { passive: false })

  canvas.addEventListener('dragover', event => {
    if (!event.dataTransfer.types.includes('application/x-ocrsd-elemento')) return
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
    canvas.classList.add('is-drop-target')
  })
  canvas.addEventListener('dragleave', event => { if (!canvas.contains(event.relatedTarget)) canvas.classList.remove('is-drop-target') })
  canvas.addEventListener('drop', event => {
    canvas.classList.remove('is-drop-target')
    const key = event.dataTransfer.getData('application/x-ocrsd-elemento')
    const card = app.querySelector(`.element-card[data-elemento="${CSS.escape(key)}"]`)
    if (!card) return
    event.preventDefault()
    stopPlacing()
    const item = cardConfig(card)
    create(item, defaultBox(item, point(event)))
  })

  app.addEventListener('dragstart', event => {
    const card = event.target.closest('.element-card')
    if (!card) return
    event.dataTransfer.setData('application/x-ocrsd-elemento', card.dataset.elemento)
    event.dataTransfer.setData('text/plain', card.querySelector('b').textContent)
    event.dataTransfer.effectAllowed = 'copy'
  })

  inspector.addEventListener('input', event => {
    const input = event.target
    const item = target()
    if (!item) return
    if (input.dataset.geometry) {
      if (input.value === '' || !Number.isFinite(Number(input.value))) return
      const key = `${selection.kind}:${selection.index}:${selection.element}:geometry`
      if (editing !== key) remember()
      editing = key
      const value = Math.max(1, Math.round(Number(input.value)))
      const box = [...item.bbox]
      if (input.dataset.geometry === 'w') box[2] = box[0] + value
      else box[3] = box[1] + value
      item.bbox = bounded(box)
      changed()
      syncGeometry()
      return
    }
    if (input.dataset.element != null) {
      const child = field().elements[Number(input.dataset.element)]
      const key = `${selection.index}:${input.dataset.element}:${input.dataset.elementKey}`
      if (editing !== key) remember()
      editing = key
      const raw = input.value
      if (input.dataset.elementKey === 'index') {
        if (raw === '') delete child.index
        else child.index = Math.trunc(Number(raw))
      } else if (raw.trim() === '' && input.dataset.elementKey === 'value') delete child.value
      else child[input.dataset.elementKey] = raw
      changed()
      return
    }
    const path = input.dataset.path
    if (!path) return
    const value = readValue(input)
    if (value === null) return
    const key = `${selection.kind}:${selection.index}:${selection.element}:${path}`
    const discrete = input.type === 'checkbox' || input.tagName === 'SELECT'
    if (editing !== key || discrete) remember()
    editing = discrete ? '' : key
    if (path.startsWith('bbox.')) {
      if (value === undefined) return
      const box = [...item.bbox]
      const index = Number(path.split('.')[1])
      const delta = value - box[index]
      box[index] += delta
      box[index + 2] += delta
      item.bbox = bounded(box)
      if (selection.kind === 'field' && selection.element == null) item.elements?.forEach(child => { child.bbox = bounded(child.bbox.map((coordinate, position) => coordinate + (position % 2 === index ? delta : 0)), 1) })
    } else setPath(item, path, value)
    const structural = path === 'data_type' || (path === 'validation.format' && ['date', 'dnv_number'].includes(item.data_type))
    if (structural) applyTypeDefaults(item)
    const title = inspector.querySelector('[data-titulo-inspetor]')
    if (title && ['label', 'field_id', 'id', 'element_id'].includes(path)) title.textContent = item.label || item.field_id || item.id || item.element_id || ''
    if (structural) {
      renderInspector()
      inspector.querySelector(`[data-path="${path}"]`)?.focus()
    } else markIssues()
    changed()
  })

  inspector.addEventListener('change', () => { editing = '' })

  function applyTypeDefaults(item) {
    const rules = item.validation || {}
    if (item.data_type === 'uf' && !rules.allowed_values) item.validation = { ...rules, allowed_values: [...data.uf_values], max_edit_distance: 1 }
    if (item.data_type === 'person_name' && rules.portuguese_rules == null) item.validation = { ...rules, allow_digits: false, portuguese_rules: true, language: 'pt-BR' }
    if (item.data_type === 'checkbox') item.selection_mode = item.selection_mode || 'single'
    if (item.data_type === 'text_sequence' && !rules.pattern) item.validation = { ...rules, pattern: '^[A-Z]+$' }
    if (item.data_type === 'date' && !rules.format) item.validation = { ...rules, format: 'DD/MM/AAAA' }
    if (item.data_type === 'document' && !rules.document_kind) item.validation = { ...rules, document_kind: 'cpf' }
    if (item.data_type === 'sex' && !rules.format) item.validation = { ...rules, format: 'completo' }
    if (item.data_type === 'dnv_number' && !rules.format) item.validation = { ...rules, format: 'agrupado', prefix: data.dnv_prefix }
    if ((splitDate(item) || dnvBoxes(item)) && !(item.elements || []).length) item.elements = makeElements(item, 11)
  }

  function splitDate(item) {
    return item?.data_type === 'date' && item.validation?.format === 'SEPARADO'
  }

  function dnvBoxes(item) {
    return item?.data_type === 'dnv_number' && item.validation?.format === 'caixas'
  }

  function duplicate() {
    const item = target()
    if (!item) return
    if (selection.kind === 'layout') {
      const copy = { ...clone(item), id: uniqueId(`${item.id}_copia`, new Set(layout.elements.map(entry => entry.id))), bbox: bounded(item.bbox.map(value => value + 24)) }
      commit(() => layout.elements.push(copy), { kind: 'layout', index: layout.elements.length, element: null })
    } else if (selection.element != null) {
      const parent = field()
      const copy = { ...clone(item), element_id: uniqueId(`${item.element_id}_copia`, new Set(parent.elements.map(entry => entry.element_id))), bbox: bounded(item.bbox.map(value => value + 18), 2) }
      commit(() => parent.elements.push(copy), { ...selection, element: parent.elements.length })
    } else {
      const id = uniqueId(`${item.field_id}_copia`, new Set(fields.map(entry => String(entry.field_id))))
      const copy = { ...clone(item), field_id: id, label: `${item.label} — cópia`, bbox: bounded(item.bbox.map(value => value + 24)) }
      copy.elements = copy.elements?.map((child, position) => ({ ...child, element_id: `${id}_${position}`, bbox: bounded(child.bbox.map(value => value + 24), 1) }))
      if (!copy.elements) delete copy.elements
      commit(() => fields.push(copy), { kind: 'field', index: fields.length, element: null })
    }
  }

  function remove() {
    if (!target()) return
    if (selection.kind === 'layout') commit(() => layout.elements.splice(selection.index, 1), null)
    else if (selection.element != null) commit(() => field().elements.splice(selection.element, 1), { ...selection, element: null })
    else commit(() => fields.splice(selection.index, 1), null)
  }

  function move(offset) {
    if (selection?.kind !== 'field' || selection.element != null) return
    const next = selection.index + offset
    if (next < 0 || next >= fields.length) return
    commit(() => { [fields[selection.index], fields[next]] = [fields[next], fields[selection.index]] }, { ...selection, index: next })
  }

  function divide() {
    const item = field()
    const count = Math.max(1, Math.min(60, Math.trunc(Number(inspector.querySelector('[data-divisoes]')?.value) || 0)))
    if (!item || !count) return
    if (item.elements?.length && !window.confirm(`Substituir os ${item.elements.length} subelementos atuais por ${count} áreas iguais?`)) return
    commit(() => { item.elements = makeElements(item, count) })
  }

  function undo() {
    const previous = past.at(-1)
    if (!previous) return
    future = [{ fields: clone(fields), layout: clone(layout) }, ...future]
    past = past.slice(0, -1)
    fields = previous.fields
    layout = previous.layout
    if (!target()) selection = null
    editing = ''
    render()
  }

  function redo() {
    const next = future[0]
    if (!next) return
    past = [...past, { fields: clone(fields), layout: clone(layout) }]
    future = future.slice(1)
    fields = next.fields
    layout = next.layout
    if (!target()) selection = null
    editing = ''
    render()
  }

  async function reload() {
    if (dirty() && !window.confirm('Descartar as alterações ainda não salvas e recarregar a versão salva?')) return
    const response = await fetch(data.source, { credentials: 'same-origin' })
    if (!response.ok) return toast('Não foi possível recarregar o mapeamento')
    const state = await response.json()
    definition = state.definition
    fields = clone(definition.fields)
    layout = state.layout ? clone(state.layout) : null
    revision = state.revision
    past = []
    future = []
    selection = null
    saved = snapshot()
    render()
    toast('Versão salva recarregada', 'sucesso')
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify({ ...definition, fields }, null, 2)], { type: 'application/json' })
    const link = h('a', { href: URL.createObjectURL(blob), download: `${data.mapping_id}.json` })
    link.click()
    setTimeout(() => URL.revokeObjectURL(link.href), 30000)
  }

  function togglePanel(side) {
    const key = side === 'esquerdo' ? 'esquerdo' : 'direito'
    if (narrow.matches) {
      shell.dataset[key] = shell.dataset[key] === 'aberto' ? '' : 'aberto'
      const other = key === 'esquerdo' ? 'direito' : 'esquerdo'
      if (shell.dataset[key] === 'aberto') shell.dataset[other] = ''
    } else {
      shell.dataset[key] = shell.dataset[key] === 'fechado' ? '' : 'fechado'
      store(`painel-${key}`, shell.dataset[key])
    }
  }

  function store(key, value) {
    try { localStorage.setItem(`ocrsd-editor-${key}`, value) } catch {}
  }

  function stored(key) {
    try { return localStorage.getItem(`ocrsd-editor-${key}`) } catch { return null }
  }

  function filterPalette() {
    const needle = normalize(app.querySelector('[data-busca-paleta]').value.trim())
    let visible = 0
    for (const group of app.querySelectorAll('.palette-group')) {
      let count = 0
      for (const card of group.querySelectorAll('.element-card')) {
        const show = !needle || needle.split(/\s+/).every(word => normalize(card.dataset.busca).includes(word))
        card.hidden = !show
        if (show) count += 1
      }
      group.querySelector('[data-contagem]').textContent = count
      group.hidden = Boolean(needle) && !count
      if (needle && count) group.querySelector('.group-toggle').setAttribute('aria-expanded', 'true')
      visible += count
    }
    app.querySelector('[data-sem-resultados]').hidden = !needle || Boolean(visible)
  }

  app.addEventListener('click', event => {
    const card = event.target.closest('.element-card')
    if (card) {
      if (card.getAttribute('aria-pressed') === 'true') stopPlacing()
      else startPlacing(cardConfig(card), card)
      return
    }
    const toggle = event.target.closest('.group-toggle')
    if (toggle) {
      const expanded = toggle.getAttribute('aria-expanded') !== 'true'
      toggle.setAttribute('aria-expanded', String(expanded))
      const collapsed = new Set(JSON.parse(stored('grupos') || '[]'))
      expanded ? collapsed.delete(toggle.closest('[data-grupo]').dataset.grupo) : collapsed.add(toggle.closest('[data-grupo]').dataset.grupo)
      store('grupos', JSON.stringify([...collapsed]))
      return
    }
    const tab = event.target.closest('[data-aba]')
    if (tab) return showTab(tab.dataset.aba)
    const row = event.target.closest('.tree-item')
    if (row) return select(parseReference(row.dataset.ref), true)
    const toolButton = event.target.closest('[data-ferramenta]')
    if (toolButton) {
      tool = toolButton.dataset.ferramenta
      stopPlacing()
      return renderHeader()
    }
    const back = event.target.closest('[data-voltar]')
    if (back) {
      if (dirty() && !window.confirm('Sair sem salvar as alterações do mapeamento?')) event.preventDefault()
      else leaving = true
      return
    }
    const action = event.target.closest('[data-acao]')?.dataset.acao
    const actions = {
      duplicar: duplicate, excluir: remove, desfazer: undo, refazer: redo, recarregar: reload, exportar: exportJson,
      afastar: () => setZoom(zoom / OCRSDPdfView.ZOOM_STEP), aproximar: () => setZoom(zoom * OCRSDPdfView.ZOOM_STEP), ajustar: () => fit('page'), largura: () => fit('width'),
      'painel-esquerdo': () => togglePanel('esquerdo'), 'painel-direito': () => togglePanel('direito'),
      'tela-cheia': () => document.fullscreenElement ? document.exitFullscreen() : app.requestFullscreen?.(),
      'cancelar-insercao': stopPlacing, 'mover-acima': () => move(-1), 'mover-abaixo': () => move(1), dividir: divide,
      'fechar-guia': () => { app.querySelector('[data-guia]').hidden = true; store('guia', 'oculta') },
      'desenhar-subelemento': () => startPlacing({ kind: 'element', fieldIndex: selection.index, title: 'subelemento' }),
      'selecionar-subelemento': () => select({ ...selection, element: Number(event.target.closest('[data-indice]').dataset.indice) }, true),
      'voltar-campo': () => select({ ...selection, element: null }),
      'salvar-tipo': () => {
        const item = field()
        htmx.ajax('GET', '/modelos/tipos/novo', { target: '#modal', values: { base_type: item.data_type || 'text', nome: item.label || '', validacao: JSON.stringify(item.validation || {}), aplicar: '1' } })
      },
    }
    actions[action]?.()
  })

  function showTab(name) {
    for (const tab of app.querySelectorAll('[data-aba]')) {
      const active = tab.dataset.aba === name
      tab.setAttribute('aria-selected', String(active))
      tab.tabIndex = active ? 0 : -1
      document.getElementById(tab.getAttribute('aria-controls')).hidden = !active
    }
    if (name === 'contextos') renderTree()
  }

  app.querySelector('.panel-tabs').addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return
    const next = app.querySelector('[data-aba][aria-selected="false"]')
    showTab(next.dataset.aba)
    next.focus()
  })

  app.querySelector('[data-busca-paleta]').addEventListener('input', filterPalette)
  app.querySelector('[data-busca-arvore]').addEventListener('input', renderTree)

  for (const resizer of app.querySelectorAll('[data-redimensionar]')) {
    const side = resizer.dataset.redimensionar
    const property = side === 'esquerdo' ? '--left' : '--right'
    const limits = side === 'esquerdo' ? [230, 440] : [270, 480]
    const apply = width => {
      const value = Math.max(limits[0], Math.min(limits[1], Math.round(width)))
      shell.style.setProperty(property, `${value}px`)
      store(`largura-${side}`, value)
    }
    const saved = Number(stored(`largura-${side}`))
    if (saved) shell.style.setProperty(property, `${Math.max(limits[0], Math.min(limits[1], saved))}px`)
    resizer.addEventListener('pointerdown', event => {
      const panel = app.querySelector(side === 'esquerdo' ? '.panel-left' : '.panel-right')
      const start = event.clientX
      const width = panel.getBoundingClientRect().width
      resizer.classList.add('is-dragging')
      resizer.setPointerCapture(event.pointerId)
      const moveHandler = moveEvent => apply(width + (side === 'esquerdo' ? 1 : -1) * (moveEvent.clientX - start))
      resizer.addEventListener('pointermove', moveHandler)
      resizer.addEventListener('pointerup', () => { resizer.classList.remove('is-dragging'); resizer.removeEventListener('pointermove', moveHandler) }, { once: true })
    })
    resizer.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return
      event.preventDefault()
      const panel = app.querySelector(side === 'esquerdo' ? '.panel-left' : '.panel-right')
      const growth = (event.key === 'ArrowRight' ? 16 : -16) * (side === 'esquerdo' ? 1 : -1)
      apply(panel.getBoundingClientRect().width + growth)
    })
  }

  form.addEventListener('input', event => {
    if (event.target.matches('[data-nome-modelo]')) {
      app.querySelector('[data-titulo]').textContent = event.target.value || data.model.label
      app.querySelector('[data-trilha-nome]').textContent = event.target.value || data.model.label
    }
    renderHeader()
  })

  function blockingIssues() {
    const blocking = fields.flatMap((item, index) => fieldIssues(item, index).some(issue => issue.blocking) ? [index] : [])
    const layoutBlocking = (layout?.elements ?? []).flatMap((item, index) => layoutIssues(item, index).length ? [index] : [])
    if (selection?.kind === 'field' && blocking.includes(selection.index)) return { ...selection, element: null }
    if (selection?.kind === 'layout' && layoutBlocking.includes(selection.index)) return selection
    if (blocking.length) return { kind: 'field', index: blocking[0], element: null }
    if (layoutBlocking.length) return { kind: 'layout', index: layoutBlocking[0], element: null }
    return null
  }

  function payload(parameters) {
    parameters.definition = JSON.stringify({ ...definition, fields })
    parameters.layout = layout ? JSON.stringify(layout) : ''
  }

  document.addEventListener('htmx:configRequest', event => {
    if (event.detail.elt === form) {
      payload(event.detail.parameters)
      event.detail.parameters.revision = revision
    } else if (event.detail.elt.closest?.('[data-previa]')) payload(event.detail.parameters)
  })

  form.addEventListener('htmx:beforeRequest', event => {
    if (event.detail.elt !== form) return
    const problem = blockingIssues()
    if (problem) {
      event.preventDefault()
      select(problem, true)
      toast('Corrija os itens destacados antes de salvar.')
      return
    }
    saving = true
    saved.pending = { ...snapshot() }
    renderHeader()
  })

  form.addEventListener('htmx:afterRequest', event => {
    if (event.detail.elt !== form) return
    saving = false
    if (!event.detail.successful) saveState.dataset.estado = 'erro'
    renderHeader()
  })

  document.body.addEventListener('editor-salvo', event => {
    revision = event.detail.revision
    saved = saved.pending || snapshot()
    saveState.dataset.estado = 'salvo'
    if (data.model.label !== undefined && form.elements.label) data.model.label = form.elements.label.value
    render()
  })

  document.body.addEventListener('campo-personalizado', event => {
    const detail = event.detail
    if (detail.apply && field() && selection.element == null) {
      const item = field()
      commit(() => {
        item.data_type = detail.config.data_type
        if (detail.config.validation) item.validation = clone(detail.config.validation)
        else delete item.validation
        if (detail.config.render_as_printed) item.render_as_printed = true
        item.custom_type = detail.config.custom_type
      })
      return
    }
    startPlacing({ key: detail.config.custom_type, title: detail.title, config: detail.config, size: [700, 70] })
  })

  document.body.addEventListener('htmx:afterSettle', event => {
    if (event.detail.target?.id === 'tipos-personalizados') filterPalette()
    renderHeader()
  })

  document.body.addEventListener('htmx:oobAfterSwap', () => renderHeader())

  document.addEventListener('htmx:beforeRequest', event => {
    if (event.target.closest?.('[data-saida]')) leaving = true
  })

  window.addEventListener('keydown', event => {
    if (document.getElementById('modal')?.open) return
    const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName) || event.target.isContentEditable
    const command = event.ctrlKey || event.metaKey
    if (command && ['+', '=', '-', '0'].includes(event.key)) {
      event.preventDefault()
      if (event.key === '0') fit('page')
      else setZoom(event.key === '-' ? zoom / OCRSDPdfView.ZOOM_STEP : zoom * OCRSDPdfView.ZOOM_STEP)
      return
    }
    if (command && event.key.toLowerCase() === 's') {
      event.preventDefault()
      if (dirty() && !saving) form.requestSubmit()
      return
    }
    if (typing) return
    if (event.key === ' ' && !spaceDown) {
      spaceDown = true
      app.classList.add('is-panning')
      event.preventDefault()
      return
    }
    if (event.key === 'Escape') {
      if (drag) endDrag()
      if (placing) stopPlacing()
      else if (selection?.element != null) select({ ...selection, element: null })
      else if (selection) select(null)
      return
    }
    if (command && event.key.toLowerCase() === 'z') {
      event.preventDefault()
      event.shiftKey ? redo() : undo()
    } else if (command && event.key.toLowerCase() === 'y') {
      event.preventDefault()
      redo()
    } else if (command && event.key.toLowerCase() === 'd') {
      event.preventDefault()
      duplicate()
    } else if (!command && (event.key === 'Delete' || event.key === 'Backspace')) {
      event.preventDefault()
      remove()
    } else if (!command && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key) && target()) {
      event.preventDefault()
      const step = event.shiftKey ? 10 : 1
      const [dx, dy] = { ArrowUp: [0, -step], ArrowDown: [0, step], ArrowLeft: [-step, 0], ArrowRight: [step, 0] }[event.key]
      const key = 'nudge'
      if (editing !== key) remember()
      editing = key
      const item = target()
      const before = [...item.bbox]
      item.bbox = bounded([before[0] + dx, before[1] + dy, before[2] + dx, before[3] + dy])
      const [mx, my] = [item.bbox[0] - before[0], item.bbox[1] - before[1]]
      if (selection.kind === 'field' && selection.element == null) item.elements?.forEach(child => { child.bbox = bounded([child.bbox[0] + mx, child.bbox[1] + my, child.bbox[2] + mx, child.bbox[3] + my], 1) })
      changed()
      syncGeometry()
    } else if (!command && event.key.toLowerCase() === 'v') {
      tool = 'select'
      renderHeader()
    } else if (!command && event.key.toLowerCase() === 'h') {
      tool = 'pan'
      stopPlacing()
    }
  }, { capture: true })

  window.addEventListener('keyup', event => {
    if (event.key === ' ') {
      spaceDown = false
      app.classList.remove('is-panning')
    }
    if (editing === 'nudge' && event.key.startsWith('Arrow')) editing = ''
  })

  window.addEventListener('beforeunload', event => {
    if (dirty() && !leaving) event.preventDefault()
  })

  narrow.addEventListener('change', () => { shell.dataset.esquerdo = ''; shell.dataset.direito = '' })

  if (stored('guia') === 'oculta') app.querySelector('[data-guia]').hidden = true
  if (!narrow.matches) {
    if (stored('painel-esquerdo') === 'fechado') shell.dataset.esquerdo = 'fechado'
    if (stored('painel-direito') === 'fechado') shell.dataset.direito = 'fechado'
  }
  for (const name of JSON.parse(stored('grupos') || '[]')) app.querySelector(`[data-grupo="${CSS.escape(name)}"] .group-toggle`)?.setAttribute('aria-expanded', 'false')
  render()
  requestAnimationFrame(() => fit('width'))
})()
