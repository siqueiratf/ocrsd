(() => {
  const ZOOM_STEP = 1.2
  const WHEEL_STEP = 1.1
  const MIN_ZOOM = 0.1
  const MAX_ZOOM = 3
  const clampZoom = value => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value))
  const fitScale = (width, height, intrinsicWidth, intrinsicHeight, mode, manualScale, padding = 0) => {
    if (!intrinsicWidth || !intrinsicHeight) return 1
    const fitWidth = Math.max(100, width - padding) / intrinsicWidth
    const fitHeight = Math.max(100, height - padding) / intrinsicHeight
    return clampZoom(mode === 'page' ? Math.min(fitWidth, fitHeight) : mode === 'width' ? fitWidth : manualScale)
  }

  function createViewer({ stage, natural, render, view = { mode: 'page', scale: 1 }, output, onChange, root = stage, shortcuts = null }) {
    let current = { ...view }
    let applied = 0
    let measured = null
    let pan = null
    let suppressClick = false
    const lifecycle = new AbortController()
    const signal = lifecycle.signal
    const style = () => getComputedStyle(stage)
    const measure = () => {
      const computed = style()
      return {
        width: stage.clientWidth - parseFloat(computed.paddingLeft || 0) - parseFloat(computed.paddingRight || 0),
        height: stage.clientHeight - parseFloat(computed.paddingTop || 0) - parseFloat(computed.paddingBottom || 0),
      }
    }
    const scaleFor = size => {
      const [width, height] = natural() || []
      return width ? fitScale(size.width, size.height, width, height, current.mode, current.scale) : 0
    }
    const changed = () => {
      if (output) output.textContent = applied ? `${Math.round(applied * 100)}%` : '—'
      onChange?.({ ...current })
    }
    const apply = (force = false) => {
      const size = measure()
      const scale = scaleFor(size)
      if (!scale) return
      if (force || Math.abs(scale - applied) > 0.0005) {
        applied = scale
        render(scale)
      }
      measured = measure()
      changed()
    }
    const resized = () => {
      const size = measure()
      if (measured && Math.abs(size.width - measured.width) < 1 && Math.abs(size.height - measured.height) < 1) return
      measured = size
      if (current.mode !== 'manual') apply()
    }
    const zoomTo = (value, anchor) => {
      if (!applied) return
      const box = stage.getBoundingClientRect()
      const computed = style()
      const padLeft = parseFloat(computed.paddingLeft || 0)
      const padTop = parseFloat(computed.paddingTop || 0)
      const x = anchor ? anchor.clientX - box.left - stage.clientLeft : stage.clientWidth / 2
      const y = anchor ? anchor.clientY - box.top - stage.clientTop : stage.clientHeight / 2
      const ratio = clampZoom(value) / applied
      const left = (stage.scrollLeft + x - padLeft) * ratio + padLeft - x
      const top = (stage.scrollTop + y - padTop) * ratio + padTop - y
      current = { mode: 'manual', scale: clampZoom(value) }
      apply()
      stage.scrollLeft = left
      stage.scrollTop = top
    }
    const fit = mode => {
      current = { ...current, mode }
      apply()
      stage.scrollTo({ left: Math.max(0, (stage.scrollWidth - stage.clientWidth) / 2), top: 0 })
    }
    const center = () => stage.scrollTo({
      left: Math.max(0, (stage.scrollWidth - stage.clientWidth) / 2),
      top: Math.max(0, (stage.scrollHeight - stage.clientHeight) / 2),
      behavior: 'smooth',
    })
    const fullscreen = () => {
      if (document.fullscreenElement === root) document.exitFullscreen()
      else root.requestFullscreen?.()
    }
    const actions = {
      'zoom-in': () => zoomTo(applied * ZOOM_STEP), 'zoom-out': () => zoomTo(applied / ZOOM_STEP),
      'fit-page': () => fit('page'), 'fit-width': () => fit('width'), center, fullscreen,
    }
    root.addEventListener('click', event => {
      const control = event.target.closest('[data-pdf-action]')
      if (control && root.contains(control)) actions[control.dataset.pdfAction]?.()
    }, { signal })
    stage.addEventListener('wheel', event => {
      if (!event.ctrlKey && !event.metaKey) return
      event.preventDefault()
      zoomTo(applied * (event.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP), event)
    }, { passive: false, signal })
    stage.addEventListener('pointerdown', event => {
      if (event.button !== 0 && event.button !== 1) return
      if (event.target === stage && (event.offsetX > stage.clientWidth || event.offsetY > stage.clientHeight)) return
      if (event.button === 1) event.preventDefault()
      pan = { x: event.clientX, y: event.clientY, left: stage.scrollLeft, top: stage.scrollTop, id: event.pointerId, moved: false }
    }, { signal })
    stage.addEventListener('pointermove', event => {
      if (!pan || pan.id !== event.pointerId) return
      const dx = event.clientX - pan.x
      const dy = event.clientY - pan.y
      if (!pan.moved && Math.hypot(dx, dy) < 4) return
      if (!pan.moved) {
        pan.moved = true
        stage.setPointerCapture?.(event.pointerId)
        stage.classList.add('is-panning')
      }
      stage.scrollLeft = pan.left - dx
      stage.scrollTop = pan.top - dy
    }, { signal })
    const endPan = () => {
      if (pan?.moved) suppressClick = true
      pan = null
      stage.classList.remove('is-panning')
    }
    stage.addEventListener('pointerup', endPan, { signal })
    stage.addEventListener('pointercancel', endPan, { signal })
    stage.addEventListener('click', event => {
      if (!suppressClick) return
      suppressClick = false
      event.preventDefault()
      event.stopPropagation()
    }, { capture: true, signal })
    stage.addEventListener('load', () => apply(true), { capture: true, signal })
    if (shortcuts) {
      shortcuts.addEventListener('keydown', event => {
        if (!(event.ctrlKey || event.metaKey) || !['+', '=', '-', '0'].includes(event.key) || document.getElementById('modal')?.open) return
        event.preventDefault()
        if (event.key === '0') fit('page')
        else actions[event.key === '-' ? 'zoom-out' : 'zoom-in']()
      }, { signal })
    }
    const frame = globalThis.requestAnimationFrame?.bind(globalThis) || (callback => callback())
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => frame(resized))
    observer?.observe(stage)
    apply(true)
    return {
      apply, fit, center, zoomTo, actions,
      view: () => ({ ...current }),
      scale: () => applied,
      destroy: () => { lifecycle.abort(); observer?.disconnect() },
    }
  }

  globalThis.OCRSDPdfView = { ZOOM_STEP, WHEEL_STEP, MIN_ZOOM, MAX_ZOOM, clampZoom, fitScale, createViewer }
  if (typeof module !== 'undefined') module.exports = globalThis.OCRSDPdfView
})()
