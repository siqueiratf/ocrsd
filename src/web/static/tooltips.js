(() => {
  // One shared tooltip for every [data-tooltip] element. It lives in the top layer
  // (popover API) or, as a fallback, inside the current top-layer host with position: fixed,
  // so ancestors with overflow, transforms or z-index cannot clip or cover it.
  const SELECTOR = '[data-tooltip]'
  const GAP = 6
  const MARGIN = 8
  const HOVER_DELAY = 120
  const supportsPopover = typeof HTMLElement.prototype.showPopover === 'function'

  const tip = document.createElement('div')
  tip.id = 'ocrsd-tooltip'
  tip.className = 'tooltip'
  tip.setAttribute('role', 'tooltip')
  if (supportsPopover) tip.popover = 'manual'
  else tip.hidden = true

  let target = null
  let describedBy = null
  let timer = 0
  const watcher = new MutationObserver(() => {
    if (!target) return
    if (!target.dataset.tooltip || !visible(target)) hide()
    else { tip.textContent = target.dataset.tooltip; place() }
  })

  const visible = element => element.isConnected && element.getClientRects().length > 0
  // A modal dialog or a fullscreen element hides everything outside it, so the fallback must render inside it.
  const host = element => {
    const full = document.fullscreenElement
    if (full?.contains(element)) return full
    return element.closest('dialog[open]') || document.body
  }

  function place() {
    const box = target.getBoundingClientRect()
    const viewWidth = document.documentElement.clientWidth
    const viewHeight = document.documentElement.clientHeight
    tip.style.left = '0px'
    tip.style.top = '0px'
    const { width, height } = tip.getBoundingClientRect()
    const below = box.bottom + GAP + height <= viewHeight - MARGIN || box.top - GAP - height < MARGIN
    const left = Math.max(MARGIN, Math.min(box.left + box.width / 2 - width / 2, viewWidth - width - MARGIN))
    const top = below ? Math.min(box.bottom + GAP, viewHeight - height - MARGIN) : box.top - GAP - height
    tip.dataset.placement = below ? 'bottom' : 'top'
    tip.style.left = `${Math.round(left)}px`
    tip.style.top = `${Math.round(Math.max(MARGIN, top))}px`
  }

  function show(element) {
    clearTimeout(timer)
    const text = element.dataset.tooltip
    if (!text || !visible(element)) return
    if (target !== element) release()
    target = element
    tip.textContent = text
    const parent = host(element)
    if (tip.parentNode !== parent) {
      if (supportsPopover && tip.matches(':popover-open')) tip.hidePopover()
      parent.append(tip)
    }
    if (supportsPopover) {
      // Reopen so the tooltip goes to the top of the top-layer stack, above a dialog opened after it.
      if (tip.matches(':popover-open')) tip.hidePopover()
      tip.showPopover()
    } else tip.hidden = false
    place()
    if (element.getAttribute('aria-label') !== text) {
      describedBy = element.getAttribute('aria-describedby')
      element.setAttribute('aria-describedby', [describedBy, tip.id].filter(Boolean).join(' '))
    }
    watcher.observe(element, { attributes: true, attributeFilter: ['data-tooltip', 'hidden', 'class', 'style'] })
  }

  function release() {
    if (!target) return
    watcher.disconnect()
    if (target.getAttribute('aria-describedby')?.split(' ').includes(tip.id)) {
      if (describedBy) target.setAttribute('aria-describedby', describedBy)
      else target.removeAttribute('aria-describedby')
    }
    describedBy = null
    target = null
  }

  function hide() {
    clearTimeout(timer)
    release()
    if (supportsPopover) { if (tip.matches(':popover-open')) tip.hidePopover() }
    else tip.hidden = true
  }

  const trigger = node => node?.closest?.(SELECTOR) || null

  document.addEventListener('pointerover', event => {
    if (event.pointerType === 'touch') return
    const element = trigger(event.target)
    if (!element || element === target) return
    clearTimeout(timer)
    if (target) show(element)
    else timer = setTimeout(() => show(element), HOVER_DELAY)
  })
  document.addEventListener('pointerout', event => {
    const element = trigger(event.target)
    if (!element || element.contains(event.relatedTarget)) return
    if (element === target && element.matches(':focus-visible')) return
    if (element === target) hide()
    else clearTimeout(timer)
  })
  document.addEventListener('focusin', event => {
    const element = trigger(event.target)
    if (element && event.target.matches(':focus-visible')) show(element)
  })
  document.addEventListener('focusout', event => {
    if (target && target.contains(event.target) && !target.contains(event.relatedTarget) && !target.matches(':hover')) hide()
  })
  document.addEventListener('pointerdown', hide, true)
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && target) hide() }, true)
  document.addEventListener('scroll', () => { if (target) hide() }, true)
  window.addEventListener('resize', () => { if (target) hide() })
  window.addEventListener('blur', hide)
  document.addEventListener('fullscreenchange', hide)
  document.addEventListener('visibilitychange', hide)
  // HTMX can replace the trigger while its tooltip is open; drop a tooltip whose trigger left the page.
  for (const name of ['htmx:afterSwap', 'htmx:afterSettle', 'htmx:historyRestore']) {
    document.addEventListener(name, () => { if (target && !visible(target)) hide() })
  }
  document.addEventListener('htmx:beforeHistorySave', hide)

  window.OCRSDTooltip = { hide, active: () => target }
})()
