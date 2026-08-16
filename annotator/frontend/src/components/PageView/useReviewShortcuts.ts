import { useEffect } from 'react'

/**
 * Fast-review keyboard shortcuts: 1/0 confirm-reject the selected box,
 * arrow/Tab moves selection, E focuses the character input. Disabled
 * while any input/textarea is focused so typing a character never
 * triggers a shortcut.
 */
export function useReviewShortcuts({
  boxIds,
  selectedBoxId,
  onSelect,
  onConfirm,
  onReject,
  onFocusCharacterInput,
}: {
  boxIds: number[]
  selectedBoxId: number | null
  onSelect: (boxId: number) => void
  onConfirm: () => void
  onReject: () => void
  onFocusCharacterInput: () => void
}) {
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const target = e.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
      if (boxIds.length === 0 || selectedBoxId === null) return

      const idx = boxIds.indexOf(selectedBoxId)

      if (e.key === '1') {
        onConfirm()
      } else if (e.key === '0') {
        onReject()
      } else if (e.key === 'e' || e.key === 'E') {
        onFocusCharacterInput()
      } else if (e.key === 'Tab' || e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault()
        onSelect(boxIds[(idx + 1 + boxIds.length) % boxIds.length])
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault()
        onSelect(boxIds[(idx - 1 + boxIds.length) % boxIds.length])
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [boxIds, selectedBoxId, onSelect, onConfirm, onReject, onFocusCharacterInput])
}
