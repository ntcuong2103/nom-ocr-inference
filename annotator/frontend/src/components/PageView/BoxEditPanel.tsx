import { useEffect, useRef, useState } from 'react'
import type { Box, BoxPatch } from '../../types'
import './BoxEditPanel.css'

export function BoxEditPanel({
  box,
  saving,
  onSave,
  focusToken,
}: {
  box: Box | null
  saving: boolean
  onSave: (patch: BoxPatch) => void
  focusToken?: number
}) {
  const [character, setCharacter] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setCharacter(box?.character ?? '')
  }, [box?.box_id, box?.character])

  useEffect(() => {
    if (focusToken !== undefined) {
      inputRef.current?.focus()
      inputRef.current?.select()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusToken])

  if (!box) {
    return (
      <div className="box-edit-panel empty">
        <p>Click a box on the page to review it.</p>
      </div>
    )
  }

  const commitCharacter = () => {
    const trimmed = character.trim()
    if (trimmed !== box.character) {
      onSave({ character: trimmed })
    }
  }

  return (
    <div className="box-edit-panel">
      <h3>Box #{box.box_id}</h3>
      <label className="field">
        <span>Character</span>
        <input
          ref={inputRef}
          className="nom char-input"
          value={character}
          onChange={(e) => setCharacter(e.target.value)}
          onBlur={commitCharacter}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.currentTarget.blur()
            }
          }}
        />
      </label>
      <div className="field">
        <span>Status</span>
        <div className="flag-toggle">
          <button
            className={box.selection_flag === 1 ? 'active confirm' : 'confirm'}
            onClick={() => onSave({ selection_flag: 1 })}
          >
            Confirm
          </button>
          <button
            className={box.selection_flag === 0 ? 'active reject' : 'reject'}
            onClick={() => onSave({ selection_flag: 0 })}
          >
            Unconfirmed
          </button>
        </div>
      </div>
      {saving && <p className="saving">Saving…</p>}
      <dl className="geometry">
        <dt>x, y</dt>
        <dd>
          {box.x.toFixed(4)}, {box.y.toFixed(4)}
        </dd>
        <dt>w, h</dt>
        <dd>
          {box.w.toFixed(4)}, {box.h.toFixed(4)}
        </dd>
      </dl>
    </div>
  )
}
