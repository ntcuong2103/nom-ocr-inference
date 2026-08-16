import type { Box, PageDetail } from '../../types'
import './PageImageCanvas.css'

export function PageImageCanvas({
  page,
  selectedBoxId,
  onSelectBox,
}: {
  page: PageDetail
  selectedBoxId: number | null
  onSelectBox: (boxId: number) => void
}) {
  const { image_width: W, image_height: H } = page

  return (
    <div className="canvas-wrap">
      <img src={page.image_url} alt={page.page} width={W} height={H} />
      <svg
        className="overlay"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
      >
        {page.boxes.map((box: Box) => {
          const bw = box.w * W
          const bh = box.h * H
          const bx = box.x * W - bw / 2
          const by = box.y * H - bh / 2
          const selected = box.box_id === selectedBoxId
          const cls = [
            'box',
            box.selection_flag === 1 ? 'confirmed' : 'unconfirmed',
            selected ? 'selected' : '',
          ]
            .filter(Boolean)
            .join(' ')
          return (
            <rect
              key={box.box_id}
              className={cls}
              x={bx}
              y={by}
              width={bw}
              height={bh}
              onClick={() => onSelectBox(box.box_id)}
            />
          )
        })}
      </svg>
    </div>
  )
}
