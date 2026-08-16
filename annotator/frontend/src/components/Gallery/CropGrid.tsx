import type { CropItem } from '../../types'
import { CropCard } from './CropCard'
import './CropGrid.css'

export function CropGrid({ items }: { items: CropItem[] }) {
  if (items.length === 0) {
    return <p className="crop-grid-empty">No crops match this filter.</p>
  }
  return (
    <div className="crop-grid">
      {items.map((item) => (
        <CropCard key={`${item.volume}/${item.page}/${item.box_id}`} item={item} />
      ))}
    </div>
  )
}
