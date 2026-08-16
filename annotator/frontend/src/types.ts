export interface Box {
  box_id: number
  character: string
  x: number
  y: number
  w: number
  h: number
  selection_flag: 0 | 1
}

export interface BoxPatch {
  character?: string
  selection_flag?: 0 | 1
}

export interface PageDetail {
  volume: string
  page: string
  image_url: string
  image_width: number
  image_height: number
  edited: boolean
  boxes: Box[]
}

export interface VolumeSummary {
  volume: string
  num_pages: number
  num_boxes: number
  num_confirmed: number
  num_unconfirmed: number
}

export interface PageSummary {
  page: string
  num_boxes: number
  num_confirmed: number
  num_unconfirmed: number
  edited: boolean
}

export interface CropItem {
  volume: string
  page: string
  box_id: number
  character: string
  selection_flag: 0 | 1
  crop_url: string
}

export interface CropsPage {
  total: number
  limit: number
  offset: number
  items: CropItem[]
}

export interface CharacterCount {
  character: string
  count: number
}

export interface CropsFilter {
  selection_flag?: 0 | 1
  character?: string
  volume?: string
  page?: string
  limit: number
  offset: number
}
