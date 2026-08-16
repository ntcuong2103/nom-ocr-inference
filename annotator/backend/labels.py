"""Parsing/formatting for pseudo-label lines.

Format: ``<character> <x> <y> <w> <h> <selection_flag> <box_id>``
x/y/w/h are YOLO-normalized (center x, center y, width, height).

The character field may be an empty string (a detected-but-unlabeled box).
A naive ``line.split()`` silently drops that leading empty field and
misaligns every subsequent value, so this parser splits on the literal
single space delimiter and reconstructs the character from all leading
fields instead.
"""

from dataclasses import dataclass


@dataclass
class Box:
    box_id: int
    line_index: int
    character: str
    x: float
    y: float
    w: float
    h: float
    selection_flag: int


def parse_label_line(raw: str, line_index: int) -> Box | None:
    parts = raw.rstrip("\n").split(" ")
    if len(parts) < 6:
        return None
    *char_parts, x_s, y_s, w_s, h_s, flag_s, box_id_s = parts
    try:
        x, y, w, h = float(x_s), float(y_s), float(w_s), float(h_s)
        selection_flag = int(flag_s)
        box_id = int(box_id_s)
    except ValueError:
        return None
    character = " ".join(char_parts)
    return Box(
        box_id=box_id,
        line_index=line_index,
        character=character,
        x=x,
        y=y,
        w=w,
        h=h,
        selection_flag=selection_flag,
    )


def format_label_line(box: Box) -> str:
    return (
        f"{box.character} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f} "
        f"{box.selection_flag} {box.box_id}\n"
    )


def parse_label_file(path) -> list[Box]:
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            box = parse_label_line(line, i)
            if box is not None:
                boxes.append(box)
    return boxes
