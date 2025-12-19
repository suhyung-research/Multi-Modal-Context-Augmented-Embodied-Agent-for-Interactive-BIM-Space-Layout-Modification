# utils.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

Point = Tuple[float, float]


def _as_point(p: Any, scale: float = 1.0) -> Point:
    return (float(p[0]) * scale, float(p[1]) * scale)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _stitch_room_loop(
    elements_by_id: Dict[int, Dict[str, Any]],
    boundary_ids: List[int],
    scale: float,
    tol: float = 1e-6,
) -> Optional[List[Point]]:
    """
    boundary_ids 순서대로 선분을 이어 닫힌 루프(폴리곤)를 만들려고 시도.
    - 방향이 반대면 자동으로 뒤집어 연결
    - 연결이 깨지면 None
    """
    if not boundary_ids:
        return None

    first = elements_by_id.get(boundary_ids[0])
    if not first:
        return None

    s0 = _as_point(first["start_point"], scale)
    t0 = _as_point(first["end_point"], scale)
    poly = [s0, t0]
    curr_end = t0

    for eid in boundary_ids[1:]:
        e = elements_by_id.get(eid)
        if not e:
            return None

        s = _as_point(e["start_point"], scale)
        t = _as_point(e["end_point"], scale)

        if _dist(s, curr_end) <= tol:
            poly.append(t)
            curr_end = t
        elif _dist(t, curr_end) <= tol:
            poly.append(s)
            curr_end = s
        else:
            return None

    return poly


def _polygon_centroid(poly: List[Point]) -> Point:
    # 단순 polygon centroid (면적 계산은 안 하고, 중심점만)
    if len(poly) < 3:
        xs = [p[0] for p in poly] or [0.0]
        ys = [p[1] for p in poly] or [0.0]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    pts = poly[:]
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    A = 0.0
    Cx = 0.0
    Cy = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross

    if abs(A) < 1e-12:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    A *= 0.5
    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return (Cx, Cy)


def _fallback_room_label_pos(
    elements_by_id: Dict[int, Dict[str, Any]],
    boundary_ids: List[int],
    scale: float,
) -> Point:
    pts: List[Point] = []
    for eid in boundary_ids:
        e = elements_by_id.get(eid)
        if not e:
            continue
        pts.append(_as_point(e["start_point"], scale))
        pts.append(_as_point(e["end_point"], scale))
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def render_floorplan_png(
    floorplan: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    scale: float = 1.0,              # mm->m: 0.001
    label_dx_ratio: float = 0.06,    # element label x offset ratio
    label_dy_ratio: float = 0.02,    # element label y offset ratio
    fill_rooms: bool = True,
    dpi: int = 200,
    room_fontsize: int = 7,         # ✅ room label 크기만 줄임
    element_fontsize: int = 7,
    line_width: float = 2.5,
    room_fill_color: str = "0.85",  # room fill 고정색 (색상 사이클 영향 없음)
) -> None:
    """
    floorplan schema:
      {
        "Elements": [{"element_id": int, "start_point":[x,y], "end_point":[x,y]}, ...],
        "Rooms": [{"room_id": int, "room_name": str, "bounding_elements_ids":[int,...]}, ...]
      }

    - room label: room_name + (room_id) 만 표시 (면적 X)
    - element label: element_id 만 표시 (길이 X)
    - element 선/라벨 색상 통일: 둘 다 color=c 사용
    """
    out_path = Path(out_path)

    elements: List[Dict[str, Any]] = floorplan.get("Elements", [])
    rooms: List[Dict[str, Any]] = floorplan.get("Rooms", [])

    elements_by_id: Dict[int, Dict[str, Any]] = {}
    xs: List[float] = []
    ys: List[float] = []

    for e in elements:
        eid = int(e["element_id"])
        elements_by_id[eid] = e
        s = _as_point(e["start_point"], scale)
        t = _as_point(e["end_point"], scale)
        xs.extend([s[0], t[0]])
        ys.extend([s[1], t[1]])

    if not xs or not ys:
        raise ValueError("No elements/points found in floorplan.")

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = maxx - minx
    spany = maxy - miny

    dx = label_dx_ratio * spanx
    dy = label_dy_ratio * spany

    fig, ax = plt.subplots()

    # -----------------------------
    # ✅ [색상 매핑 정의] element_id -> color
    # 여기서 결정된 색(c)을 아래 ax.plot/ax.text 둘 다에 넣어서 "완전 동일" 보장
    # -----------------------------
    cmap = plt.get_cmap("tab20")
    unique_eids = sorted({int(e["element_id"]) for e in elements})
    eid_to_color = {eid: cmap(i % cmap.N) for i, eid in enumerate(unique_eids)}

    # ---- Rooms: fill + label(면적 표시 X) ----
    for r in rooms:
        rid = r.get("room_id")
        rname = r.get("room_name", "")
        boundary = [int(x) for x in r.get("bounding_elements_ids", [])]

        poly = _stitch_room_loop(elements_by_id, boundary, scale=scale, tol=1e-6)

        if fill_rooms and poly and len(poly) >= 3:
            ax.fill(
                [p[0] for p in poly],
                [p[1] for p in poly],
                color=room_fill_color,
                alpha=0.10,
            )

        # room label을 bounding box 중심에 배치
        if poly and len(poly) >= 3:
            xs_room = [p[0] for p in poly]
            ys_room = [p[1] for p in poly]
            cx = (min(xs_room) + max(xs_room)) / 2.0
            cy = (min(ys_room) + max(ys_room)) / 2.0
        else:
            cx, cy = _fallback_room_label_pos(elements_by_id, boundary, scale=scale)

        ax.text(
            cx, cy,
            f"{rname}\n({rid})",
            ha="center", va="center",
            fontsize=room_fontsize,
        )

    # ---- Elements: draw + label(길이 표시 X, 색상 통일) ----
    for e in elements:
        eid = int(e["element_id"])
        s = _as_point(e["start_point"], scale)
        t = _as_point(e["end_point"], scale)

        c = eid_to_color[eid]  # ✅ 이 c가 선/라벨 색상 "같은 원천"

        # ✅ [색상 적용 1] element 선 색
        ax.plot([s[0], t[0]], [s[1], t[1]], linewidth=line_width, color=c)

        mx, my = (s[0] + t[0]) / 2.0, (s[1] + t[1]) / 2.0
        label = f"{eid}"  # 길이 표시 제거

        # ✅ [색상 적용 2] element 라벨 색 (선과 동일)
        ax.text(
            mx + dx, my + dy,
            label,
            ha="center", va="center",
            fontsize=element_fontsize,
            color=c,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(minx - 0.12 * spanx, maxx + 0.12 * spanx)
    ax.set_ylim(miny - 0.12 * spany, maxy + 0.12 * spany)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_floorplan_png_from_json(
    json_path: Union[str, Path],
    out_png_path: Union[str, Path],
    *,
    scale: float = 1.0,
    label_dx_ratio: float = 0.06,
    label_dy_ratio: float = 0.02,
    fill_rooms: bool = True,
    dpi: int = 200,
    room_fontsize: int = 7,
    element_fontsize: int = 7,
    line_width: float = 2.5,
    room_fill_color: str = "0.85",
) -> None:
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        floorplan = json.load(f)

    render_floorplan_png(
        floorplan,
        out_png_path,
        scale=scale,
        label_dx_ratio=label_dx_ratio,
        label_dy_ratio=label_dy_ratio,
        fill_rooms=fill_rooms,
        dpi=dpi,
        room_fontsize=room_fontsize,
        element_fontsize=element_fontsize,
        line_width=line_width,
        room_fill_color=room_fill_color,
    )
