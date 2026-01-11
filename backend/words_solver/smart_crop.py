import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int
    extent: float = 1.0

    @property
    def x0(self) -> int:
        return self.x

    @property
    def y0(self) -> int:
        return self.y

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass(frozen=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def clamp(self, w: int, h: int) -> "BBox":
        x0 = max(0, min(self.x0, w - 1))
        y0 = max(0, min(self.y0, h - 1))
        x1 = max(1, min(self.x1, w))
        y1 = max(1, min(self.y1, h))
        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        return BBox(x0, y0, x1, y1)


def build_white_mask(img_bgr: np.ndarray, s_max: int, v_min: int) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, v_min], dtype=np.uint8)
    upper = np.array([179, s_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Keep it mild. Aggressive close tends to glue tiles together,
    # especially on JPEG where compression creates "almost white" in gaps.
    k3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    return mask


def apply_roi(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def rects_from_mask(mask: np.ndarray) -> List[Rect]:
    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: List[Rect] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        rect_area = float(w * h)
        extent = area / rect_area if rect_area > 0 else 0.0
        rects.append(Rect(int(x), int(y), int(w), int(h), extent=float(extent)))
    return rects


def filter_tile_candidates(rects: Sequence[Rect], img_w: int, img_h: int) -> List[Rect]:
    img_area = float(img_w * img_h)
    out: List[Rect] = []

    for r in rects:
        if r.w <= 0 or r.h <= 0:
            continue

        area = float(r.w * r.h)
        if area < img_area * 0.0003 or area > img_area * 0.08:
            continue

        ar = r.w / float(r.h)
        if ar < 0.70 or ar > 1.35:
            continue

        # Avoid huge UI blocks
        if r.w > img_w * 0.45 or r.h > img_h * 0.45:
            continue

        out.append(r)

    if not out:
        return out

    ws = np.array([r.w for r in out], dtype=np.float32)
    hs = np.array([r.h for r in out], dtype=np.float32)
    mw = float(np.median(ws))
    mh = float(np.median(hs))

    es = np.array([r.extent for r in out], dtype=np.float32)
    me = float(np.median(es))
    extent_thr = max(0.55, me * 0.6)

    keep: List[Rect] = []
    for r in out:
        if mw > 0 and abs(r.w - mw) / mw > 0.40:
            continue
        if mh > 0 and abs(r.h - mh) / mh > 0.40:
            continue
        if r.extent < extent_thr:
            continue
        keep.append(r)

    return keep


def connected_components(rects: Sequence[Rect], max_dist: float) -> List[List[int]]:
    n = len(rects)
    if n == 0:
        return []

    centers = np.array([[r.cx, r.cy] for r in rects], dtype=np.float32)
    max_dist_sq = float(max_dist * max_dist)

    neigh: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(centers[i, 0] - centers[j, 0])
            dy = float(centers[i, 1] - centers[j, 1])
            if dx * dx + dy * dy <= max_dist_sq:
                neigh[i].append(j)
                neigh[j].append(i)

    comps: List[List[int]] = []
    seen = [False] * n
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = [i]
        while stack:
            k = stack.pop()
            for j in neigh[k]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
                    comp.append(j)
        comps.append(comp)

    comps.sort(key=len, reverse=True)
    return comps


def bbox_from_rects(rects: Sequence[Rect]) -> BBox:
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)
    return BBox(int(x0), int(y0), int(x1), int(y1))


def adjust_bbox_multiple_of_5(bbox: BBox, img_w: int, img_h: int) -> BBox:
    # Downstream splitter uses integer division. Keeping bbox sizes divisible
    # by 5 makes every cell the same size (no truncated tail pixels).
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1

    w = x1 - x0
    h = y1 - y0

    if w <= 0 or h <= 0:
        return bbox.clamp(img_w, img_h)

    rw = w % 5
    rh = h % 5

    if rw:
        add = 5 - rw
        if x1 + add <= img_w:
            x1 += add
        elif x0 - add >= 0:
            x0 -= add

    if rh:
        add = 5 - rh
        if y1 + add <= img_h:
            y1 += add
        elif y0 - add >= 0:
            y0 -= add

    return BBox(x0, y0, x1, y1).clamp(img_w, img_h)


def find_board_bbox(
    img_bgr: np.ndarray,
    *,
    roi_top: float = 0.18,
    roi_bottom: float = 0.92,
    s_max: int = 60,
    v_min: int = 200,
    min_tiles: int = 15,
    pad_frac: float = 0.02,
    debug: bool = False,
) -> Tuple[Optional[BBox], Dict[str, float], Optional[np.ndarray], List[Rect], List[Rect]]:
    h, w = img_bgr.shape[:2]

    mask = build_white_mask(img_bgr, s_max=s_max, v_min=v_min)

    y0 = int(h * roi_top)
    y1 = int(h * roi_bottom)
    mask_roi = apply_roi(mask, 0, y0, w, y1)

    all_rects = rects_from_mask(mask_roi)
    candidates = filter_tile_candidates(all_rects, img_w=w, img_h=h)

    if not candidates:
        info = {"candidates": 0.0, "cluster": 0.0}
        return None, info, mask_roi if debug else None, [], []

    mw = float(np.median([r.w for r in candidates]))
    mh = float(np.median([r.h for r in candidates]))
    tile_size = float(max(mw, mh))

    comps = connected_components(candidates, max_dist=tile_size * 2.7)
    if not comps:
        info = {"candidates": float(len(candidates)), "cluster": 0.0}
        return None, info, mask_roi if debug else None, candidates if debug else [], []

    cluster = [candidates[i] for i in comps[0]]

    # If a couple of UI elements slip in, keep the most tile-like ones.
    if len(cluster) > 25:
        cluster.sort(key=lambda r: r.extent, reverse=True)
        cluster = cluster[:25]

    info = {
        "candidates": float(len(candidates)),
        "cluster": float(len(cluster)),
        "tile_size": float(tile_size),
    }

    if len(cluster) < min_tiles:
        return (
            None,
            info,
            mask_roi if debug else None,
            candidates if debug else [],
            cluster if debug else [],
        )

    bbox = bbox_from_rects(cluster).clamp(w, h)

    # Small pad to avoid shaving off tile borders/corners.
    pad = int(round(tile_size * pad_frac))
    if pad > 0:
        bbox = BBox(bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad).clamp(w, h)

    bbox = adjust_bbox_multiple_of_5(bbox, img_w=w, img_h=h)

    if not debug:
        return bbox, info, None, [], []

    return bbox, info, mask_roi, candidates, cluster
