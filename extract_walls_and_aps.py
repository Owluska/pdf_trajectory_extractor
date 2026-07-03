#!/usr/bin/env python3
"""Extract united wall curves and AP positions from a PDF plan.

Usage:
    python extract_walls_and_aps.py --config config.yaml

Final output_dir contains only:
    - wall_curves.csv
    - ap_positions.csv

Intermediate renders/masks/diagnostics are written to processing_dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import pymupdf as fitz  # PyMuPDF
import numpy as np
import yaml
from sklearn.cluster import DBSCAN
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract wall curves and AP positions from a PDF plan.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config is empty or invalid: {path}")
    return cfg


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def prepare_dir(path: Path, clean: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not clean:
        return
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def color_close(c: Sequence[float] | None, target: Sequence[float], tol: float) -> bool:
    return c is not None and max(abs(float(c[i]) - float(target[i])) for i in range(3)) <= tol


def rect_intersects(r: fitz.Rect, roi: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = roi
    return r.x1 >= x0 and r.x0 <= x1 and r.y1 >= y0 and r.y0 <= y1


def path_points_from_item(item: tuple, scale: float = 1.0, n_curve: int = 32) -> List[Point]:
    typ = item[0]
    if typ == "l":
        return [(item[1].x * scale, item[1].y * scale), (item[2].x * scale, item[2].y * scale)]
    if typ == "c":
        p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
        pts: List[Point] = []
        for i in range(n_curve + 1):
            t = i / n_curve
            x = (
                (1 - t) ** 3 * p0.x
                + 3 * (1 - t) ** 2 * t * p1.x
                + 3 * (1 - t) * t * t * p2.x
                + t**3 * p3.x
            )
            y = (
                (1 - t) ** 3 * p0.y
                + 3 * (1 - t) ** 2 * t * p1.y
                + 3 * (1 - t) * t * t * p2.y
                + t**3 * p3.y
            )
            pts.append((x * scale, y * scale))
        return pts
    if typ == "re":
        r = item[1]
        return [
            (r.x0 * scale, r.y0 * scale),
            (r.x1 * scale, r.y0 * scale),
            (r.x1 * scale, r.y1 * scale),
            (r.x0 * scale, r.y1 * scale),
            (r.x0 * scale, r.y0 * scale),
        ]
    if typ == "qu":
        try:
            return [(p.x * scale, p.y * scale) for p in item[1]]
        except Exception:
            return []
    return []


def polyline_length(pts: Sequence[Point]) -> float:
    if len(pts) < 2:
        return 0.0
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))


def draw_polyline(mask: np.ndarray, pts_px: Sequence[Point], width_px: int) -> None:
    if len(pts_px) < 2:
        return
    pts_i = np.array([[int(round(x)), int(round(y))] for x, y in pts_px], dtype=np.int32)
    cv2.polylines(mask, [pts_i], False, 255, width_px, cv2.LINE_AA)


def point_dist(a: Point, b: Point) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def angle_deg(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0])))


def angle_diff_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def collect_line_graph_points(items: Sequence[tuple], tol: float) -> Tuple[List[Point], List[Tuple[int, int]]]:
    """Merge endpoints and return graph vertices/edges for line-only drawing items."""
    pts: List[Point] = []
    edges: List[Tuple[int, int]] = []

    def add_point(p: Point) -> int:
        for i, q in enumerate(pts):
            if point_dist(p, q) <= tol:
                return i
        pts.append((float(p[0]), float(p[1])))
        return len(pts) - 1

    for item in items:
        if item[0] != "l":
            continue
        ia = add_point((item[1].x, item[1].y))
        ib = add_point((item[2].x, item[2].y))
        if ia != ib:
            edges.append((ia, ib))
    return pts, edges


def is_three_point_footnote_leader(d: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """Detect compact 3-point leader marks: short shelf + inclined foot.

    These are black vector paths just like walls, so they are removed before
    raster closing/united wall-curve tracing.
    """
    if not flt.get("enabled", True):
        return False
    if d.get("fill") is not None:
        return False

    col = d.get("color")
    if col is None or max(float(v) for v in col) > float(flt["black_max_channel"]):
        return False

    width = float(d.get("width") or 0.0)
    if not (float(flt["stroke_width_pdf_pt_min"]) <= width <= float(flt["stroke_width_pdf_pt_max"])):
        return False

    items = d.get("items", [])
    line_item_count = sum(1 for it in items if it[0] == "l")
    if line_item_count != int(flt["required_line_item_count"]):
        return False
    if any(it[0] != "l" for it in items):
        return False

    pts, edges = collect_line_graph_points(items, tol=float(flt["point_merge_tol_pdf_pt"]))
    if len(pts) != int(flt["required_vertex_count"]) or len(edges) != 2:
        return False

    deg = [0] * len(pts)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    if sorted(deg) != [1, 1, 2]:
        return False

    mid = deg.index(2)
    ends = [i for i in range(3) if i != mid]
    p_mid = pts[mid]
    p0 = pts[ends[0]]
    p1 = pts[ends[1]]

    len0 = point_dist(p_mid, p0)
    len1 = point_dist(p_mid, p1)
    max_len = max(len0, len1)
    min_len = min(len0, len1)
    r = d["rect"]
    bbox_span = max(r.width, r.height)

    if min_len < float(flt["branch_length_min_pdf_pt"]):
        return False
    if min_len > float(flt["short_branch_length_max_pdf_pt"]):
        return False
    if max_len > float(flt["branch_length_max_pdf_pt"]):
        return False
    if bbox_span > float(flt["bbox_span_max_pdf_pt"]):
        return False

    a0 = angle_deg(p_mid, p0)
    a1 = angle_deg(p_mid, p1)
    bend = angle_diff_deg(a0, a1)
    if not (float(flt["bend_angle_deg_min"]) <= bend <= float(flt["bend_angle_deg_max"])):
        return False

    def dist_to_horizontal(a: float) -> float:
        return min(angle_diff_deg(a, 0.0), angle_diff_deg(a, 180.0))

    h0 = dist_to_horizontal(a0)
    h1 = dist_to_horizontal(a1)
    has_shelf = min(h0, h1) <= float(flt["shelf_angle_to_horizontal_max_deg"])
    diag_angle = a1 if h0 <= h1 else a0
    diag_from_horizontal = dist_to_horizontal(diag_angle)
    has_45deg_foot = (
        float(flt["foot_angle_to_horizontal_min_deg"])
        <= diag_from_horizontal
        <= float(flt["foot_angle_to_horizontal_max_deg"])
    )
    return has_shelf and has_45deg_foot


def component_keep_decision(rp: Any, union_cfg: Dict[str, Any]) -> bool:
    minr, minc, maxr, maxc = rp.bbox
    w = maxc - minc
    h = maxr - minr
    span = max(w, h)
    area = float(rp.area)
    slender = (w >= 3 * max(h, 1)) or (h >= 3 * max(w, 1))

    if area >= float(union_cfg["min_component_area_px"]) and span >= float(union_cfg["min_component_span_px"]):
        return True
    if slender and area >= float(union_cfg["min_slender_component_area_px"]) and span >= float(union_cfg["min_slender_component_span_px"]):
        return True
    if span >= float(union_cfg["min_long_component_span_px"]) and area >= float(union_cfg["min_long_component_area_px"]):
        return True
    return False


def trace_skeleton_paths(skel_bool: np.ndarray, min_curve_length_px: float, simplify_eps_px: float) -> List[Tuple[np.ndarray, float]]:
    coords = set(map(tuple, np.argwhere(skel_bool)))
    if not coords:
        return []

    def nbrs(p: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
        y, x = p
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (y + dy, x + dx)
                if q in coords:
                    yield q

    deg = {p: sum(1 for _ in nbrs(p)) for p in coords}
    nodes = {p for p, d in deg.items() if d != 2}
    visited_edges = set()
    paths: List[List[Tuple[int, int]]] = []

    def edge_key(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return tuple(sorted([a, b]))  # type: ignore[return-value]

    # Trace paths from junctions/endpoints.
    for node in list(nodes):
        for nb in nbrs(node):
            e = edge_key(node, nb)
            if e in visited_edges:
                continue
            path = [node, nb]
            visited_edges.add(e)
            prev, cur = node, nb
            while cur not in nodes:
                candidates = [q for q in nbrs(cur) if q != prev]
                if not candidates:
                    break
                nxt = candidates[0]
                e = edge_key(cur, nxt)
                if e in visited_edges:
                    break
                visited_edges.add(e)
                path.append(nxt)
                prev, cur = cur, nxt
            paths.append(path)

    # Trace remaining cycles.
    for p in list(coords):
        for nb in nbrs(p):
            e = edge_key(p, nb)
            if e in visited_edges:
                continue
            path = [p, nb]
            visited_edges.add(e)
            prev, cur = p, nb
            guard = 0
            while guard < 100000:
                guard += 1
                candidates = [q for q in nbrs(cur) if q != prev]
                if not candidates:
                    break
                nxt = candidates[0]
                e = edge_key(cur, nxt)
                if e in visited_edges:
                    break
                visited_edges.add(e)
                path.append(nxt)
                prev, cur = cur, nxt
            paths.append(path)

    out: List[Tuple[np.ndarray, float]] = []
    for path in paths:
        if len(path) < 2:
            continue
        # Convert (row, col) -> (x, y).
        xy = np.array([[float(x), float(y)] for y, x in path], dtype=np.float32)
        length = sum(float(np.linalg.norm(b - a)) for a, b in zip(xy, xy[1:]))
        if length < min_curve_length_px:
            continue
        approx = cv2.approxPolyDP(xy.reshape(-1, 1, 2), epsilon=simplify_eps_px, closed=False).reshape(-1, 2)
        if len(approx) < 2:
            continue
        approx_len = sum(float(np.linalg.norm(b - a)) for a, b in zip(approx, approx[1:]))
        if approx_len < min_curve_length_px:
            continue
        out.append((approx, approx_len))
    return out


def extract_ap_positions(drawings: Sequence[Dict[str, Any]], roi_pdf: Tuple[float, float, float, float], dpi_scale: float, ap_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    color_tol = float(ap_cfg["color_tolerance"])
    aps: List[Dict[str, Any]] = []

    for ap_type, target in ap_cfg["colors_rgb_0_1"].items():
        pts: List[List[float]] = []
        rects: List[Tuple[float, float, float, float]] = []
        for d in drawings:
            if not (color_close(d.get("color"), target, color_tol) or color_close(d.get("fill"), target, color_tol)):
                continue
            r = d["rect"]
            if not rect_intersects(r, roi_pdf):
                continue
            pts.append([(r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2])
            rects.append((r.x0, r.y0, r.x1, r.y1))

        if not pts:
            continue

        labels = DBSCAN(eps=float(ap_cfg["dbscan_eps_pdf_pt"]), min_samples=int(ap_cfg["dbscan_min_samples"])).fit_predict(np.array(pts))
        for lab_id in sorted(set(labels)):
            if lab_id == -1:
                continue
            inds = np.where(labels == lab_id)[0]
            xs: List[float] = []
            ys: List[float] = []
            for idx in inds:
                ax0, ay0, ax1, ay1 = rects[idx]
                xs += [ax0, ax1]
                ys += [ay0, ay1]

            bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
            bw, bh = bx1 - bx0, by1 - by0
            if len(inds) < int(ap_cfg["min_cluster_items"]):
                continue
            if not (float(ap_cfg["bbox_width_pdf_pt_min"]) <= bw <= float(ap_cfg["bbox_width_pdf_pt_max"])):
                continue
            if not (float(ap_cfg["bbox_height_pdf_pt_min"]) <= bh <= float(ap_cfg["bbox_height_pdf_pt_max"])):
                continue

            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            aps.append(
                {
                    "id": None,
                    "type": ap_type,
                    "x_pdf_pt": round(cx, 3),
                    "y_pdf_pt": round(cy, 3),
                    "x_px": round(cx * dpi_scale, 1),
                    "y_px": round(cy * dpi_scale, 1),
                    "bbox_pdf_pt": [round(bx0, 3), round(by0, 3), round(bx1, 3), round(by1, 3)],
                    "cluster_items": int(len(inds)),
                }
            )

    aps.sort(key=lambda a: (a["y_pdf_pt"], a["x_pdf_pt"], a["type"]))
    for i, a in enumerate(aps, 1):
        a["id"] = f"AP_{i:03d}"
    return aps


def write_ap_positions_csv(path: Path, aps: Sequence[Dict[str, Any]]) -> None:
    fields = ["id", "type", "x_pdf_pt", "y_pdf_pt", "x_px", "y_px", "bbox_pdf_pt", "cluster_items"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ap in aps:
            row = dict(ap)
            row["bbox_pdf_pt"] = json.dumps(row["bbox_pdf_pt"], ensure_ascii=False)
            writer.writerow(row)


def write_wall_curves_csv(path: Path, wall_curves: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "curve_id",
                "vertex_index",
                "x_pdf_pt",
                "y_pdf_pt",
                "x_px",
                "y_px",
                "curve_length_pdf_pt",
                "curve_length_px",
            ]
        )
        for c in wall_curves:
            for j, (pp, px) in enumerate(zip(c["points_pdf_pt"], c["points_px"])):
                writer.writerow([c["id"], j, pp[0], pp[1], px[0], px[1], c["length_pdf_pt"], c["length_px"]])


def save_debug_images(processing_dir: Path, roi_px: Tuple[int, int, int, int], raw_mask: np.ndarray, clean_mask_bool: np.ndarray, skeleton: np.ndarray, wall_curves: Sequence[Dict[str, Any]]) -> None:
    x0, y0, x1, y1 = roi_px
    cv2.imwrite(str(processing_dir / "raw_wall_candidates_mask.png"), raw_mask)
    cv2.imwrite(str(processing_dir / "united_wall_mask.png"), (clean_mask_bool.astype(np.uint8) * 255))
    cv2.imwrite(str(processing_dir / "wall_skeleton.png"), (skeleton.astype(np.uint8) * 255))

    curve_vis = np.full((raw_mask.shape[0], raw_mask.shape[1], 3), 255, np.uint8)
    for c in wall_curves:
        pts = np.array([[int(round(px)), int(round(py))] for px, py in c["points_px"]], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(curve_vis, [pts], False, (255, 170, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(processing_dir / "wall_curves_preview_crop.png"), curve_vis[y0:y1, x0:x1])


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(config_path)
    base_dir = config_path.parent

    pdf_path = resolve_path(cfg["input"]["pdf_path"], base_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    processing_dir = resolve_path(cfg["folders"]["processing_dir"], base_dir)
    output_dir = resolve_path(cfg["folders"]["output_dir"], base_dir)
    prepare_dir(processing_dir, clean=bool(cfg["folders"].get("clean_processing_dir", False)))
    prepare_dir(output_dir, clean=bool(cfg["folders"].get("clean_output_dir", False)))

    dpi = int(cfg["render"]["dpi"])
    scale = dpi / 72.0
    roi_cfg = cfg["plan_roi_pdf_pt"]
    roi_pdf = (float(roi_cfg["x0"]), float(roi_cfg["y0"]), float(roi_cfg["x1"]), float(roi_cfg["y1"]))
    roi_px = tuple(int(round(v * scale)) for v in roi_pdf)

    doc = fitz.open(pdf_path)
    page_index = int(cfg["input"].get("page_index", 0))
    page = doc[page_index]
    drawings = page.get_drawings()

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    render_path = processing_dir / f"render_page_{page_index + 1}_{dpi}dpi.png"
    pix.save(str(render_path))
    img = cv2.imread(str(render_path))
    if img is None:
        raise RuntimeError(f"Could not read rendered page: {render_path}")
    height_px, width_px = img.shape[:2]

    ap_positions = extract_ap_positions(drawings, roi_pdf, scale, cfg["ap_extraction"])

    footnote_indices = {
        i
        for i, d in enumerate(drawings)
        if rect_intersects(d["rect"], roi_pdf) and is_three_point_footnote_leader(d, cfg["footnote_leader_filter"])
    }

    wall_cfg = cfg["wall_extraction"]
    raw_wall_mask = np.zeros((height_px, width_px), np.uint8)
    x0_pdf, y0_pdf, x1_pdf, y1_pdf = roi_pdf
    raw_wall_candidate_count = 0
    for draw_idx, d in enumerate(drawings):
        if draw_idx in footnote_indices:
            continue
        col = d.get("color")
        width = float(d.get("width") or 0.0)
        if not rect_intersects(d["rect"], roi_pdf):
            continue
        if col is None or max(float(v) for v in col) > float(wall_cfg["black_max_channel"]):
            continue
        if width < float(wall_cfg["min_stroke_width_pdf_pt"]):
            continue

        for item in d.get("items", []):
            pts_pdf = path_points_from_item(item, 1.0, n_curve=int(wall_cfg["curve_sampling_points"]))
            if len(pts_pdf) < 2:
                continue
            xs = [p[0] for p in pts_pdf]
            ys = [p[1] for p in pts_pdf]
            if max(xs) < x0_pdf or min(xs) > x1_pdf or max(ys) < y0_pdf or min(ys) > y1_pdf:
                continue
            if polyline_length(pts_pdf) < float(wall_cfg["min_polyline_length_pdf_pt"]):
                continue
            pts_px = [(x * scale, y * scale) for x, y in pts_pdf]
            draw_w = max(
                int(wall_cfg["min_draw_width_px"]),
                int(round(max(width, 0.08) * scale * float(wall_cfg["draw_width_scale"]))),
            )
            draw_polyline(raw_wall_mask, pts_px, draw_w)
            raw_wall_candidate_count += 1

    # Clip raw mask to ROI and suppress configurable border bands.
    x0, y0, x1, y1 = roi_px
    roi_only = np.zeros_like(raw_wall_mask)
    roi_only[y0:y1, x0:x1] = raw_wall_mask[y0:y1, x0:x1]
    suppress = wall_cfg["suppress_border_px"]
    if int(suppress.get("top", 0)) > 0:
        roi_only[y0 : y0 + int(suppress["top"]), x0:x1] = 0
    if int(suppress.get("bottom", 0)) > 0:
        roi_only[y1 - int(suppress["bottom"]) : y1, x0:x1] = 0
    if int(suppress.get("left", 0)) > 0:
        roi_only[y0:y1, x0 : x0 + int(suppress["left"])] = 0
    if int(suppress.get("right", 0)) > 0:
        roi_only[y0:y1, x1 - int(suppress["right"]) : x1] = 0
    raw_wall_mask = roi_only

    union_cfg = cfg["wall_union"]
    kernel_size = int(union_cfg["close_kernel_px"])
    closed = cv2.morphologyEx(
        raw_wall_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=int(union_cfg["close_iterations"]),
    ) > 0

    lab = label(closed, connectivity=2)
    clean_wall_mask_bool = np.zeros_like(closed, dtype=bool)
    component_rows: List[Dict[str, Any]] = []
    for rp in regionprops(lab):
        minr, minc, maxr, maxc = rp.bbox
        keep = component_keep_decision(rp, union_cfg)
        if keep:
            clean_wall_mask_bool[lab == rp.label] = True
        component_rows.append(
            {
                "component_id": int(rp.label),
                "keep": bool(keep),
                "area_px": int(rp.area),
                "bbox_px": [int(minc), int(minr), int(maxc), int(maxr)],
                "width_px": int(maxc - minc),
                "height_px": int(maxr - minr),
                "span_px": int(max(maxc - minc, maxr - minr)),
            }
        )

    wall_skeleton = skeletonize(clean_wall_mask_bool)
    curve_paths = trace_skeleton_paths(
        wall_skeleton,
        min_curve_length_px=float(union_cfg["min_curve_length_px"]),
        simplify_eps_px=float(union_cfg["curve_simplify_epsilon_px"]),
    )

    wall_curves: List[Dict[str, Any]] = []
    for approx, length_px in curve_paths:
        pts_px = [[round(float(x), 1), round(float(y), 1)] for x, y in approx]
        pts_pdf = [[round(float(x / scale), 3), round(float(y / scale), 3)] for x, y in approx]
        wall_curves.append(
            {
                "id": None,
                "length_px": round(length_px, 1),
                "length_pdf_pt": round(length_px / scale, 3),
                "points_pdf_pt": pts_pdf,
                "points_px": pts_px,
            }
        )

    wall_curves.sort(key=lambda c: (c["points_pdf_pt"][0][1], c["points_pdf_pt"][0][0]))
    for i, c in enumerate(wall_curves, 1):
        c["id"] = f"WC_{i:04d}"

    # Final outputs only.
    write_wall_curves_csv(output_dir / cfg["output"]["wall_curves_csv"], wall_curves)
    write_ap_positions_csv(output_dir / cfg["output"]["ap_positions_csv"], ap_positions)

    # Processing diagnostics only.
    if bool(cfg.get("debug", {}).get("save_processing_images", False)):
        save_debug_images(processing_dir, roi_px, raw_wall_mask, clean_wall_mask_bool, wall_skeleton, wall_curves)

    if bool(cfg.get("debug", {}).get("save_removed_footnote_leaders_csv", False)):
        with (processing_dir / "removed_footnote_leaders.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["drawing_index", "x0_pdf_pt", "y0_pdf_pt", "x1_pdf_pt", "y1_pdf_pt", "line_item_count"])
            for idx in sorted(footnote_indices):
                r = drawings[idx]["rect"]
                writer.writerow([idx, round(r.x0, 3), round(r.y0, 3), round(r.x1, 3), round(r.y1, 3), len(drawings[idx].get("items", []))])

    if bool(cfg.get("debug", {}).get("save_component_diagnostics_csv", False)):
        with (processing_dir / "wall_components.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["component_id", "keep", "area_px", "bbox_px", "width_px", "height_px", "span_px"])
            writer.writeheader()
            for row in component_rows:
                out = dict(row)
                out["bbox_px"] = json.dumps(out["bbox_px"])
                writer.writerow(out)

    kept_components = sum(1 for row in component_rows if row["keep"])
    print(f"PDF: {pdf_path}")
    print(f"Final output_dir: {output_dir}")
    print(f"Processing/debug dir: {processing_dir}")
    print(f"AP positions: {len(ap_positions)}")
    print(f"Removed footnote leaders: {len(footnote_indices)}")
    print(f"Raw wall candidates after leader removal: {raw_wall_candidate_count}")
    print(f"Kept wall components: {kept_components}")
    print(f"Wall curves: {len(wall_curves)}")
    print("Output files:")
    print(f"  - {output_dir / cfg['output']['wall_curves_csv']}")
    print(f"  - {output_dir / cfg['output']['ap_positions_csv']}")


if __name__ == "__main__":
    main()
