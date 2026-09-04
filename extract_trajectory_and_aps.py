#!/usr/bin/env python3
"""Extract DXF trajectory, detect named PDF APs, register, and export results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pymupdf
from scipy.cluster.hierarchy import fclusterdata
from scipy.optimize import differential_evolution, linear_sum_assignment
from scipy.spatial import cKDTree


AP_LABEL = re.compile(r"^(?:D[12]|U[12])[-–— ]?\d{1,2}$")


def load_config(path: Path):
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    base = path.resolve().parent
    for section, names in (("input", ("dxf", "pdf")), ("output", ("trajectory_csv", "ap_csv", "combined_pdf"))):
        for name in names:
            config[section][name] = (base / config[section][name]).resolve()
    return config


def read_dxf_entities(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    pairs = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]
    entities, current, pending_x = [], None, None

    def finish():
        if not current or current.get("layer") not in {"WALLS", "TRAJECTORY"}:
            return
        if current["type"] == "LINE" and all(code in current for code in ("10", "20", "11", "21")):
            entities.append({"layer": current["layer"], "points": [(current["10"], current["20"]), (current["11"], current["21"])]})
        elif current["type"] == "LWPOLYLINE" and len(current["points"]) >= 2:
            entities.append({"layer": current["layer"], "points": current["points"][:]})

    for code, value in pairs:
        if code == "0":
            finish()
            current = {"type": value, "layer": None, "points": []} if value in {"LINE", "LWPOLYLINE"} else None
            pending_x = None
        elif current:
            if code == "8":
                current["layer"] = value
            elif current["type"] == "LINE" and code in {"10", "20", "11", "21"}:
                try:
                    current[code] = float(value)
                except ValueError:
                    pass
            elif current["type"] == "LWPOLYLINE" and code == "10":
                try:
                    pending_x = float(value)
                except ValueError:
                    pending_x = None
            elif current["type"] == "LWPOLYLINE" and code == "20" and pending_x is not None:
                try:
                    current["points"].append((pending_x, float(value)))
                except ValueError:
                    pass
                pending_x = None
    finish()
    if not any(entity["layer"] == "TRAJECTORY" for entity in entities):
        raise ValueError("DXF contains no TRAJECTORY entities")
    if not any(entity["layer"] == "WALLS" for entity in entities):
        raise ValueError("DXF contains no WALLS entities")
    return entities


def drawing_points(item, curve_samples=8):
    if item[0] == "l":
        return [(item[1].x, item[1].y), (item[2].x, item[2].y)]
    if item[0] == "c":
        p0, p1, p2, p3 = item[1:5]
        result = []
        for t in np.linspace(0, 1, curve_samples):
            result.append(((1-t)**3*p0.x + 3*(1-t)**2*t*p1.x + 3*(1-t)*t*t*p2.x + t**3*p3.x,
                           (1-t)**3*p0.y + 3*(1-t)**2*t*p1.y + 3*(1-t)*t*t*p2.y + t**3*p3.y))
        return result
    if item[0] == "re":
        rect = item[1]
        return [(rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)]
    return []


def pdf_registration_points(page, config):
    x0, y0, x1, y1 = config["pdf"]["plan_roi_pdf_pt"]
    black_max = config["registration"]["black_max_channel"]
    points = []
    for drawing in page.get_drawings():
        color, rect = drawing.get("color"), drawing["rect"]
        if color is None or max(color) > black_max:
            continue
        if rect.x1 < x0 or rect.x0 > x1 or rect.y1 < y0 or rect.y0 > y1:
            continue
        for item in drawing["items"]:
            points.extend(drawing_points(item))
    if not points:
        raise ValueError("no black PDF plan vectors found inside plan_roi_pdf_pt")
    return np.unique(np.round(np.array(points), 3), axis=0)


def detect_markers(page, config):
    markers = []
    tolerance = config["ap_detection"]["color_tolerance"]
    cluster_distance = config["ap_detection"]["cluster_distance_pdf_pt"]
    for kind, target in config["ap_detection"]["colors_rgb_0_1"].items():
        points = []
        for drawing in page.get_drawings():
            matched = any(color and max(abs(color[i] - target[i]) for i in range(3)) <= tolerance
                          for color in (drawing.get("color"), drawing.get("fill")))
            if matched:
                rect = drawing["rect"]
                points.append(((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2))
        if not points:
            continue
        groups = fclusterdata(np.array(points), cluster_distance, criterion="distance", method="single")
        for group in sorted(set(groups)):
            cluster = np.array([point for point, value in zip(points, groups) if value == group])
            lo, hi = cluster.min(axis=0), cluster.max(axis=0)
            if len(cluster) >= config["ap_detection"]["min_cluster_items"] and max(hi - lo) <= config["ap_detection"]["max_cluster_span_pdf_pt"]:
                markers.append({"type": kind, "x": float(cluster[:, 0].mean()), "y": float(cluster[:, 1].mean()), "items": len(cluster)})
    if not markers:
        raise ValueError("no AP markers detected in PDF")
    return markers


def assign_names(page, markers):
    occurrences = {}
    for word in page.get_text("words"):
        text = word[4].replace("–", "-").replace("—", "-").replace(" ", "")
        if AP_LABEL.match(text):
            occurrences.setdefault(text, []).append(((word[0] + word[2]) / 2, (word[1] + word[3]) / 2))
    names = sorted(occurrences)
    if len(names) < len(markers):
        raise ValueError(f"found {len(markers)} AP markers but only {len(names)} unique names")
    costs = np.array([[min(math.hypot(marker["x"]-x, marker["y"]-y) for x, y in occurrences[name])
                       for name in names] for marker in markers])
    marker_indexes, name_indexes = linear_sum_assignment(costs)
    assigned = {i: [names[j]] for i, j in zip(marker_indexes, name_indexes)}
    used = {names[j] for j in name_indexes}
    for name in set(names) - used:
        index = min(range(len(markers)), key=lambda i: min(math.hypot(markers[i]["x"]-x, markers[i]["y"]-y) for x, y in occurrences[name]))
        assigned.setdefault(index, []).append(name)
    for index, marker in enumerate(markers):
        marker["names"] = sorted(assigned[index])
        marker["label_distance"] = min(min(math.hypot(marker["x"]-x, marker["y"]-y) for x, y in occurrences[name]) for name in marker["names"])
    return names


def fit_registration(source, target, config):
    rng = np.random.default_rng(config["registration"]["random_seed"])
    max_source = config["registration"]["max_source_points"]
    fit_source = source[rng.choice(len(source), min(max_source, len(source)), replace=False)]
    source_center, centered = fit_source.mean(axis=0), fit_source - fit_source.mean(axis=0)
    max_target = config["registration"]["max_target_points"]
    fit_target = target[rng.choice(len(target), min(max_target, len(target)), replace=False)]
    tree = cKDTree(fit_target)
    target_center, span = target.mean(axis=0), np.ptp(target, axis=0)
    expected_scale = np.linalg.norm(np.ptp(target, axis=0)) / np.linalg.norm(np.ptp(source, axis=0))
    bounds = [(0.6*expected_scale, 1.4*expected_scale), (-math.pi, math.pi),
              (target_center[0]-span[0], target_center[0]+span[0]),
              (target_center[1]-span[1], target_center[1]+span[1])]
    candidates = []
    for reflection in (1.0, -1.0):
        def objective(parameters):
            scale, angle, tx, ty = parameters
            c, s = math.cos(angle), math.sin(angle)
            matrix = np.array([[c, -s], [s, c]]) @ np.diag([scale, scale*reflection])
            distances = tree.query(centered @ matrix.T + (tx, ty), k=1)[0]
            keep = max(1, int(config["registration"]["trim_fraction"] * len(distances)))
            return float(np.mean(np.partition(distances, keep-1)[:keep]))
        result = differential_evolution(objective, bounds, seed=config["registration"]["random_seed"],
                                        popsize=15, maxiter=180, tol=1e-7, polish=True)
        candidates.append((result.fun, reflection, result.x))
    _, reflection, parameters = min(candidates, key=lambda item: item[0])
    scale, angle, tx, ty = parameters
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.array([[c, -s], [s, c]]) @ np.diag([scale, scale*reflection])
    offset = np.array([tx, ty]) - matrix @ source_center
    return matrix, offset, scale, math.degrees(angle), int(reflection)


def write_trajectory(path, entities):
    rows = []
    for entity_index, entity in enumerate((e for e in entities if e["layer"] == "TRAJECTORY"), 1):
        for vertex_index, (x, y) in enumerate(entity["points"]):
            rows.append({"entity_id": f"TR_{entity_index:04d}", "vertex_index": vertex_index, "x_dxf": f"{x:.6f}", "y_dxf": f"{y:.6f}"})
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def write_ap_csv(path, markers, matrix, offset, dpi, metrics):
    rows = []
    for index, marker in enumerate(sorted(markers, key=lambda item: (item["y"], item["x"])), 1):
        mapped = matrix @ np.array([marker["x"], marker["y"]]) + offset
        rows.append({
            "id": f"AP_{index:03d}", "type": marker["type"], "ap_name": "|".join(marker["names"]),
            "x_pdf_pt": f"{marker['x']:.3f}", "y_pdf_pt": f"{marker['y']:.3f}",
            "x_pdf_px": f"{marker['x']*dpi/72:.1f}", "y_pdf_px": f"{marker['y']*dpi/72:.1f}",
            "x_dxf": f"{mapped[0]:.6f}", "y_dxf": f"{mapped[1]:.6f}",
            "cluster_items": marker["items"], "label_distance_pdf_pt": f"{marker['label_distance']:.3f}",
            "mapping_quality": "merged-pair" if len(marker["names"]) > 1 else "good",
            "registration_median_error_dxf": f"{metrics['median']:.6f}",
            "registration_p95_error_dxf": f"{metrics['p95']:.6f}",
        })
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def write_combined_pdf(path, entities, aps, metrics):
    all_points = np.array([point for entity in entities for point in entity["points"]])
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    margin, width = 34.0, 700.0
    height = max(500.0, width*(hi[1]-lo[1])/(hi[0]-lo[0]))
    scale = min((width-2*margin)/(hi[0]-lo[0]), (height-2*margin)/(hi[1]-lo[1]))
    def pt(x, y): return pymupdf.Point(margin+(x-lo[0])*scale, height-margin-(y-lo[1])*scale)
    document = pymupdf.open(); page = document.new_page(width=width, height=height)
    for entity in entities:
        shape = page.new_shape(); shape.draw_polyline([pt(x, y) for x, y in entity["points"]])
        shape.finish(color=(0.1, 0.35, 0.9) if entity["layer"] == "TRAJECTORY" else (0.68, 0.72, 0.77),
                     width=1.25 if entity["layer"] == "TRAJECTORY" else 0.28); shape.commit()
    colors = {"orange": (0.97, 0.38, 0.05), "violet": (0.48, 0.23, 0.93)}
    for row in aps:
        point = pt(float(row["x_dxf"]), float(row["y_dxf"])); shape = page.new_shape(); shape.draw_circle(point, 3.2)
        shape.finish(color=(1, 1, 1), fill=colors[row["type"]], width=0.7); shape.commit()
        page.insert_text(pymupdf.Point(point.x+4.5, point.y+2), row["ap_name"], fontsize=6, color=(0.05, 0.08, 0.13))
    page.insert_text(pymupdf.Point(18, 19), "DXF trajectory and registered Wi-Fi AP positions", fontsize=11)
    page.insert_text(pymupdf.Point(18, 31), f"Registration median {metrics['median']:.3f}; p95 {metrics['p95']:.3f} DXF units", fontsize=7, color=(0.3, 0.34, 0.4))
    document.save(path, garbage=4, deflate=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args(); config = load_config(args.config)
    for path in config["output"].values(): path.parent.mkdir(parents=True, exist_ok=True)
    entities = read_dxf_entities(config["input"]["dxf"])
    document = pymupdf.open(config["input"]["pdf"]); page = document[config["input"]["pdf_page_index"]]
    source = pdf_registration_points(page, config)
    target = np.array([point for entity in entities if entity["layer"] == "WALLS" for point in entity["points"]])
    matrix, offset, scale, angle, reflection = fit_registration(source, target, config)
    residuals = cKDTree(target).query(source @ matrix.T + offset, k=1)[0]
    metrics = {"median": float(np.median(residuals)), "p95": float(np.percentile(residuals, 95))}
    markers = detect_markers(page, config); unique_names = assign_names(page, markers)
    trajectory = write_trajectory(config["output"]["trajectory_csv"], entities)
    aps = write_ap_csv(config["output"]["ap_csv"], markers, matrix, offset, config["pdf"]["render_dpi"], metrics)
    write_combined_pdf(config["output"]["combined_pdf"], entities, aps, metrics)
    print(f"trajectory_entities={len({row['entity_id'] for row in trajectory})} trajectory_vertices={len(trajectory)}")
    print(f"ap_markers={len(aps)} unique_ap_names={len(unique_names)}")
    print(f"transform_matrix={matrix.tolist()} offset={offset.tolist()}")
    print(f"scale={scale:.9f} angle_deg={angle:.6f} reflection={reflection}")
    print(f"registration_median={metrics['median']:.6f} registration_p95={metrics['p95']:.6f}")
    for path in config["output"].values(): print(path)


if __name__ == "__main__":
    main()
