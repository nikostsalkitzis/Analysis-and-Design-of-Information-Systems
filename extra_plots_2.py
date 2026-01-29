#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SVG plotting script for clustering evaluation:

  - reads final_report2/tables/clustering_eval_agg.csv
  - per (dataset, method) it plots:
      * ARI vs dim (mean ± std) with one series per algo
      * Silhouette vs dim (mean ± std) with one series per algo
  - writes SVG figures to plot_2/

This script does NOT import matplotlib, so it avoids libstdc++ ABI issues.
"""

import os
import numpy as np
import pandas as pd

TABLE_CSV = "final_report2/tables/clustering_eval_agg.csv"
OUT_DIR_FIGS = "final_report2/figures"
os.makedirs(OUT_DIR_FIGS, exist_ok=True)

# -------------------------------
# Data prep
# -------------------------------

def load_data(per_run_csv=TABLE_CSV):
    """Load aggregated clustering CSV."""
    df = pd.read_csv(per_run_csv)
    df["dim"] = df["dim"].astype(int)
    # make sure std columns have no NaN (for plotting)
    for col in ["ari_std", "sil_std"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df

# -------------------------------
# SVG plot helpers (same style as your working script)
# -------------------------------

def _linspace_ticks(vmin, vmax, num=5):
    if vmin == vmax:
        return [vmin] * num
    return list(np.linspace(vmin, vmax, num))

def _scale_x(val, x_min, x_max, margin_left, plot_w):
    if x_max == x_min:
        return margin_left + plot_w/2.0
    return margin_left + (val - x_min) * (plot_w / (x_max - x_min))

def _scale_y(val, y_min, y_max, margin_top, plot_h):
    if y_max == y_min:
        return margin_top + plot_h/2.0
    return margin_top + plot_h - (val - y_min) * (plot_h / (y_max - y_min))

def _svg_escape(txt):
    return str(txt).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def save_metric_svg(
    series_list,
    x_label,
    y_label,
    title,
    out_path,
    width=600,
    height=400,
):
    """
    series_list: list of dicts, each:
        {
            "name": "KMeans",
            "color": "#1f77b4",
            "x":   [16,32,64,...],
            "y":   [0.01,0.02,...],      # means
            "yerr":[0.005,0.003,...],    # stds
        }

    Writes an SVG to out_path.
    """

    # layout
    margin_left   = 60
    margin_right  = 20
    margin_top    = 60
    margin_bottom = 60
    plot_w = width  - margin_left - margin_right
    plot_h = height - margin_top  - margin_bottom

    # collect all x and y ranges
    all_x = []
    all_y_low = []
    all_y_high = []
    for s in series_list:
        xs = list(s["x"])
        ys = list(s["y"])
        ys_std = list(s["yerr"])
        for xv, mv, sv in zip(xs, ys, ys_std):
            if np.isnan(mv):
                continue
            all_x.append(xv)
            all_y_low.append(mv - (sv if not np.isnan(sv) else 0.0))
            all_y_high.append(mv + (sv if not np.isnan(sv) else 0.0))

    if not all_x:
        # nothing to plot, skip creating file
        return

    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y_low)
    y_max = max(all_y_high)

    # add a little vertical padding
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    else:
        pad = 0.05 * (y_max - y_min)
        y_min -= pad
        y_max += pad

    # ticks
    x_ticks = sorted(set(all_x))
    y_ticks = _linspace_ticks(y_min, y_max, num=5)

    # start svg
    svg_lines = []
    svg_lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'font-family="sans-serif" font-size="12">'
    )
    svg_lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    # grid + y ticks
    for yt in y_ticks:
        ypix = _scale_y(yt, y_min, y_max, margin_top, plot_h)
        svg_lines.append(
            f'<line x1="{margin_left}" y1="{ypix:.2f}" '
            f'x2="{margin_left+plot_w}" y2="{ypix:.2f}" '
            f'stroke="#cccccc" stroke-width="1" stroke-dasharray="4,4"/>'
        )
        svg_lines.append(
            f'<text x="{margin_left-8}" y="{ypix+4:.2f}" '
            f'text-anchor="end" fill="#000000">{yt:.3g}</text>'
        )

    # axes
    x_axis_y = margin_top + plot_h
    y_axis_x = margin_left
    svg_lines.append(
        f'<line x1="{y_axis_x}" y1="{margin_top}" '
        f'x2="{y_axis_x}" y2="{margin_top+plot_h}" '
        f'stroke="#000000" stroke-width="1.5"/>'
    )
    svg_lines.append(
        f'<line x1="{margin_left}" y1="{x_axis_y:.2f}" '
        f'x2="{margin_left+plot_w}" y2="{x_axis_y:.2f}" '
        f'stroke="#000000" stroke-width="1.5"/>'
    )

    # x ticks
    for xv in x_ticks:
        xx = _scale_x(xv, x_min, x_max, margin_left, plot_w)
        svg_lines.append(
            f'<line x1="{xx:.2f}" y1="{x_axis_y:.2f}" '
            f'x2="{xx:.2f}" y2="{x_axis_y+5:.2f}" '
            f'stroke="#000000" stroke-width="1.5"/>'
        )
        svg_lines.append(
            f'<text x="{xx:.2f}" y="{x_axis_y+20:.2f}" '
            f'text-anchor="middle" fill="#000000">{xv}</text>'
        )

    # draw each series: ribbon (mean ± std), line, points
    for s in series_list:
        xs = list(s["x"])
        ys = list(s["y"])
        es = list(s["yerr"])
        color = s["color"]

        # drop NaN means
        clean_pts = [
            (xv, mv, sv)
            for xv, mv, sv in zip(xs, ys, es)
            if not np.isnan(mv)
        ]
        if not clean_pts:
            continue

        xs_clean = [p[0] for p in clean_pts]
        ys_clean = [p[1] for p in clean_pts]
        es_clean = [p[2] for p in clean_pts]

        # Ribbon polygon only if we have >=2 points
        if len(xs_clean) >= 2:
            up_pts = []
            lo_pts = []
            for xv, mv, sv in clean_pts:
                up_pts.append(
                    f'{_scale_x(xv, x_min, x_max, margin_left, plot_w):.2f},'
                    f'{_scale_y(mv + (sv if not np.isnan(sv) else 0.0), y_min, y_max, margin_top, plot_h):.2f}'
                )
                lo_pts.append(
                    f'{_scale_x(xv, x_min, x_max, margin_left, plot_w):.2f},'
                    f'{_scale_y(mv - (sv if not np.isnan(sv) else 0.0), y_min, y_max, margin_top, plot_h):.2f}'
                )
            poly_pts = " ".join(up_pts + lo_pts[::-1])
            svg_lines.append(
                f'<polygon points="{poly_pts}" fill="{color}" opacity="0.2" stroke="none"/>'
            )

        # main line
        line_pts = " ".join(
            f'{_scale_x(xv, x_min, x_max, margin_left, plot_w):.2f},'
            f'{_scale_y(mv, y_min, y_max, margin_top, plot_h):.2f}'
            for xv, mv in zip(xs_clean, ys_clean)
        )
        svg_lines.append(
            f'<polyline points="{line_pts}" fill="none" '
            f'stroke="{color}" stroke-width="2"/>'
        )

        # markers
        for xv, mv in zip(xs_clean, ys_clean):
            svg_lines.append(
                f'<circle cx="{_scale_x(xv, x_min, x_max, margin_left, plot_w):.2f}" '
                f'cy="{_scale_y(mv, y_min, y_max, margin_top, plot_h):.2f}" '
                f'r="3" fill="{color}" stroke="none"/>'
            )

    # legend
    legend_x = margin_left + plot_w - 120
    legend_y = margin_top + 10
    dy = 18
    for i, s in enumerate(series_list):
        name = _svg_escape(s["name"])
        color = s["color"]
        y0 = legend_y + i * dy
        svg_lines.append(
            f'<rect x="{legend_x}" y="{y0}" width="12" height="12" '
            f'fill="{color}" opacity="0.6" stroke="{color}" stroke-width="2"/>'
        )
        svg_lines.append(
            f'<text x="{legend_x+18}" y="{y0+10}" fill="#000000">{name}</text>'
        )

    # title
    svg_lines.append(
        f'<text x="{width/2:.2f}" y="{margin_top-30}" text-anchor="middle" '
        f'font-size="14" font-weight="bold" fill="#000000">{_svg_escape(title)}</text>'
    )

    # x label
    svg_lines.append(
        f'<text x="{margin_left+plot_w/2:.2f}" y="{height-20}" text-anchor="middle" '
        f'fill="#000000">{_svg_escape(x_label)}</text>'
    )

    # y label (rotated)
    svg_lines.append(
        f'<text x="20" y="{margin_top+plot_h/2:.2f}" '
        f'transform="rotate(-90 20,{margin_top+plot_h/2:.2f})" '
        f'text-anchor="middle" fill="#000000">{_svg_escape(y_label)}</text>'
    )

    # close svg
    svg_lines.append('</svg>')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))

# -------------------------------
# High-level plotting logic for clustering
# -------------------------------

def _series_from_metric(df_dm, metric_mean_col, metric_std_col, label_map, color_map):
    """
    Build a list of series dicts for save_metric_svg() from clustering df.

    df_dm: filtered df for one (dataset, method)
    metric_mean_col / metric_std_col: e.g. 'ari_mean', 'ari_std'
    label_map: {'kmeans':'KMeans', 'spectral':'Spectral'}
    color_map: {'kmeans':'#1f77b4', 'spectral':'#d62728'}
    """
    series_list = []
    for algo in df_dm["algo"].unique():
        sub = df_dm[df_dm["algo"] == algo].sort_values("dim")
        if sub.empty:
            continue
        x_vals = sub["dim"].astype(int).tolist()
        y_vals = sub[metric_mean_col].astype(float).tolist()
        y_errs = sub[metric_std_col].astype(float).tolist()

        # if all y are NaN, skip series
        if all(np.isnan(v) for v in y_vals):
            continue

        series_list.append(dict(
            name=label_map.get(algo, algo),
            color=color_map.get(algo, "#000000"),
            x=x_vals,
            y=y_vals,
            yerr=y_errs,
        ))
    return series_list

def make_all_plots(df, out_dir=OUT_DIR_FIGS):
    """
    For each dataset+method pair:
      - ARI vs dim (per algo)
      - Silhouette vs dim (per algo)
    """
    color_map = {
        "kmeans": "#1f77b4",
        "spectral": "#d62728",
    }
    label_map = {
        "kmeans": "KMeans",
        "spectral": "Spectral",
    }

    for dataset in df["dataset"].unique():
        df_d = df[df["dataset"] == dataset]
        for method in df_d["method"].unique():
            df_dm = df_d[df_d["method"] == method]
            if df_dm.empty:
                continue

            # how many runs we averaged over
            n_runs = int(df_dm["n_runs"].max()) if "n_runs" in df_dm else None
            runs_txt = f"(n_runs={n_runs})" if n_runs else ""

            # --- ARI ---
            series_ari = _series_from_metric(
                df_dm, "ari_mean", "ari_std", label_map, color_map
            )
            if series_ari:
                save_metric_svg(
                    series_list=series_ari,
                    x_label="Embedding dimension",
                    y_label="ARI (mean ± std)",
                    title=f"{dataset} — ARI vs. Dim ({method}) {runs_txt}",
                    out_path=os.path.join(out_dir, f"{dataset}_{method}_ARI_vs_dim.svg"),
                )

            # --- Silhouette ---
            series_sil = _series_from_metric(
                df_dm, "sil_mean", "sil_std", label_map, color_map
            )
            if series_sil:
                save_metric_svg(
                    series_list=series_sil,
                    x_label="Embedding dimension",
                    y_label="Silhouette (mean ± std)",
                    title=f"{dataset} — Silhouette vs. Dim ({method}) {runs_txt}",
                    out_path=os.path.join(out_dir, f"{dataset}_{method}_Silhouette_vs_dim.svg"),
                )

def main():
    df = load_data(TABLE_CSV)
    if df.empty:
        raise RuntimeError(
            f"{TABLE_CSV} is empty or missing rows. "
            "Make sure clustering_eval_agg.csv exists and is non-empty."
        )
    make_all_plots(df, OUT_DIR_FIGS)
    print("SVG clustering plots written to", OUT_DIR_FIGS)

if __name__ == "__main__":
    main()

