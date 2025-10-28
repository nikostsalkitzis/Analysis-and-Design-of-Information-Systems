#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone plotting script that:
  - reads report/tables/classification_eval.csv
  - computes mean/std across seeds
  - writes SVG figures to report/figures/

This script does NOT import matplotlib (so it avoids libstdc++ ABI issues).
The SVGs include:
  * Accuracy vs dim (SVM & MLP, mean ± std ribbon)
  * F1 vs dim
  * ROC-AUC vs dim
  * Train time vs dim
  * Train peak Python mem vs dim
  * Embedding generation time vs dim
  * Embedding generation peak Python mem vs dim

How to run (from your project root, without changing your env):
    python3 make_extra_plots_svg.py
"""

import os
import numpy as np
import pandas as pd

TABLE_CSV = "final_final_report1/tables/classification_eval.csv"
OUT_DIR_FIGS = "final_final_report1/figures"
os.makedirs(OUT_DIR_FIGS, exist_ok=True)

# -------------------------------
# Data prep
# -------------------------------

def load_data(per_run_csv=TABLE_CSV):
    """Load per-run CSV."""
    df = pd.read_csv(per_run_csv)
    df["dim"] = df["dim"].astype(int)
    df["seed"] = df["seed"].astype(int)
    return df

def make_long_perf(df):
    """
    Turn wide rows into (svm/mlp) long rows.
    Output columns:
        dataset, method, dim, seed, clf,
        acc, f1, auc,
        train_time_sec, train_py_peak_mb,
        gen_time_sec, gen_py_peak_mb
    """
    recs = []
    for _, r in df.iterrows():
        for clf in ["svm", "mlp"]:
            recs.append(dict(
                dataset=r["dataset"],
                method=r["method"],
                dim=int(r["dim"]),
                seed=int(r["seed"]),
                clf=clf,

                acc=float(r[f"acc_{clf}"]),
                f1=float(r[f"f1_{clf}"]),
                auc=float(r[f"auc_{clf}"]),

                train_time_sec=float(r[f"train_time_{clf}_sec"]),
                train_py_peak_mb=float(r[f"train_py_peak_{clf}_mb"]),

                gen_time_sec=float(r["gen_time_sec"]),
                gen_py_peak_mb=float(r["gen_py_peak_mb"]),
            ))
    return pd.DataFrame(recs)

def aggregate(long_df):
    """
    mean/std over seeds for each (dataset, method, dim, clf).
    """
    agg = long_df.groupby(["dataset", "method", "dim", "clf"]).agg(
        acc_mean=("acc", "mean"),         acc_std=("acc", "std"),
        f1_mean=("f1", "mean"),           f1_std=("f1", "std"),
        auc_mean=("auc", "mean"),         auc_std=("auc", "std"),

        train_time_mean=("train_time_sec", "mean"),
        train_time_std=("train_time_sec", "std"),

        train_py_peak_mean=("train_py_peak_mb", "mean"),
        train_py_peak_std=("train_py_peak_mb", "std"),

        gen_time_mean=("gen_time_sec", "mean"),
        gen_time_std=("gen_time_sec", "std"),

        gen_py_peak_mean=("gen_py_peak_mb", "mean"),
        gen_py_peak_std=("gen_py_peak_mb", "std"),

        n_runs=("acc", "count"),
    ).reset_index()

    # replace NaN std (single seed) with 0.0 for plotting
    for col in list(agg.columns):
        if col.endswith("_std"):
            agg[col] = agg[col].fillna(0.0)
    return agg

# -------------------------------
# SVG plot helpers
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
            "name": "SVM",
            "color": "#1f77b4",
            "x":   [32,64,128,...],
            "y":   [0.7,0.75,...],      # means
            "yerr":[0.05,0.03,...],     # stds (same len as x)
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

        # drop NaN means (we can't render "nan" in SVG geometry)
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
# High-level plotting logic
# -------------------------------

def _series_from_metric(df_dm, metric_mean_col, metric_std_col, label_map, color_map):
    """
    Build a list of series dicts for save_metric_svg().

    df_dm: aggregated df filtered to (dataset, method)
    metric_mean_col / metric_std_col: e.g. 'acc_mean', 'acc_std'
    label_map: {'svm':'SVM', 'mlp':'MLP'}
    color_map: {'svm':'#1f77b4', 'mlp':'#d62728'}
    """
    series_list = []
    for clf in df_dm["clf"].unique():
        sub = df_dm[df_dm["clf"] == clf].sort_values("dim")
        if sub.empty:
            continue
        x_vals = sub["dim"].tolist()
        y_vals = sub[metric_mean_col].astype(float).tolist()
        y_errs = sub[metric_std_col].astype(float).tolist()

        # if all y are NaN, skip series
        if all(np.isnan(v) for v in y_vals):
            continue

        series_list.append(dict(
            name=label_map.get(clf, clf),
            color=color_map.get(clf, "#000000"),
            x=x_vals,
            y=y_vals,
            yerr=y_errs,
        ))
    return series_list

def make_all_plots(agg_df, out_dir=OUT_DIR_FIGS):
    """
    For each dataset+method pair:
      - accuracy
      - f1
      - auc
      - train_time
      - train_mem
      - gen_time
      - gen_mem
    """
    color_map = {"svm": "#1f77b4", "mlp": "#d62728"}
    label_map_generic = {"svm": "SVM", "mlp": "MLP"}

    for dataset in agg_df["dataset"].unique():
        df_d = agg_df[agg_df["dataset"] == dataset]
        for method in df_d["method"].unique():
            df_dm = df_d[df_d["method"] == method]

            # how many seeds we averaged over
            n_runs = int(df_dm["n_runs"].max()) if "n_runs" in df_dm else None
            seeds_txt = f"(n_seeds={n_runs})" if n_runs else ""

            # --- Accuracy ---
            series_acc = _series_from_metric(
                df_dm, "acc_mean", "acc_std", label_map_generic, color_map
            )
            if series_acc:
                save_metric_svg(
                    series_list=series_acc,
                    x_label="Embedding dimension",
                    y_label="Accuracy (mean ± std)",
                    title=f"{dataset} — Accuracy vs. Dim ({method}) {seeds_txt}",
                    out_path=os.path.join(out_dir, f"{dataset}_{method}_acc_vs_dim.svg"),
                )

            # --- F1 ---
            series_f1 = _series_from_metric(
                df_dm, "f1_mean", "f1_std", label_map_generic, color_map
            )
            if series_f1:
                save_metric_svg(
                    series_list=series_f1,
                    x_label="Embedding dimension",
                    y_label="Macro-F1 (mean ± std)",
                    title=f"{dataset} — F1 vs. Dim ({method}) {seeds_txt}",
                    out_path=os.path.join(out_dir, f"{dataset}_{method}_f1_vs_dim.svg"),
                )

            # --- ROC-AUC ---
            series_auc = _series_from_metric(
                df_dm, "auc_mean", "auc_std", label_map_generic, color_map
            )
            if series_auc:
                save_metric_svg(
                    series_list=series_auc,
                    x_label="Embedding dimension",
                    y_label="ROC-AUC (macro OvR, mean ± std)",
                    title=f"{dataset} — ROC-AUC vs. Dim ({method}) {seeds_txt}",
                    out_path=os.path.join(out_dir, f"{dataset}_{method}_auc_vs_dim.svg"),
                )

            # --- Train time ---
            series_ttime = _series_from_metric(
                df_dm, "train_time_mean", "train_time_std", label_map_generic, color_map
            )
            if series_ttime:
                save_metric_svg(
                    series_list=series_ttime,
                    x_label="Embedding dimension",
                    y_label="Train time (s, mean ± std)",
                    title=f"{dataset} — Train Time vs. Dim ({method}) {seeds_txt}",
                    out_path=os.path.join(out_dir, f"{dataset}_{method}_train_time_vs_dim.svg"),
                )

            # --- Train memory ---
            series_tmem = _series_from_metric(
                df_dm, "train_py_peak_mean", "train_py_peak_std", label_map_generic, color_map
            )
            if series_tmem:
                save_metric_svg(
                    series_list=series_tmem,
                    x_label="Embedding dimension",
                    y_label="Train peak Python mem (MB, mean ± std)",
                    title=f"{dataset} — Train Peak Mem vs. Dim ({method}) {seeds_txt}",
                    out_path=os.path.join(out_dir, f"{dataset}_{method}_train_mem_vs_dim.svg"),
                )

            # --- Embedding generation time ---
            # gen_time/gen_mem are the same for SVM+MLP rows, so just take clf=="svm"
            df_dm_svm = df_dm[df_dm["clf"] == "svm"].sort_values("dim")
            if not df_dm_svm.empty:
                series_gen_time = [dict(
                    name="Embedding gen time",
                    color="#2ca02c",
                    x=df_dm_svm["dim"].tolist(),
                    y=df_dm_svm["gen_time_mean"].astype(float).tolist(),
                    yerr=df_dm_svm["gen_time_std"].astype(float).tolist(),
                )]
                if not all(np.isnan(v) for v in series_gen_time[0]["y"]):
                    save_metric_svg(
                        series_list=series_gen_time,
                        x_label="Embedding dimension",
                        y_label="Generation time (s, mean ± std)",
                        title=f"{dataset} — Embedding Gen Time vs. Dim ({method}) {seeds_txt}",
                        out_path=os.path.join(out_dir, f"{dataset}_{method}_gentime_vs_dim.svg"),
                    )

                series_gen_mem = [dict(
                    name="Embedding gen peak mem",
                    color="#9467bd",
                    x=df_dm_svm["dim"].tolist(),
                    y=df_dm_svm["gen_py_peak_mean"].astype(float).tolist(),
                    yerr=df_dm_svm["gen_py_peak_std"].astype(float).tolist(),
                )]
                if not all(np.isnan(v) for v in series_gen_mem[0]["y"]):
                    save_metric_svg(
                        series_list=series_gen_mem,
                        x_label="Embedding dimension",
                        y_label="Peak Python mem (MB, mean ± std)",
                        title=f"{dataset} — Embedding Gen Peak Mem vs. Dim ({method}) {seeds_txt}",
                        out_path=os.path.join(out_dir, f"{dataset}_{method}_genmem_vs_dim.svg"),
                    )

def main():
    df = load_data(TABLE_CSV)
    if df.empty:
        raise RuntimeError(
            f"{TABLE_CSV} is empty or missing rows. "
            "Run your experiment script first so it generates classification_eval.csv."
        )

    long_df = make_long_perf(df)
    agg_df = aggregate(long_df)
    make_all_plots(agg_df, OUT_DIR_FIGS)
    print("SVG plots written to", OUT_DIR_FIGS)

if __name__ == "__main__":
    main()

