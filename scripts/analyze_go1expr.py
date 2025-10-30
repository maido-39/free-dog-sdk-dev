#!/usr/bin/env python3
"""Analyze go1 expression CSV and produce plots and CLI stats.

Usage: python scripts/analyze_go1expr.py <csv-path> [--output out.png]
"""
import argparse
import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def minmax_normalize(s: pd.Series) -> pd.Series:
    mn = s.min()
    mx = s.max()
    if np.isclose(mx, mn):
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def annotate_extrema(ax, x, y, label_prefix=""):
    # annotate max and min points with small text
    idx_max = y.idxmax()
    idx_min = y.idxmin()
    x_max, y_max = x.loc[idx_max], y.loc[idx_max]
    x_min, y_min = x.loc[idx_min], y.loc[idx_min]
    ax.plot([x_max], [y_max], marker='o', color='C3', markersize=4)
    ax.plot([x_min], [y_min], marker='o', color='C4', markersize=4)
    ax.text(x_max, y_max, f"{label_prefix}max: {y_max:.3f}", fontsize=8, color='C3',
            verticalalignment='bottom', horizontalalignment='right')
    ax.text(x_min, y_min, f"{label_prefix}min: {y_min:.3f}", fontsize=8, color='C4',
            verticalalignment='top', horizontalalignment='left')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', nargs='?', default=None, help='CSV file to analyze (if omitted, use newest in expr_data/)')
    p.add_argument('--output', '-o', default=None, help='Output image path (png)')
    args = p.parse_args()

    csv_path = args.csv
    if csv_path is None:
        # find newest CSV in expr_data directory
        expr_dir = Path('../expr_data')
        if not expr_dir.exists():
            raise SystemExit('../expr_data/ directory not found and no CSV path provided')
        candidates = list(expr_dir.glob('*.csv'))
        if not candidates:
            raise SystemExit('No CSV files found in ../expr_data/')
        # choose by modification time
        csv_path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"No CSV provided. Using newest file: {csv_path}")
    else:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise SystemExit(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure numeric conversion where possible
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass

    # derive time index
    if 'elapsed' in df.columns:
        t = df['elapsed']
    else:
        t = pd.Series(np.arange(len(df)))

    # Position 2D
    pos_x = df.get('pos_x')
    pos_y = df.get('pos_y')

    # Attitude: roll, pitch, yaw -> convert to degrees
    roll = df.get('roll')
    pitch = df.get('pitch')
    yaw = df.get('yaw')
    if roll is None or pitch is None or yaw is None:
        raise SystemExit('roll/pitch/yaw columns required in CSV')
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    # Velocity magnitude (m/s)
    vx = df.get('vel_x', pd.Series(0, index=df.index))
    vy = df.get('vel_y', pd.Series(0, index=df.index))
    vz = df.get('vel_z', pd.Series(0, index=df.index))
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # Yaw speed -> convert to deg/s
    yawspeed = df.get('yaw_speed', pd.Series(0, index=df.index))
    yawspeed_deg = np.degrees(yawspeed)

    # Metrics: choose numeric columns excluding positional / attitude / velocity / time fields
    exclude = set(["timestamp", "elapsed", "battery", "mode",
                   "pos_x", "pos_y", "pos_z",
                   "vel_x", "vel_y", "vel_z",
                   "yaw_speed", "roll", "pitch", "yaw",
                   # remove foot force columns from metrics
                   "footf_0", "footf_1", "footf_2", "footf_3"])
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    metrics = df[numeric_cols]

    # Compute normalized metrics (min-max) for plotting only
    normalized = pd.DataFrame({c: minmax_normalize(metrics[c]) for c in metrics.columns}) if not metrics.empty else pd.DataFrame()

    # Stats: mean, std, var for metrics (raw values)
    stats_lines = []
    print('\nMetric statistics (raw values):')
    for c in metrics.columns:
        mean = metrics[c].mean()
        std = metrics[c].std()
        var = metrics[c].var()
        print(f"{c}: mean={mean:.6g}, std={std:.6g}, var={var:.6g}")
        stats_lines.append(f"{c}: mean={mean:.3g}, std={std:.3g}, var={var:.3g}")

    # Build figure
    fig = plt.figure(constrained_layout=True, figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])

    ax_pos = fig.add_subplot(gs[0, :])
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_att = fig.add_subplot(gs[1, 1])
    ax_vel = fig.add_subplot(gs[2, 0])
    ax_text = fig.add_subplot(gs[2, 1])
    ax_text.axis('off')

    # Position plot
    ax_pos.plot(pos_x, pos_y, '-o', markersize=3)
    ax_pos.set_title('2D Position')
    ax_pos.set_xlabel('pos_x (m)')
    ax_pos.set_ylabel('pos_y (m)')
    # make x/y scale equal so distances are not distorted
    try:
        ax_pos.set_aspect('equal', adjustable='datalim')
    except Exception:
        # fallback if matplotlib backend doesn't allow
        pass
    # mark start and end
    if len(df) > 0:
        ax_pos.plot(pos_x.iloc[0], pos_y.iloc[0], marker='s', color='green', label='start')
        ax_pos.plot(pos_x.iloc[-1], pos_y.iloc[-1], marker='X', color='red', label='end')
        ax_pos.legend()

    # Metrics normalized - single subplot
    if not normalized.empty:
        for i, c in enumerate(normalized.columns):
            ax_metrics.plot(t, normalized[c], label=c)
        ax_metrics.set_title('Normalized metrics (min-max)')
        ax_metrics.set_xlabel('elapsed (s)')
        ax_metrics.set_ylabel('normalized')
        ax_metrics.legend(fontsize=8, loc='upper right')
    else:
        ax_metrics.text(0.5, 0.5, 'No metric columns found', ha='center', va='center')

    # Attitude: roll & pitch on left axis, yaw on right axis (separate scale)
    ax_att_left = ax_att
    ax_att_right = ax_att.twinx()

    ax_att_left.plot(t, roll_deg, label='roll (deg)', color='C0')
    ax_att_left.plot(t, pitch_deg, label='pitch (deg)', color='C2')
    ax_att_right.plot(t, yaw_deg, label='yaw (deg)', color='C1')

    ax_att_left.set_title('Attitude: roll / pitch (left) and yaw (right)')
    ax_att_left.set_xlabel('elapsed (s)')
    ax_att_left.set_ylabel('roll / pitch (deg)')
    ax_att_right.set_ylabel('yaw (deg)')

    # combined legend
    lines_left, labels_left = ax_att_left.get_legend_handles_labels()
    lines_right, labels_right = ax_att_right.get_legend_handles_labels()
    ax_att_left.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right', fontsize=9)

    # annotate extrema on respective axes
    annotate_extrema(ax_att_left, t, pd.Series(roll_deg, index=df.index), label_prefix='')
    annotate_extrema(ax_att_left, t, pd.Series(pitch_deg, index=df.index), label_prefix='')
    annotate_extrema(ax_att_right, t, pd.Series(yaw_deg, index=df.index), label_prefix='')

    # Velocity and yaw speed: use twin y-axis because yaw (deg/s) scale differs
    ax_vel_left = ax_vel
    ax_vel_right = ax_vel.twinx()
    ax_vel_left.plot(t, vel_mag, label='velocity magnitude (m/s)', color='C0')
    ax_vel_right.plot(t, yawspeed_deg, label='yaw speed (deg/s)', color='C1')
    ax_vel_left.set_title('Velocity (m/s) and Yaw speed (deg/s)')
    ax_vel_left.set_xlabel('elapsed (s)')
    ax_vel_left.set_ylabel('velocity (m/s)')
    ax_vel_right.set_ylabel('yaw speed (deg/s)')
    # legends: combine
    lines_left, labels_left = ax_vel_left.get_legend_handles_labels()
    lines_right, labels_right = ax_vel_right.get_legend_handles_labels()
    ax_vel_left.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right', fontsize=9)
    # annotate extrema on both axes
    annotate_extrema(ax_vel_left, t, pd.Series(vel_mag, index=df.index), label_prefix='')
    annotate_extrema(ax_vel_right, t, pd.Series(yawspeed_deg, index=df.index), label_prefix='')

    # Put stats into ax_text
    stats_text = '\n'.join(stats_lines) if stats_lines else 'No metrics to report.'
    # make the analysis text more readable by increasing font size
    ax_text.text(0.01, 0.99, 'Metric statistics (mean / std / var):\n' + stats_text,
                 va='top', ha='left', fontsize=11, family='monospace')

    fig.suptitle(f'Analysis: {csv_path.name}')

    out_path = args.output
    if out_path is None:
        out_dir = Path('plots')
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / (csv_path.stem + '_analysis.png')
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to: {out_path}")

    try:
        plt.show()
    except Exception:
        # In headless envs this may fail; it's okay because file was saved
        pass


if __name__ == '__main__':
    main()
