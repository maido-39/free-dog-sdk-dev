#!/usr/bin/env python3
"""
Live-tail the most recent CSV in expr_data/ and print metric columns every 0.2s.

Usage: python tools/live_csv_tail.py
"""
import os
import time
import csv
import glob
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
EXPR_DIR = os.path.join(ROOT, "expr_data")
POLL_INTERVAL = 0.2


def find_latest_csv():
    files = glob.glob(os.path.join(EXPR_DIR, "*.csv"))
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    return latest


def read_header(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return []
        return next(csv.reader([first]))


def parse_row(header, line):
    try:
        row = next(csv.reader([line]))
    except Exception:
        return None
    if len(row) != len(header):
        # tolerate trailing commas or missing; pad
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        else:
            row = row[: len(header)]
    return dict(zip(header, row))


def format_metrics(row):
    # pick useful fields, convert to floats where possible
    def f(k, fmt="{:.3f}"):
        v = row.get(k, "")
        try:
            return fmt.format(float(v))
        except Exception:
            return str(v)

    # Only return the four metric lines (smoothed values)
    stab = f("stability_smooth", "{:.3f}")
    speed_s = f("speed_score_smooth", "{:.1f}")
    time_s = f("time_score_smooth", "{:.1f}")
    total = f("total_score", "{:.3f}")

    out = (
        f"Stability: {stab}\n"
        f"SpeedScore: {speed_s}\n"
        f"TimeScore: {time_s}\n"
        f"Total: {total}"
    )
    return out


def tail_latest():
    last_path = None
    f = None
    last_pos = 0
    header = None

    print("Live CSV tail: watching directory:", EXPR_DIR)
    try:
        while True:
            latest = find_latest_csv()
            if latest is None:
                print("No csv files in expr_data/, waiting...")
                time.sleep(1.0)
                continue

            if latest != last_path:
                # open new file
                if f:
                    try:
                        f.close()
                    except Exception:
                        pass
                last_path = latest
                header = read_header(latest)
                f = open(latest, "r", encoding="utf-8")
                # jump to end, but still try to get the last non-empty line to print immediately
                try:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    if size == 0:
                        last_pos = f.tell()
                        last_line = None
                    else:
                        # read last line by reading backwards a chunk
                        with open(latest, "rb") as rb:
                            rb.seek(0, os.SEEK_END)
                            end = rb.tell()
                            # read up to 16KB from end
                            to_read = min(16384, end)
                            rb.seek(end - to_read)
                            data = rb.read().decode(errors="ignore")
                            lines = [l for l in data.splitlines() if l.strip()]
                            last_line = lines[-1] if lines else None
                        last_pos = f.tell()
                except Exception:
                    last_pos = f.tell()
                    last_line = None

                if last_line:
                    parsed = parse_row(header, last_line)
                    if parsed:
                        print("\n----- New file detected:", os.path.basename(latest), "-----\n")
                        print(format_metrics(parsed))

            else:
                # same file: check growth / truncation
                try:
                    cur_size = os.path.getsize(last_path)
                except Exception:
                    cur_size = None

                if cur_size is not None and cur_size < last_pos:
                    # file truncated/rotated, reopen
                    f.close()
                    f = open(last_path, "r", encoding="utf-8")
                    header = read_header(last_path)
                    last_pos = f.tell()

                # read new lines
                f.seek(last_pos)
                new_data = f.read()
                if new_data:
                    lines = [l for l in new_data.splitlines() if l.strip()]
                    if lines:
                        last_line = lines[-1]
                        parsed = parse_row(header, last_line)
                        if parsed:
                            print("\n[", datetime.now().isoformat(timespec='seconds'), "]")
                            print(format_metrics(parsed))
                last_pos = f.tell()

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            if f:
                f.close()
        except Exception:
            pass


if __name__ == "__main__":
    tail_latest()
