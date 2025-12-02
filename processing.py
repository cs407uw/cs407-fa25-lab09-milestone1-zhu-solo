import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import io

def load_data(path, header=0):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cleaned_lines = []
    for line in lines:
        if not line.strip():
            continue
        cleaned = line.rstrip()
        while cleaned.endswith(','):
            cleaned = cleaned[:-1].rstrip()
        cleaned_lines.append(cleaned + '\n')
    text = ''.join(cleaned_lines)
    df = pd.read_csv(io.StringIO(text), header=header)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def centered_moving_avg_series(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()

def analyze_acceleration(csv_path='ACCELERATION.csv', plot=True, save_prefix='part1'):
    df = load_data(csv_path)
    t = df.iloc[:,0].astype(float).values          
    a_true = df.iloc[:,1].astype(float).values
    a_noisy = df.iloc[:,2].astype(float).values

    def cum_trap(y, x):
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        if y.size == 0:
            return np.array([], dtype=float)
        out = np.empty_like(y, dtype=float)
        out[0] = 0.0
        for i in range(1, len(y)):
            out[i] = out[i-1] + 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
        return out

    v_true = cum_trap(a_true, t)
    v_noisy = cum_trap(a_noisy, t)
    s_true = cum_trap(v_true, t)
    s_noisy = cum_trap(v_noisy, t)

    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(t, a_true, label='accel_true')
        plt.plot(t, a_noisy, label='accel_noisy', alpha=0.7)
        plt.legend(); plt.title('Acceleration and Noisy Acceleration'); plt.xlabel('t (s)'); plt.grid(True)
        plt.tight_layout(); plt.savefig(f'{save_prefix}_accel.png'); plt.close()

        plt.figure(figsize=(8,4))
        plt.plot(t, v_true, label='v_true')
        plt.plot(t, v_noisy, label='v_noisy', alpha=0.7)
        plt.legend(); plt.title('Speeds Obtained from Actual Acceleration and Noisy Acceleration'); plt.xlabel('t (s)'); plt.grid(True)
        plt.tight_layout(); plt.savefig(f'{save_prefix}_speeds.png'); plt.close()

        plt.figure(figsize=(8,4))
        plt.plot(t, s_true, label='s_true')
        plt.plot(t, s_noisy, label='s_noisy', alpha=0.7)
        plt.legend(); plt.title('Distance Traveled from Actual Acceleration and Noisy Acceleration'); plt.xlabel('t (s)'); plt.grid(True)
        plt.tight_layout(); plt.savefig(f'{save_prefix}_distance.png'); plt.close()

    return {'distance_true': float(s_true[-1]), 'distance_noisy': float(s_noisy[-1])}

def step_detection(csv_path='WALKING.csv', ax_col='accel_x', ay_col='accel_y', az_col='accel_z',
                   ts_col='timestamp', window=5, k=0.5, min_interval_s=0.3, plot=True, save_prefix='part2'):
    df = load_data(csv_path)
    ts = df[ts_col].astype(float).values
    ts_s = ts / 1e9
    ax = df[ax_col].astype(float).values
    ay = df[ay_col].astype(float).values
    az = df[az_col].astype(float).values

    mag = np.sqrt(ax*ax + ay*ay + az*az)
    mag_s = pd.Series(mag)
    smoothed = centered_moving_avg_series(mag_s, window).values

    mean_s = np.nanmean(smoothed)
    std_s = np.nanstd(smoothed)
    threshold = mean_s + k * std_s

    median_dt = np.median(np.diff(ts_s)) if len(ts_s)>1 else 0.005
    min_samples = max(1, int(round(min_interval_s / median_dt)))

    peaks = []
    last = -min_samples - 1
    for i in range(1, len(smoothed)-1):
        if smoothed[i] > threshold and smoothed[i] > smoothed[i-1] and smoothed[i] >= smoothed[i+1]:
            if (i - last) >= min_samples:
                peaks.append(i)
                last = i

    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(ts_s, mag, label='magnitude', alpha=0.6)
        plt.plot(ts_s, smoothed, label='smoothed', linewidth=1.5)
        plt.scatter(ts_s[peaks], smoothed[peaks], color='red', label='peaks')
        plt.legend(); plt.title('Accel magnitude and detected steps'); plt.xlabel('t (s)'); plt.grid(True)
        plt.tight_layout(); plt.savefig(f'{save_prefix}_steps.png'); plt.close()

    out_df = pd.DataFrame({
        'timestamp_s': ts_s,
        'ax': ax, 'ay': ay, 'az': az,
        'magnitude': mag,
        'smoothed': smoothed
    })
    return len(peaks), peaks, out_df

def direction_detection(csv_path='TURNING.csv', gyro_col='gyro_z', ts_col='timestamp',
                        window=5, angle_threshold_deg=30, plot=True, save_prefix='part3'):
    df = load_data(csv_path)
    ts = df[ts_col].astype(float).values
    ts_s = ts / 1e9
    gz = df[gyro_col].astype(float).values 

    dt = np.concatenate(([np.median(np.diff(ts_s))], np.diff(ts_s)))
    heading = np.cumsum(gz * dt)
    heading_deg = np.degrees(heading)
    hd_sm = centered_moving_avg_series(pd.Series(heading_deg), window).values

    d = np.diff(hd_sm, prepend=hd_sm[0])
    idxs = np.where(np.abs(d) > 0.5)[0]

    turns = []
    if len(idxs) > 0:
        groups = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
        for g in groups:
            if len(g) == 0:
                continue
            start, end = g[0], g[-1]
            angle = hd_sm[end] - hd_sm[start]
            if abs(angle) < angle_threshold_deg / 2:
                continue

            turns.append((int((start + end) // 2), float(angle)))

    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(ts_s, heading_deg, label='heading_deg', alpha=0.6)
        plt.plot(ts_s, hd_sm, label='smoothed', linewidth=1.2)
        for (idx, ang) in turns:
            plt.axvline(ts_s[idx], color='r', linestyle='--')
        plt.legend()
        plt.title('Direction Detection')
        plt.xlabel('t (s)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_direction.png')
        plt.close()

    return turns

def trajectory_plotting(walk_csv='WALKING_AND_TURNING.csv', step_window=5, k=0.5, step_length=1.0, save_prefix='part4'):
    n_steps, peaks, _ = step_detection(walk_csv, window=step_window, k=k, plot=False)
    df = load_data(walk_csv)
    ts = df['timestamp'].astype(float).values / 1e9
    gz = df['gyro_z'].astype(float).values
    dt = np.concatenate(([np.median(np.diff(ts))], np.diff(ts)))
    heading = np.cumsum(gz * dt)

    x = 0.0; y = 0.0
    xs = [x]; ys = [y]
    for p in peaks:
        idx = min(p, len(heading)-1)
        theta = heading[idx]
        x += step_length * math.cos(theta)
        y += step_length * math.sin(theta)
        xs.append(x); ys.append(y)

    plt.figure(figsize=(5,5))
    plt.plot(xs, ys, '-o')
    plt.title('Trajectory Plotting')
    plt.xlabel('t (s)')
    plt.axis('equal'); plt.grid(True)
    plt.tight_layout(); plt.savefig(f'{save_prefix}_trajectory.png'); plt.close()
    return xs, ys

if __name__ == "__main__":
    print("Part 1:", analyze_acceleration('ACCELERATION.csv'))
    steps, peaks, _ = step_detection('WALKING.csv', window=5, k=0.6)
    print("Steps:", steps)
    print("Turns:", direction_detection('TURNING.csv'))
    trajectory_plotting('WALKING_AND_TURNING.csv', step_window=5, k=0.6)