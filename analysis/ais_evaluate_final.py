import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm

warnings.filterwarnings("ignore")


CONFIG = {
    "CSJ": {
        "dt": 60,
        "source_dt": 60,
        "obs_len": 90,
        "pred_len": 15,
        "max_speed_mps": 15.0,
        "kde_bandwidth": 0.5,
        "entropy_samples": 50,
        "context_radius": 5000.0,
        "time_tolerance": 30.0,
        "files": [
            "./data_unsampling/final_1min_csj_train.parquet",
            "./data_unsampling/final_1min_csj_test.parquet",
            "./data_unsampling/final_1min_csj_val.parquet",
        ],
    },
    "ZS": {
        "dt": 60,
        "source_dt": 60,
        "obs_len": 90,
        "pred_len": 15,
        "max_speed_mps": 15.0,
        "kde_bandwidth": 0.5,
        "entropy_samples": 50,
        "context_radius": 5000.0,
        "time_tolerance": 30.0,
        "files": [
            "./data_unsampling/final_1min_zs_train.parquet",
            "./data_unsampling/final_1min_zs_test.parquet",
            "./data_unsampling/final_1min_zs_val.parquet",
        ],
    },
}


def load_and_prep_data(file_paths):
    dfs = []
    print("Loading AIS data...")

    for file_idx, path in enumerate(file_paths):
        if not os.path.exists(path):
            print(f"  [skip] missing file: {path}")
            continue

        cols = ["segment_id", "timestamp", "x", "y", "v_mps"]
        try:
            df = pd.read_parquet(path, columns=cols)
        except Exception as exc:
            print(f"  [error] failed to read {path}: {exc}")
            continue

        df["segment_id"] = df["segment_id"].astype(str) + f"_f{file_idx}"
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        dfs.append(df)

    if not dfs:
        raise ValueError("No parquet files could be loaded.")

    return pd.concat(dfs, ignore_index=True).sort_values(["segment_id", "timestamp"])


def build_trajlets(df, cfg):
    step_factor = int(cfg["dt"] / cfg["source_dt"])
    target_total = cfg["obs_len"] + cfg["pred_len"]
    raw_total_len = target_total * step_factor
    expected_seconds = np.arange(target_total) * cfg["dt"]
    time_tolerance = cfg["time_tolerance"]
    trajlets = []

    for _, group in tqdm(df.groupby("segment_id"), desc="Building trajlets"):
        if len(group) < raw_total_len:
            continue

        data_x = group["x"].to_numpy()
        data_y = group["y"].to_numpy()
        data_v = group["v_mps"].to_numpy()
        data_t = group["timestamp"].to_numpy()
        stride = max(1, raw_total_len // 2)

        for start_idx in range(0, len(group) - raw_total_len + 1, stride):
            idx = np.arange(start_idx, start_idx + raw_total_len, step_factor)[:target_total]
            if len(idx) < target_total:
                continue

            time_offsets = (
                (data_t[idx] - data_t[idx[0]]) / np.timedelta64(1, "s")
            ).astype(float)
            if np.max(np.abs(time_offsets - expected_seconds)) > time_tolerance:
                continue

            window_v = data_v[idx]
            if np.any(~np.isfinite(window_v)):
                continue
            if np.any(window_v < 0.0) or np.any(window_v > cfg["max_speed_mps"]):
                continue

            window_x = data_x[idx]
            window_y = data_y[idx]
            obs_x = window_x[: cfg["obs_len"]]
            obs_y = window_y[: cfg["obs_len"]]
            pred_x = window_x[cfg["obs_len"] :]
            pred_y = window_y[cfg["obs_len"] :]

            trajlets.append(
                {
                    "obs": np.column_stack((obs_x, obs_y)),
                    "pred": np.column_stack((pred_x, pred_y)),
                    "rel_obs": np.column_stack((obs_x - obs_x[0], obs_y - obs_y[0])),
                    "rel_pred": np.column_stack((pred_x - obs_x[0], pred_y - obs_y[0])),
                    "v": window_v,
                    "meta": {
                        "track_id": group["segment_id"].iloc[0],
                        "timestamp": np.datetime64(data_t[idx[cfg["obs_len"] - 1]]),
                        "query_pos": (float(obs_x[-1]), float(obs_y[-1])),
                    },
                }
            )

    return trajlets


def compute_lateral_deviation(traj):
    if len(traj) < 3:
        return 0.0

    start = traj[0]
    end = traj[-1]
    chord = end - start
    chord_len = np.linalg.norm(chord)

    if chord_len < 1e-6:
        path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        if path_len < 1e-6:
            return 0.0
        return float(np.mean(np.linalg.norm(traj[1:-1] - start, axis=1)) / path_len)

    offsets = traj[1:-1] - start
    cross = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0])
    distances = cross / chord_len
    path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    if path_len < 1e-6:
        return 0.0
    return float(np.mean(distances) / path_len)


def wrap_angle_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def calc_entropy(trajlets, cfg):
    print("Computing predictability...")

    if len(trajlets) < 2:
        return pd.DataFrame({"entropy": [np.nan] * len(trajlets)})

    obs_feats = np.array([t["rel_obs"].flatten() for t in trajlets], dtype=float)
    pred_feats = np.array([t["rel_pred"].flatten() for t in trajlets], dtype=float)

    obs_scale = obs_feats.std(axis=0).clip(min=1.0)
    pred_scale = pred_feats.std(axis=0).clip(min=1.0)
    obs_norm = obs_feats / obs_scale
    pred_norm = pred_feats / pred_scale

    h = float(cfg["kde_bandwidth"])
    sample_count = int(cfg["entropy_samples"])
    pred_dim = pred_norm.shape[1]
    rng = np.random.default_rng(42)
    log_norm_const = -0.5 * pred_dim * np.log(2.0 * np.pi * h * h)

    entropies = []
    for i in tqdm(range(len(trajlets)), desc="Predictability"):
        diff_obs = obs_norm - obs_norm[i]
        sq_obs = np.sum(diff_obs * diff_obs, axis=1) / (2.0 * h * h)
        log_w = -sq_obs
        log_w[i] = -np.inf

        if not np.isfinite(log_w).any():
            entropies.append(np.nan)
            continue

        valid_mask = np.isfinite(log_w)
        valid_idx = np.flatnonzero(valid_mask)
        log_w_valid = log_w[valid_mask]
        log_w_valid -= log_w_valid.max()
        weights = np.exp(log_w_valid)
        weights /= weights.sum()

        chosen_idx = rng.choice(valid_idx, size=min(sample_count, len(valid_idx)), p=weights)
        samples = pred_norm[chosen_idx] + rng.standard_normal((len(chosen_idx), pred_dim)) * h
        pred_bank = pred_norm[valid_idx]
        log_weight_terms = np.log(weights + 1e-300)

        log_p_values = []
        for sample in samples:
            diff_pred = pred_bank - sample
            sq_pred = np.sum(diff_pred * diff_pred, axis=1) / (2.0 * h * h)
            log_terms = log_weight_terms + log_norm_const - sq_pred
            max_term = np.max(log_terms)
            log_p_values.append(float(max_term + np.log(np.sum(np.exp(log_terms - max_term)))))

        entropies.append(-float(np.mean(log_p_values)))

    return pd.DataFrame({"entropy": entropies})


def calc_regularity(trajlets, dt):
    print("Computing regularity...")
    results = []

    for trajlet in tqdm(trajlets, desc="Regularity"):
        full_traj = np.concatenate([trajlet["obs"], trajlet["pred"]], axis=0)
        vel = np.diff(full_traj, axis=0)
        step_dist = np.linalg.norm(vel, axis=1)
        speed_series = step_dist / dt if len(step_dist) > 0 else np.array([], dtype=float)

        dist_direct = np.linalg.norm(full_traj[-1] - full_traj[0])
        dist_path = np.sum(step_dist)
        efficiency = dist_direct / dist_path if dist_path > 1.0 else 1.0

        if len(speed_series) == 0:
            speed_avg = 0.0
            speed_range = 0.0
            acc_max = 0.0
        else:
            speed_avg = float(np.mean(speed_series))
            speed_range = float(np.max(speed_series) - np.min(speed_series))
            acc_max = (
                float(np.max(np.abs(np.diff(speed_series) / dt)))
                if len(speed_series) > 1
                else 0.0
            )

        if len(vel) == 0:
            angle_dev = 0.0
        else:
            step_norm = np.linalg.norm(vel, axis=1)
            valid_steps = step_norm > 1e-8
            if np.count_nonzero(valid_steps) >= 2:
                headings = np.arctan2(vel[valid_steps, 1], vel[valid_steps, 0])
                heading_delta = wrap_angle_to_pi(np.diff(headings))
                angle_dev = float(np.sum(np.abs(heading_delta))) if len(heading_delta) else 0.0
            else:
                angle_dev = 0.0

        results.append(
            {
                "speed_avg": speed_avg,
                "speed_range": speed_range,
                "acc_max": acc_max,
                "efficiency": float(efficiency),
                "angle_dev": angle_dev,
                "d_perp": compute_lateral_deviation(full_traj),
            }
        )

    return pd.DataFrame(results)


def calc_context(trajlets, df_full, radius, time_tolerance):
    print("Computing context complexity...")

    coords = df_full[["x", "y"]].to_numpy(dtype=float)
    times = df_full["timestamp"].to_numpy()
    track_ids = df_full["segment_id"].to_numpy()
    tree = BallTree(coords, metric="euclidean")

    densities = []
    nnd_list = []

    for trajlet in tqdm(trajlets, desc="Context"):
        qx, qy = trajlet["meta"]["query_pos"]
        q_time = trajlet["meta"]["timestamp"]
        q_id = trajlet["meta"]["track_id"]

        indices = tree.query_radius([[qx, qy]], r=radius)[0]
        if len(indices) == 0:
            densities.append(0)
            nnd_list.append(float(radius))
            continue

        t_diff = np.abs((times[indices] - q_time) / np.timedelta64(1, "s"))
        valid = indices[(t_diff <= time_tolerance) & (track_ids[indices] != q_id)]
        if len(valid) == 0:
            densities.append(0)
            nnd_list.append(float(radius))
            continue

        neighbor_dist = np.linalg.norm(coords[valid] - np.array([qx, qy]), axis=1)
        densities.append(int(np.unique(track_ids[valid]).size))
        nnd_list.append(float(np.min(neighbor_dist)))

    return pd.DataFrame({"local_density": densities, "nnd": nnd_list})


def run_evaluation(region_name):
    cfg = CONFIG[region_name]
    print(f"\n{'=' * 40}\nEvaluating region: {region_name}\n{'=' * 40}")

    df_all = load_and_prep_data(cfg["files"])
    trajlets = build_trajlets(df_all, cfg)
    print(f"Built {len(trajlets)} valid trajlets.")

    df_reg = calc_regularity(trajlets, cfg["dt"])
    df_ctx = calc_context(
        trajlets,
        df_all,
        radius=cfg["context_radius"],
        time_tolerance=cfg["time_tolerance"],
    )
    df_ent = calc_entropy(trajlets, cfg)
    df_meta = pd.DataFrame(
        {
            "track_id": [t["meta"]["track_id"] for t in trajlets],
            "timestamp": pd.to_datetime([t["meta"]["timestamp"] for t in trajlets]),
        }
    )

    final_df = pd.concat([df_meta, df_reg, df_ctx, df_ent], axis=1)
    output_file = f"opentraj_metrics_{region_name}_90_15_full.csv"
    final_df.to_csv(output_file, index=False)

    summary_cols = [
        "speed_avg",
        "speed_range",
        "acc_max",
        "efficiency",
        "angle_dev",
        "d_perp",
        "local_density",
        "nnd",
        "entropy",
    ]
    print(f"\n{region_name} summary:")
    print(final_df[summary_cols].describe(percentiles=[0.95]).loc[["mean", "50%", "95%", "max"]])
    print(f"Saved metrics to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AIS trajectories with adapted OpenTraj metrics.")
    parser.add_argument(
        "-r",
        "--region",
        choices=["CSJ", "ZS"],
        required=True,
        help="Region to evaluate.",
    )
    args = parser.parse_args()
    run_evaluation(args.region)
