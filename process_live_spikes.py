import os
import pickle
import numpy as np
import pandas as pd
import sys
import re

# Add parent directories to path to import project modules
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)

from process_signal import load_experimental_data
import read_data.lists_sessions as lists_sessions

def compute_metrics_single_channel(
    spikes_ms, 
    ripples_ms, 
    tolerance=20, 
    max_detection_offset=100, 
    fp_grouping_window=50, # jitter
    extra_tolerance=50
):
    """
    Compute TP, FP, FN for a single channel.
    
    Args:
        spikes_ms: 1D array of spike times in ms.
        ripples_ms: 2D array of ripple intervals [start, end] in ms.
        tolerance: Tolerance in ms (added before ripple start).
        max_detection_offset: Max duration in ms to look for a spike after ripple start.
        fp_grouping_window: Window in ms to group consecutive FP spikes into a single FP event.
    """
    
    if len(ripples_ms) == 0:
        # No ripples: all spikes are FPs
        # Group FPs
        fp_count = 0
        if len(spikes_ms) > 0:
            sorted_spikes = np.sort(spikes_ms)
            current_fp_end = -np.inf
            for spk in sorted_spikes:
                if spk > current_fp_end:
                    fp_count += 1
                    current_fp_end = spk + fp_grouping_window # jitter 
                else:
                    pass
        return 0, fp_count, 0, [] # TP, FP, FN, latencies

    # Define valid detection windows for each ripple
    # Window: [start - tolerance, start + max_detection_offset + tolerance]
    
    ripple_starts = ripples_ms[:, 0]
    valid_windows = []
    for start in ripple_starts:
        w_start = start - tolerance
        w_end = start + max_detection_offset
        valid_windows.append((w_start, w_end))
    
    # Sort spikes
    spikes_ms = np.sort(spikes_ms)
    
    tp_count = 0
    fn_count = 0
    latencies = []
    
    # Track which spikes are used for TPs to exclude them from FP count
    used_spike_indices = set()
    
    # 1. Check for TPs and FNs
    for r_idx, (w_start, w_end) in enumerate(valid_windows):
        # Find spikes in window
        idx_start = np.searchsorted(spikes_ms, w_start) # find first spike >= w_start
        idx_end = np.searchsorted(spikes_ms, w_end) # find first spike > w_end

        in_window_indices = np.arange(idx_start, idx_end) 
        
        if len(in_window_indices) > 0:
            tp_count += 1
            # Calculate latency (first spike relative to ripple start)
            first_spike = spikes_ms[in_window_indices[0]]
            latencies.append(first_spike - ripple_starts[r_idx])
            
            for idx in in_window_indices:
                used_spike_indices.add(idx)
        else:
            fn_count += 1
            
    # 2. Count FPs (grouping remaining spikes)
    fp_count = 0
    current_fp_end = -np.inf
    
    for i, spk in enumerate(spikes_ms):
        if i in used_spike_indices:
            continue
            
        # Check if this spike falls into ANY valid ripple window
        is_in_valid_window = False

        # add extra tolerance to avoid edge cases
        valid_windows_extended = [(w_start, w_end + extra_tolerance) for (w_start, w_end) in valid_windows]
        for w_start, w_end in valid_windows_extended:
            if w_start <= spk <= w_end:
                is_in_valid_window = True
                break
        
        if is_in_valid_window:
            continue
            
        # It's an FP candidate
        if spk > current_fp_end:
            fp_count += 1
            current_fp_end = spk + fp_grouping_window
            
    return tp_count, fp_count, fn_count, latencies

def process_predictions_file(pkl_path, data_path, session, threshold, output_csv=None):
    print(f"Loading predictions from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        session_predictions = pickle.load(f)
    
    results = []
    
    # Get channel
    channel = lists_sessions.channel_sessions.get(session)
    if channel is None:
        print(f"Warning: Channel not found for session {session}")
        return pd.DataFrame()

    # Load GT ripples
    try:
        # verbose=False to suppress output
        _, ripples = load_experimental_data(
            data_path,
            session,
            channel=[channel],
            load_data=False, 
            verbose=False
        )
    except Exception as e:
        print(f"  Error loading data for {session}: {e}")
        return pd.DataFrame()

    if ripples is None:
        ripples_ms = np.empty((0, 2))
    else:
        # Ripples are in 30kHz samples. Convert to ms.
        ripples_ms = ripples / 30.0

    Fs = 1250 # Hz

    for model_name, detections in session_predictions.items():
        # detections is Nx2 array of start/end indices at 1250 Hz
        if len(detections) == 0:
            spikes_ms = np.array([])
        else:
            # Use start of interval
            # middle_samples = np.mean(detections, axis=1)
            # spikes_ms = (middle_samples / 1250.0) * 1000.0 # ms
            start_samples = detections[:, 0]
            spikes_ms = (start_samples / Fs) * 1000.0 # ms

        tp, fp, fn, latencies = compute_metrics_single_channel(
            spikes_ms, 
            ripples_ms, 
            tolerance=20, 
            max_detection_offset=100, 
            fp_grouping_window=100,
            extra_tolerance=100
        )
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_latency = np.mean(latencies) if latencies else np.nan
        
        print(f"  Model: {model_name} -> TP: {tp}, FP: {fp}, FN: {fn} -> F1: {f1:.4f}")
        
        results.append({
            "Session": session,
            "Channel": channel,
            "Threshold": threshold,
            "Model": model_name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Mean_Latency": mean_latency,
            "Num_Spikes": len(spikes_ms),
            "Num_Ripples": len(ripples_ms)
        })
        
    df = pd.DataFrame(results)
    
    if output_csv and not df.empty:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
    return df

if __name__ == "__main__":
    # Configuration
    PREDICTIONS_ROOT = os.path.join(curr_dir, "detections")
    DATA_PATH = r"C:\Madrid_tests"
    OUTPUT_FILE = os.path.join(PREDICTIONS_ROOT, "all_networks_metrics_ripplAI.csv")
    
    print(f"Searching for prediction files in: {PREDICTIONS_ROOT}")
    
    all_dfs = []
    
    for root, dirs, files in os.walk(PREDICTIONS_ROOT):
        for file in files:
            if file.endswith(".pkl") and "SWR_detections_" in file:
                # Parse session and threshold
                # Format: SWR_detections_{session}_th{threshold}.pkl
                # Regex
                match = re.match(r"SWR_detections_(.+)_th([\d\.]+)\.pkl", file)
                if match:
                    session = match.group(1)
                    threshold = float(match.group(2))
                    
                    pkl_path = os.path.join(root, file)
                    
                    # Output CSV per session/threshold
                    session_csv = os.path.join(root, f"{session}_metrics_{threshold}.csv")
                    
                    try:
                        df = process_predictions_file(
                            pkl_path,
                            DATA_PATH,
                            session,
                            threshold,
                            output_csv=None
                        )
                        all_dfs.append(df)
                    except Exception as e:
                        print(f"Failed to process {file}: {e}")
                        import traceback
                        traceback.print_exc()

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save aggregated results
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nAll results aggregated and saved to: {OUTPUT_FILE}")
        
        # Print summary
        print("\n--- Summary by Threshold ---")
        summary = final_df.groupby("Threshold")[["F1", "Precision", "Recall"]].mean()
        print(summary)
    else:
        print("No prediction files found.")
