from process_signal import load_experimental_data
import read_data.lists_sessions as lists_sessions
import rippl_AI
import numpy as np
import pickle
import os
import gc
from argparse import ArgumentParser


def run_inference(data_path, session, channels_sessions=None, threshold=0.5):
    save_path = "probabilities"
    os.makedirs(save_path, exist_ok=True)
    probs_file = os.path.join(save_path, f"SWR_prob_{session}.pkl")

    if os.path.exists(probs_file):
        print(f"Loading probabilities from {probs_file}")
        with open(probs_file, "rb") as f:
            all_probs = pickle.load(f)
    else:
        # Load experimental data
        if channels_sessions is None:
            raise ValueError("channels_sessions must be provided")

        channel=channels_sessions.get(session,None)-1
        shank=channel//8
        channel_within_shank=channel%8
        channels=np.arange(shank*8,shank*8+8)
        signal_og, ripples = load_experimental_data(data_path, session, channel=channels,normalize=False,scale_data=False)
        model_names=["CNN1D", "CNN2D", "LSTM", "SVM", "XGBOOST"]
        # Run inference
        all_probs = {}
        for model_name in model_names:
            for i in range(1,6):
                try:
                    signal=signal_og.copy()
                    SWR_prob, SWR_norm=rippl_AI.predict(signal, 30000, arch=model_name, model_number=i, channels=np.arange(8), d_sf=1250)
                    all_probs[f"{model_name}_{i}"] = SWR_prob
                    
                    # --- ADDED CLEANUP ---
                    del signal
                    gc.collect()
                    # If using Keras, uncomment the line below to clear GPU/CPU memory
                    # K.clear_session() 
                    # ---------------------

                except Exception as e:
                    try:
                        print(f"Error with model {model_name} number {i} using all channels: {e}. Trying subset of channels.")
                        # signal.shape = (n_samples, n_channels)
                        n_channels_shank = signal.shape[1]  # channels per shank
                        center = channel_within_shank  # 0-based

                        # Determine start and end indices for 3 channels around center
                        start_idx = max(center - 1, 0)          # don't go below 0
                        end_idx = start_idx + 3                 # want 3 channels total

                        # Adjust if end_idx exceeds shank boundary
                        if end_idx > n_channels_shank:
                            end_idx = n_channels_shank
                            start_idx = end_idx - 3             # keep 3 channels
                            start_idx = max(start_idx, 0)      # just in case shank < 3 channels

                        # Slice the signal
                        signal_subset = signal[:, start_idx:end_idx]    
                        print(signal_subset.shape)
                        SWR_prob, SWR_norm=rippl_AI.predict(signal_subset, 30000, arch=model_name, model_number=i, channels=np.arange(3), d_sf=1250)
                        all_probs[f"{model_name}_{i}"] = SWR_prob
                    except Exception as e2:
                        print(f"Error with model {model_name} number {i}: {e2}")
        
        with open(probs_file, "wb") as f:
            pickle.dump(all_probs, f)
        print(f"Saved SWR probabilities to {probs_file}")

    # Generate detections
    all_detections = {}
    for key, prob in all_probs.items():
        detections = rippl_AI.get_predictions_index(prob, threshold=threshold)
        all_detections[key] = detections

    save_path_det = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections",str(threshold))
    if not os.path.exists(save_path_det):
        os.makedirs(save_path_det,exist_ok=True)

    with open(os.path.join(save_path_det,f"SWR_detections_{session}_th{threshold}.pkl"), "wb") as f:
        pickle.dump(all_detections, f)
    print(f"Saved SWR detections to SWR_detections_{session}_th{threshold}.pkl")

def parse_args():
    parser = ArgumentParser(description="Run SWR detection inference on experimental data.")
    parser.add_argument("--session","-s", type=str, required=True, help="Session identifier to process.")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Detection threshold.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    threshold=args.threshold
    session=args.session
    data_path=r"C:\Madrid_tests"
    # sessions=lists_sessions.annotated_sessions

    # Original Sessions 
    session_list = [
        "2025-09-22_17-55-26",  # R
        "2025-09-23_15-50-26",  # R
        "2025-09-24_10-24-40",  # R
        "2025-09-24_14-22-55",  # H
        "2025-09-24_15-13-10",  # H
        "2025-09-25_16-41-14",  # R # OG

        "2025-09-24_16-29-07",  # R
        "2025-09-24_17-38-17",  # R
        "2025-09-22_17-42-27",
        "2025-09-23_16-17-52",
        "2025-09-24_11-34-51",
        "2025-09-25_11-21-53",
        "2025-09-25_12-52-22",
    ]
    channel_sessions=lists_sessions.channel_sessions

    print(f"Processing session: {session}, Threshold: {threshold}")
    try:
        run_inference(data_path, session, channels_sessions=channel_sessions, threshold=threshold)
    except Exception as e:
        print(f"Error processing session {session}: {e}")
        raise e
