from read_data.read_data import read_data
import os
from read_data.signal_aid import bandpass_filter
from read_data.utils import *
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import butter, filtfilt, hilbert

def load_experimental_data(path,name, downsample = False, normalize = False, numSamples = False, 
                           start = 0, verbose=True, original_fs=30000,channel=None,
                           invert=False,offset=0.16,load_data=True,scale_data=False):
    
    try:
        session_date = datetime.strptime(name.split('_')[0], "%Y-%m-%d").day
    except Exception as e:
        print(f"⚠️ Skipping {name} - invalid format: {e}")
        return None
    
    
    # determine invert rule
    invert = session_date >= 24

    liset=read_data(path,name, downsample = downsample, normalize = normalize, numSamples = numSamples, 
                           start = start, verbose=verbose, original_fs=original_fs,channel_num=None,
                           invert=invert,offset=offset,load_data=load_data,channels=channel,scale_data=scale_data)
    
    filtered_signal = None
    if load_data:
        # channel_data= liset.data[:].reshape(-1)
        liset_data= liset.data
        # filtered_signal=bandpass_filter(channel_data, bandpass=[100,250], fs=liset.fs, order=4)
    ripples=liset.annotated.ripples_GT #frequency here...

    return liset_data, ripples

# Double Checked - should work okay and return a spikified signal in the shape [n_samples(ms), 2 (UP/DN)] 

def spikify_signal(
    signal,
    fs,
    time_max=20.0,
    overlap=0.5,
    adapt_threshold=True,
    percentile=False,
    window_size=0.10,
    sample_ratio=0.25,
    scaling_factor=1.0,
    refractory=0,
    factor=30,
    initial_value=None,
    verbose=False,
    ripples=None,     # kept for compatibility but unused
):

    N = len(signal)
    out_len = N // factor if factor > 1 else N
    spikified = np.zeros((out_len, 2))

    win = int(fs * time_max)
    step = int(fs * overlap * time_max)

    # -------- THRESHOLD HELPERS --------
    def compute_threshold(x):
        return calculate_threshold(x, fs, window_size, sample_ratio, scaling_factor)

    def get_threshold_window(signal, t):
        """Sliding window for adaptive threshold."""
        if t < win:
            return signal[:win]
        else:
            return signal[t - win:t]

    if verbose:
        print(f"[spikify] N={N}, out_len={out_len}, win={win}, step={step}, factor={factor}")

    # =============================
    # ADAPTIVE THRESHOLD MODE
    # =============================
    if adapt_threshold:

        thresholds = []

        for t in range(0, N, step):

            # --- sliding threshold window ---
            tw = get_threshold_window(signal, t)

            thr = compute_threshold(tw)
            thresholds.append(thr)

            # --- extract chunk ---
            r_edge = min(t + step, N)
            chunk = signal[t:r_edge]

            # --- spiking ---
            spk, initial_value = up_down_channel(
                chunk, thr, fs, refractory,
                initial_value=initial_value, return_value=True
            )

            # Downsample spikes if using factor
            if factor > 1:
                spk, _ = extract_spikes_downsample(spk, factor)

            L = t // factor
            R = r_edge // factor
            spikified[L:R] = spk

        if verbose:
            print("Spikification complete.")
            print(f"UP spikes:   {np.sum(spikified[:,0])}")
            print(f"DOWN spikes: {np.sum(spikified[:,1])}")

        return spikified, thresholds

    # =============================
    # FIXED THRESHOLD MODE
    # =============================
    else:
        thr = compute_threshold(signal[:win])

        spk = up_down_channel(signal, thr, fs, refractory,
                              initial_value=None, return_value=False)

        if factor > 1:
            spk, _ = extract_spikes_downsample(spk, factor)

        return spk, thr


def ripple_band_power_trace(signal, fs, bandpass=(100, 250), smooth_ms=10, log_power=False,zscore=False):
    """
    Compute continuous ripple-band power over time.

    Parameters
    ----------
    signal : np.ndarray
        1D LFP trace.
    fs : float
        Sampling frequency (Hz).
    ripple_band : tuple
        Ripple frequency range (e.g., (120, 250)).
    smooth_ms : float
        Window for moving average smoothing (in milliseconds).
    log_power : bool
        If True, return log10(power) instead of linear power.

    Returns
    -------
    power_trace : np.ndarray
        Ripple-band power time series (same length as signal).
    """

    # Bandpass filter
    # nyq = fs / 2
    # b, a = butter(4, [bandpass[0]/nyq, bandpass[1]/nyq], btype='band')
    # filtered = filtfilt(b, a, signal)
    filtered=signal  # Already filtered
    # Hilbert transform to get analytic envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    # Compute power
    power = envelope ** 2
    if log_power:
        power = np.log10(power + 1e-12)

     # ----- Optional z-score normalization -----
    if zscore:
        mu = np.mean(power)
        sigma = np.std(power)
        if sigma > 0:
            power = (power - mu) / sigma
        else:
            power = power - mu   # avoid division by zero
            
    # 4 Smooth (optional)
    if smooth_ms > 0:
        win_samples = int(fs * smooth_ms / 1000)
        if win_samples > 1:
            kernel = np.ones(win_samples) / win_samples
            power = np.convolve(power, kernel, mode='same')

    return power

def plot_signal_spikes(
    signal,
    spikes,
    fs_signal=30000,
    fs_spikes=1000,
    ripples=None,
    window=None,
    ripple_color="yellow",
    ripple_alpha=0.25,
    figsize=(15,5)
):
    """
    signal:      filtered LFP already aligned to window (shape N)
    fs_signal:   sampling rate of signal (e.g., 30000)
    spikes:      spikified data already aligned (N_spikes, 2)
    fs_spikes:   sampling rate of spikes (e.g., 1000)
    ripples:     ripple timestamps in *seconds* relative to whole session
    window:      (start_s, end_s) used only to determine which ripples appear
    """

    w_start, w_end = window

    # ------------------------------------------------------
    # Build time axes — these are already correct
    # ------------------------------------------------------
    t_sig = np.arange(len(signal)) / fs_signal        # in seconds
    t_spk = np.arange(len(spikes)) / fs_spikes        # in seconds

    # Convert both to milliseconds for nicer plotting
    t_sig_ms = t_sig * 1000
    t_spk_ms = t_spk * 1000

    plt.figure(figsize=figsize)

    # ------------------------------------------------------
    # Plot the LFP signal
    # ------------------------------------------------------
    plt.plot(t_sig_ms, signal, color="black", lw=0.8, label="Filtered LFP")

    # ------------------------------------------------------
    # Plot UP/DOWN spikes (already aligned)
    # ------------------------------------------------------
    up_times  = t_spk_ms[spikes[:,0] > 0]
    dn_times  = t_spk_ms[spikes[:,1] > 0]

    # Place spikes above the signal amplitude
    ymax = np.max(signal)
    ymin= np.min(signal)

    plt.vlines(up_times, ymax*0.75, ymax*1, color="red", lw=1, label="UP")

    plt.vlines(dn_times, ymin, ymin*0.75, color="blue", lw=1,label="DOWN")

    # ------------------------------------------------------
    # Plot ripples (now relative to window start)
    # ------------------------------------------------------
    for idx, r in enumerate(ripples):

        # Single timestamp or start/end?
        if isinstance(r, (int, float)):
            r_start = r
            r_end   = r
        else:
            r_start, r_end = r

        # Skip outside the window
        if r_end < w_start or r_start > w_end:
            continue

        # Convert to milliseconds relative to window start
        rs_ms = (r_start - w_start) * 1000
        re_ms = (r_end   - w_start) * 1000

        plt.axvspan(
            rs_ms, re_ms,
            color=ripple_color,
            alpha=ripple_alpha,
            label="Ripple" if idx == 0 else None
        )

    # ------------------------------------------------------
    # Formatting
    # ------------------------------------------------------
    plt.xlabel("Time (ms, relative to window start)")
    plt.ylabel("Filtered LFP amplitude")
    plt.title("Signal + UP/DN Spikes + Ripples")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.xlim(0, (w_end - w_start)*1000)  # in ms
