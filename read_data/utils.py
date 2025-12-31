import numpy as np

def up_down_channel(signal,threshold,downsampled_fs,refractory=0,initial_value=None,return_value=False):
    # Define parameters
    # print("Threshold=",threshold)
    num_timesteps = len(signal)
    spikified = np.zeros((num_timesteps, 2 ))
    if initial_value is not None:	
        value = initial_value
    else:
        value=signal[0]
    refractory_samples = int(refractory*downsampled_fs)
    
    if refractory_samples == 0:
        refractory_samples = 1

    i = 0
    # print("Max Signal:", max(signal),"\n Min Signal:",min(signal))
    while i < num_timesteps:
        delta = signal[i] - value
        if delta >= threshold:
            spikified[i,0] = 1
            value = signal[i]
            i += refractory_samples  # skip refractory period
            # print(delta)
        elif delta <= -threshold:
            spikified[i,1] = 1
            value = signal[i]
            i += refractory_samples  # skip refractory period    
            # print(delta)
        else:
            i += 1  # no spike, move to next time step
    if return_value:
        return spikified, value
    else:
        return spikified
    
def extract_spikes_downsample(spike_train,factor,verbose=False):
    """
    Downsamples a binary (UP/DOWN) spike train from original_freq to target_freq
    keeping at most 1 spike per ms. Chooses the direction with more spikes in each window.
    
    Parameters:
        spike_train (np.ndarray): Shape (n_samples, 2), binary values for UP and DOWN
        original_freq (int): Original sampling rate (default: 30000 Hz)
        target_freq (int): Target sampling rate (default: 1000 Hz)
    
    Returns:
        np.ndarray: Downsampled spike train (shape: n_bins, 2)
    """

    n_samples = spike_train.shape[0]
    n_bins = n_samples // factor
    
    # Trim excess samples if needed
    trimmed = spike_train[:n_bins * factor]
    # Total spikes before downsampling
    total_spikes_before = np.sum(trimmed)
    # Reshape to (n_bins, factor, 2)
    # Each row corresponds to 1 ms window with 30 time points of 2D spikes (UP/DOWN)
    reshaped = trimmed.reshape(n_bins, factor, 2)
    
    # Sum UP and DOWN spikes within each bin
    up_sum = reshaped[:, :, 0].sum(axis=1)
    down_sum = reshaped[:, :, 1].sum(axis=1)
    
    # Allocate result array
    result = np.zeros((n_bins, 2), dtype=int)
    
    # Assign dominant spike direction
    result[up_sum > down_sum, 0] = 1  # UP spike
    result[down_sum > up_sum, 1] = 1  # DOWN spike
    # If equal or both zero → remains [0, 0]
    # Total spikes after downsampling
    total_spikes_after = np.sum(result)
    # Print lost spike count
    spikes_lost = total_spikes_before - total_spikes_after
    if verbose:
        print(f"Total spikes before: {total_spikes_before}")
        print(f"Total spikes after: {total_spikes_after}")
        print(f"Spikes lost during downsampling: {spikes_lost}")

    return result,spikes_lost

# Based on https://github.com/kburel/snn-hfo-detection/blob/main/snn_hfo_detection/functions/signal_to_spike/utility.py#L43
def calculate_threshold(signal,downsampled_fs,window_size,sample_ratio,scaling_factor,plot=False,verbose=False):
    times=np.arange(0, len(signal)) / downsampled_fs  # Time in seconds 
    min_time = np.min(times)
   
    if np.min(times) < 0:
        raise ValueError(
            f'Tried to find thresholds for a dataset with a negative time: {min_time}')
    duration = np.max(times) - min_time
    if verbose:
        print(f"Duration of the signal: {duration} seconds, between {np.min(times)} and {np.max(times)}")
        
    if duration <= 0:
        raise ValueError(
            f'Tried to find thresholds for a dataset with a duration that under or equal to zero. Got duration: {duration}')

    if len(signal) == 0:
        raise ValueError('signals is not allowed to be empty, but was'
                         )
    if len(times) == 0:
        raise ValueError('times is not allowed to be empty, but was')

    if len(signal) != len(times):
        raise ValueError(
            f'signals and times need to have corresponding indices, but signals has length {len(signal)} while times has length {len(times)}')

    if not 0 < sample_ratio < 1:
        raise ValueError(
            f'sample_ratio must be a value between 0 and 1, but was {sample_ratio}'
        )

    num_timesteps = int(np.ceil(duration / window_size))
    if verbose:
        print(f"Number of time steps: {num_timesteps} for window size {window_size} seconds")
    max_min_amplitude = np.zeros((num_timesteps, 2))
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=duration, step=window_size)):
        interval_end = interval_start + window_size
        index = np.where((times >= interval_start) & (times <= interval_end))
        max_amplitude = np.max(signal[index])
        min_amplitude = np.min(signal[index])
        max_min_amplitude[interval_nr, 0] = max_amplitude
        max_min_amplitude[interval_nr, 1] = min_amplitude
   
    variation=np.abs(max_min_amplitude[:,0]-max_min_amplitude[:,1])
    sorted_variation = np.sort(variation)
    chosen = max(1, int(len(sorted_variation) * sample_ratio))
    threshold=scaling_factor*np.mean(sorted_variation[:chosen])
        
    return threshold