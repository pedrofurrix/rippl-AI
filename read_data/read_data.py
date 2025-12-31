###########################################################################################
#                                                                                         #
#                         Developed by: Pedro Félix Alves                                 #
#                           Contact: pedrofelixalves@gmail.com                            #
#                                                                                         #
###########################################################################################

#
#  This module is developed for 3 main reasons:
#           - Loading correctly the data obtained from the Open Ephys system.
#           - Visualizing performance and data quality.
#           - Assisting in the analysis of ripple events.

import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from copy import deepcopy
from read_data.liset_aux import *
from read_data.load_data import *
from read_data.signal_aid import *
import pickle as pkl




class read_data():
    """
    Class for handling data processing and visualization related to ripple events.

    Parameters:
    - data_path (str): Path to the directory containing the data.
    - shank (int): Shank of the electrode.
    - downsample (bool, optional): Whether to downsample the data. Default is False.
    - normalize (bool, optional): Whether to normalize the data. Default is True.
    - numSamples (int, optional): Number of samples. Default is False.
    - verbose (bool, optional): Whether to display verbose output. Default is True.
    """

    def __init__(self, data_path, name, shank=0, downsample = False, normalize = True, numSamples = False, start = 0, verbose=True, original_fs=30000,channel_num=None,invert=False,offset=0.16,buffer_size=20,load_data=True, channels=None,scale_data=False):
        
        self.oe_path=os.path.join(data_path,"Open Ephys",name)
        self.annotation_path=os.path.join(data_path,"annotations")
        self.offline_path=os.path.join(data_path,"offline",f"{buffer_size}")
        
        self.scale_data=scale_data
        # Set the verbose
        self.verbose = verbose
        if self.verbose:
            print(f"Loading data from session: {name}")
        self.numSamples = numSamples
        self.start = start
        self.original_fs = original_fs
        self.annotated=RippleEvents(offset=offset)
        self.from_data=RippleEvents(offset=offset)
        self.normalize=normalize
        self.name=name
        self.channels=channels
        if downsample:
            self.downsampled_fs = downsample
            self.fs_conv_fact = self.original_fs/self.downsampled_fs
        else:
            self.fs_conv_fact = 1
            self.downsampled_fs = self.original_fs

        if channel_num is None:
            self.get_channel_num(self.oe_path)
        else:
            self.channel_num=channel_num

        # Load the data.
        if load_data:
            self.load(self.oe_path, shank, downsample = downsample, normalize=normalize,invert=invert, channels=channels)
        else:
            self.load_only_ripples(self.oe_path,invert=invert)    

        self._check_data()
        if self.verbose:
            print(f"✅ Loading complete!- session {name}")

    def load_only_ripples(self,path,invert=False):
        try:
            list_dir=os.listdir(path)
            file=list_dir[1]
            record_file=os.path.join(path,file,"experiment1","recording1","continuous")
            list_2=os.listdir(record_file)
            folder=os.path.join(record_file,list_2[0])
            timestamps_file= os.path.join(folder,"timestamps.npy")
            self.timestamps = np.load(timestamps_file)
        except:
            if self.verbose:
                print('❗timestamps.npy file not in path❗, cannot load ripples.')
            return False
        self.numSamples=np.inf
        self.fs = self.downsampled_fs
        
        self.load_annotations(self.annotation_path,self.name)
        self.load_ripple_times(self.oe_path,invert)
        self.load_offline()

    def ripples_in_chunk(self, ripples, start, numSamples, fs, prop):
        if not numSamples:
            numSamples = self.file_samples - self.start

        in_chunk = ripples[(ripples[:,0] > start/prop/fs) & (ripples[:,0] < (start + numSamples)/prop/fs)]

        return in_chunk


    def load_dat(self, path, channels, numSamples = False, verbose=False):
        """
        Load data from a .dat file.

        Parameters:
        - path (str): Path to the directory containing the .dat file.
        - channels (list): Lis  t of channel IDs to load.
        - numSamples (int, optional): Number of samples to load. Default is False (load all samples).
        - sampleSize (int, optional): Size of each sample in bytes. Default is 2.
        - verbose (bool, optional): Whether to display verbose output. Default is False.

        Returns:
        - data (numpy.ndarray): Loaded data as a NumPy array.
        """
        try:
            list_dir=os.listdir(path)
            file=list_dir[1]
            record_file=os.path.join(path,file,"experiment1","recording1","continuous")
            list_2=os.listdir(record_file)
            folder=os.path.join(record_file,list_2[0])
            filename=os.path.join(folder,"continuous.dat")
            self.file_len = os.path.getsize(filename=filename)
            self.file_samples = self.file_len / self.channel_num / 2
            timestamps_file= os.path.join(folder,"timestamps.npy")
            self.timestamps = np.load(timestamps_file)
            if self.timestamps[0]>0:
                print(f"Warning: The first timestamp is not zero, there might be an offset in the data - {self.timestamps[0]}")
            else:
                print("Starting from 0 :D")
        except:
            if self.verbose:
                print('❗timestamps.npy file not in path❗, cannot load ripples/data.')
            return False
        
        nChannels = len(channels)
        if (len(channels) > nChannels):
            if self.verbose:
                print("Cannot load specified channels (listed channel IDs inconsistent with total number of channels).")
            return False
        
        start = self.start * self.channel_num * 2
        numSamples = self.numSamples * self.channel_num * 2

        if start > self.file_len:
            if self.verbose:
                print(f'the start must be lower than the total file samples.\nTotal file samples: {self.file_samples}')
            return False
        if (numSamples + start) > self.file_len:
            numSamples = self.file_len - start


        if (self.file_len < numSamples) or ((numSamples + self.start) > self.file_len):
            if self.verbose:
                print(f'file has only {self.file_samples} samples')
            return False
            
        with open(filename, "rb") as f:
            # Put the reader at the starting index
            f.seek(start)

            if numSamples:
                raw = f.read(numSamples)
            else:
                raw = f.read(self.file_len - start)
            data = np.frombuffer(raw, dtype=np.int16)
            data = RAW2ORDERED(data, channels,num_channels_raw=self.channel_num )
            return data
            
    def get_channel_num(self,data_path):
        list_dir=os.listdir(data_path)
        for file in list_dir:
            if "Record Node" in file:
                try:
                    record_file=os.path.join(data_path,file,"experiment1","recording1","structure.oebin")
                    with open(record_file, "r") as f:
                        data = json.load(f)
                        num_channels = data["continuous"][0]["num_channels"]
                        self.channel_num=num_channels
                        self.bit_uvolts=data["continuous"][0]["channels"][0]["bit_volts"] # conversion to microvolts...
                        for channel in data["continuous"][0]["channels"]:
                            if "ADC" in channel["channel_name"]:
                                self.ttl_bit_volts=channel["bit_volts"]
                                break

                except Exception as e:
                    print(f"Error reading structure.oebin: {e}")
                    self.channel_num=32  # default value
    



    def load(self, data_path, shank, downsample, normalize,invert, channels=None):
        """
        Load all, optionally downsample and normalize it.

        Parameters:
        - data_path (str): Path to the data directory.
        - shank (int): Shank of the electrode.
        - downsample (float): Downsample factor.
        - normalize (bool): Whether to normalize the data.
    
        Returns:
        - data (numpy.ndarray): Loaded and processed data.
        """

        # try:
        #     info = loadmat(f'{data_path}/info.mat')
        # except:
        #     try:
        #         info = loadmat(f'{data_path}/neurospark.mat')
        #     except:
        #         print('.mat file cannot be opened or is not in path.')
        #         return
            
        # try:
        #     channels = info['neurosparkmat']['channels'][0][0][8 * (shank -1):8 * shank]
        # except Exception as err:
        #     print(f'No data available for shank {shank}\n\n{err}')
        #     return 
        # channels=[24,20,23,27,25,21,22,26,28,16,19,31,29,17,18,30,15,3,0,12,14,2,1,13,11,7,4,8,10,6,5,9]
        if channels is None:
            channels=range(self.channel_num)
            self.channels=channels
        # channels=channels[shank*8-8:shank*8]
      
        raw_data = self.load_dat(data_path, channels, numSamples=self.numSamples)

        if hasattr(raw_data, 'shape'):
            self.data = self.clean(raw_data, downsample, normalize)
            self.duration = self.data.shape[0]/self.fs

        self.load_annotations(self.annotation_path,self.name)
        self.load_ripple_times(data_path,invert)
        self.load_offline()

    


    def clean(self, data, downsample, normalize):
        """
        Clean the loaded data by downsampling and normalizing it.

        Parameters:
        - data (numpy.ndarray): Raw data to be cleaned.
        - downsample (bool): Whether to downsample the data.
        - normalize (bool): Whether to normalize the data.

        Returns:
        - data (numpy.ndarray): Cleaned data after downsampling and normalization.
        """

        if downsample:
            self.fs = self.downsampled_fs
            # Downsample data
            if self.verbose:
                print("Downsampling data from %d Hz to %d Hz..."%(self.original_fs, self.downsampled_fs), end=" ")
            data = downsample_data(data, self.original_fs, self.downsampled_fs)
            if self.verbose:
                print("Done!")
        else:
            self.fs = self.original_fs


        if normalize:
            # Normalize it with z-score
            if self.verbose:
                print("Normalizing data...", end=" ")
            data = z_score_normalization(data)
            if self.verbose:
                print("Done!")
                print("Shape of loaded data after downsampling and z-score: ", np.shape(data))

        if self.scale_data:
            data = data[:,:32] * self.bit_uvolts # Convert to microvolts
            if self.channel_num==40 and 32 in self.channels:
                data= data[:,32:]* self.ttl_bit_volts  # Convert TTL channel to microvolts
        return data

    def load_ripple_times(self,path,invert=False):
        
        list_dir=os.listdir(path)
        file=list_dir[1]
        record_file=os.path.join(path,file,"experiment1","recording1", "events")
        for i in os.listdir(record_file):
            if "Network_Events" in i:
                event_folder=os.path.join(record_file,i,"TTL")

        sample_numbers = os.path.join(event_folder, "sample_numbers.npy")
        timestamps_ttl = os.path.join(event_folder, "timestamps.npy")

        sample_numbers = np.load(sample_numbers)   # might not be used here
        # if self.timestamps[0]>0:
        #     self.timestamps -= self.timestamps[0]
        # self.duration=self.timestamps[-1]
        
        timestamps_ttl = np.load(timestamps_ttl) - self.timestamps[0]  # Adjust timestamps to start from zero
        if invert:
            detections=timestamps_ttl[1:]
        else:
            detections=timestamps_ttl[0:]
        # Ensure even length (pairs of start/end)
        if len(detections) % 2 != 0:
            detections=np.append(detections, detections[-1]+0.020)  # or handle as appropriate
            # raise ValueError("Timestamps array must contain an even number of elements (start/end pairs).")

        ripples = detections.reshape(-1, 2)  # shape [X, 2]
        self.from_data.snn_predicts=self.ripples_in_chunk(ripples,self.start,self.numSamples,self.fs,self.fs_conv_fact)
        self.get_ttls(invert)
        self.from_data.seconds_to_samples(self.fs, self.start, self.fs_conv_fact)

    
    def get_ttls(self, invert=False):
        if self.normalize:
            min_max=0
        else:
            min_max=0
        
        if self.timestamps[0]>0:
            self.timestamps -= self.timestamps[0]
        self.duration=self.timestamps[-1]

        if self.channel_num == 40 and 32 in self.channels:
            ttl_channel = np.where(np.array(self.channels) == 32)[0][0]
            if hasattr(self, 'data'):
                ttl_signal = self.data[:, ttl_channel]
            else:
                print("⚠️ Data not loaded, cannot extract TTL signal.")
                return

            if ttl_signal.max() <min_max or self.annotated.light_stim is None:
                print("⚠️ TTL signal not detected or too low amplitude.")
                return

            else:

                # --- Step 1: Set a threshold automatically (midpoint between min and max)
                threshold = (ttl_signal.max() + ttl_signal.min()) / 2

                # --- Step 2: Detect rising edges (transitions from low -> high)
                ttl_binary = ttl_signal > threshold # detects when high
                edges = np.diff(ttl_binary.astype(int)) # detects transitions - low to high

                # Rising edges (0 -> 1)
                rising_edges = np.where(edges == 1)[0]
                # print(len(rising_edges))

                # Falling edges (1 -> 0)
                falling_edges = np.where(edges == -1)[0]
                # print(len(falling_edges))

                # Optionally invert TTL logic # when the other is not inverted
                # if not invert:
                #     rising_edges, falling_edges = falling_edges, rising_edges
                if len(falling_edges)!=len(rising_edges):
                    if len(rising_edges)>len(falling_edges):
                        print(" ⚠️ More falling edges than rising edges, adjusting...")
                        falling_edges = np.append(falling_edges, rising_edges[-1] + int(0.001 * self.fs))
                        
                    else:    
                        print(" ⚠️ More rising edges than falling edges, adjusting...")
                        min_len=min(len(falling_edges),len(rising_edges))
                        # print(min_len)
                        rising_edges=rising_edges[0:min_len]
                        falling_edges=falling_edges[1:min_len+1]
                
                ttls= np.column_stack((rising_edges, falling_edges))
                ttl_times=ttls/self.fs
                self.from_data.light_stim=self.ripples_in_chunk(ttl_times,self.start,self.numSamples,self.fs,self.fs_conv_fact)
    
    def _check_data(self):

        if self.annotated.ripples_GT is None:
            print("⚠️ No ground truth annotations loaded.")
        else:
            print(f"Loaded {len(self.annotated.ripples_GT)} ground truth ripple events.")
        if self.annotated.snn_predicts is None:
            print("⚠️ No ground truth SNN predictions loaded.")
        elif self.from_data.snn_predicts.shape != self.annotated.snn_predicts.shape:
            print(f"⚠️ Warning: Number of SNN predictions from data ({self.from_data.snn_predicts.shape[0]}) does not match number of annotations ({self.annotated.snn_predicts.shape[0]}).")
        else:
            print(f"Loaded {len(self.from_data.snn_predicts)} SNN predicted ripple events.")

        if self.from_data.light_stim is None and self.annotated.light_stim is None:
            print("⚠️ No light stimulation events loaded.")
        else:
            if self.annotated.light_stim is None:
                print("⚠️ No ground truth light stimulation annotations loaded.")
                print(f"Loaded {len(self.from_data.light_stim)} light stimulation events.")
            elif self.from_data.light_stim is None:
                print("⚠️ No light stimulation events loaded from data.")
            elif self.from_data.light_stim.shape != self.annotated.light_stim.shape:
                print(f"⚠️ Warning: Number of raw light stimulation events ({self.from_data.light_stim.shape[0]}) does not match number of annotated light stimulation events ({self.annotated.light_stim.shape[0]}).")
            else:
                print(f"Loaded {len(self.from_data.light_stim)} light stimulation events.")


    def load_annotations(self,path,name):
        list_dir=os.listdir(path)
        for file in list_dir:
            sessions=os.listdir(os.path.join(path,file))
            # print(sessions)
            if any(name in s for s in sessions):
                folder_path=os.path.join(path,file,f"events_{name}","events")
                print(folder_path)
                self.annotated.ripples_GT=self.ripples_in_chunk(np.loadtxt(os.path.join(folder_path,"events_selected_manually.txt")),self.start,self.numSamples,self.fs,self.fs_conv_fact)
                # print(self.annotated.ripples_GT)
                self.annotated.snn_predicts=self.ripples_in_chunk(np.loadtxt(os.path.join(folder_path,"events_SNN.txt")),self.start,self.numSamples,self.fs,self.fs_conv_fact)
                # print(self.annotated.snn_predicts)
                try:
                    self.annotated.light_stim=self.ripples_in_chunk(np.loadtxt(os.path.join(folder_path,"events_light.txt")),self.start,self.numSamples,self.fs,self.fs_conv_fact)
                    print(f"Loaded {len(self.annotated.light_stim)} annotated light stimulation events.")
                except Exception as e:
                    print(f"Light stim file not found: {e}")
                break
            else:
                print(f"Annotation for session {name} not found in {os.path.join(path,file)}.")
        self.annotated.seconds_to_samples(self.fs, self.start, self.fs_conv_fact)
    
    def load_offline(self,):
        filename=f"spike_data_offline_{self.name}.pkl"
        try:
            with open(os.path.join(self.offline_path,filename), 'rb') as f:
                offline_data = pkl.load(f)
            self.list_TTLs = offline_data.get("ttls", [[], [], []])  # [Detection, Input, Seizure]
            self.offline_detections = np.round(np.array(self.list_TTLs[0]) / self.fs_conv_fact).astype(int)            
        except Exception as e:
            print(f"Error loading offline data: {e}")
            self.offline_detections = np.array([])



    @plain_plot
    @hide_y_ticks_on_offset
    def plot_event(self, 
                   event, 
                   offset=0, 
                   extend=0, 
                   delimiter=False, 
                   show=True, 
                   filtered=[], 
                   title='', 
                   label='', 
                   ch=False,
                   ylim=False,
                   line_color=False,
                   show_ground_truth=False, 
                   show_predictions=False, 
                   plain=False):
        """
        Plot the ripple signal number idx.

        Parameters:
        - idx (int): Index of the ripple to plot.
        - offset (float): Offset between channels for visualization.
        - extend (float): Extend the plotted time range before and after the ripple.
        - delimiter (bool): Whether to highlight the ripple area.

        Returns:
        - fig (matplotlib.figure.Figure): The generated figure.
        - ax (matplotlib.axes.Axes): The axes object containing the plot.
        """
            
        prop = self.fs_conv_fact
        interval = deepcopy(event)
        handles = []
        labels = []

        try:
            if extend != 0:
                if (interval[0] - extend) < 0:
                    interval[0] = int(self.start / prop)
                else:
                    interval[0] = interval[0] - extend

                if (interval[1] + extend) > self.numSamples/prop:
                    interval[1] = int((self.start + self.numSamples)/prop)
                else:
                    interval[1] = interval[1] + extend

        except IndexError:
            print('IndexError')
            print(f'There no data available for the selected samples.\nLength of loaded data: {int(self.numSamples/self.fs_conv_fact)}')
            return None, None

        # Define window data
        self.window_interval = interval
        mask = (self.ripples_GT[:, 1] >= interval[0]) & (self.ripples_GT[:, 0] <= interval[1])
        self.window_ripples = self.ripples_GT[mask]

        interval_data = self.data[interval[0]: interval[1]][:]
        self.window = deepcopy(interval_data)
        
        time_vector = np.linspace(interval[0] / self.fs, interval[1] / self.fs, interval_data.shape[0])
        if show:
            fig, ax = plt.subplots(figsize=(10, 6))
        for i, chann in enumerate(interval_data.transpose()):
            if filtered:
                bandpass = filtered
                chann = bandpass_filter(chann, bandpass, self.fs)
                self.window[:, i] = chann
            if show:
                if ch:
                    if i in ch:
                        if line_color:
                            ax.plot(time_vector, chann + i * offset, line_color)
                        else:
                            ax.plot(time_vector, chann + i * offset)
                else:
                    if line_color:
                        ax.plot(time_vector, chann + i * offset, line_color)
                    else:
                        ax.plot(time_vector, chann + i * offset)
            
            if ylim:
                ax.set_ylim(ylim)
                

        max_val = np.max(self.window.reshape((self.window.shape[0]*self.window.shape[1]))) + offset*8
        min_val = np.min(self.window.reshape((self.window.shape[0]*self.window.shape[1])))

        if delimiter and show:
            if extend > 0:
                ripple_area = [time_vector[round(extend)], time_vector[-round(extend)]]
                if not label:
                    label='Event area'
                fill_DEL = ax.fill_between(ripple_area, min_val, max_val, color="tab:blue", alpha=0.2)
                handles.append(fill_DEL)
                labels.append(label)
            else:
                if self.verbose:
                    print('Delimiter not applied because there is no extend.')

        if show_ground_truth:
            if hasattr(self.ripples_GT, 'dtype'):
                for ripple in self.window_ripples:
                    fill_GT = ax.fill_between([ripple[0] / self.fs, ripple[1] / self.fs],  min_val, max_val, color="tab:red", alpha=0.3)

            if 'fill_GT' in locals():
                handles.append(fill_GT)
                labels.append('Ground truth' if not label else label)

        if show_predictions:
            if hasattr(self, 'prediction_idxs'):
                mask = (self.prediction_idxs[:, 1] >= interval[0]) & (self.prediction_idxs[:, 0] <= interval[1])
                self.prediction_times_from_window = self.prediction_times[mask]
                for times in self.prediction_times_from_window:
                    fill_PRED = ax.fill_between([times[0], times[1]], min_val, max_val, color="tab:blue", alpha=0.3)

            if 'fill_PRED' in locals():
                handles.append(fill_PRED)
                labels.append(f'{self.model_type} predict')

        # Figure styles
        if filtered and not title:
            title = f'Filtered channels\nEvent {interval}\nBandpass: {bandpass[0]}-{bandpass[1]}'
        if not title:
            title = f'Channels for samples {interval}'

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        if not len(handles) == 0:        
            ax.legend(handles, labels)

        text = ax.set_title(title, loc='center', fontfamily='serif', fontsize=12, fontweight='bold')
        ax.grid(True)
        self.fig = fig
        self.ax = ax

        if show:
            return fig, ax

    def plot_all(self, ch=None, offset=0, filtered=None, extend=0.5, title='Overview', window=None):
        """
        Plot data with overlays for:
        - Ground truth ripples (yellow)
        - Model predicted ripples (blue)
        - Light stimulation TTLs (red)

        Parameters
        ----------
        ch : int or list of ints, optional
            Channel(s) to plot. If None, plots the first channel.
        offset : float, optional
            Vertical offset between channels.
        filtered : tuple (low, high), optional
            Bandpass filter range (Hz).
        extend : float, optional
            Extra context (in seconds) around each event.
        title : str, optional
            Title for the plot.
        window : tuple (start, end), optional
            Time window in seconds to plot (e.g., (10, 20)).
        """

        # ---------------------
        # Select channels
        # ---------------------
        if ch is None:
            ch = [0]
        elif isinstance(ch, int):
            ch = [ch]

        n_samples = self.data.shape[0]
        time = np.arange(n_samples) / self.fs

        # ---------------------
        # Apply time window
        # ---------------------
        if window is not None:
            start_s, end_s = window
            start_idx = int(start_s * self.fs)
            end_idx = int(end_s * self.fs)
            time = time[start_idx:end_idx]
            data_slice = self.data[start_idx:end_idx, :]
        else:
            data_slice = self.data

        fig, ax = plt.subplots(figsize=(15, 6))
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (normalized units)")

        # ---------------------
        # Plot data (optionally filtered)
        # ---------------------
        for i, ch_idx in enumerate(ch):
            sig = deepcopy(data_slice[:, ch_idx])
            if filtered:
                from signal_aid import bandpass_filter
                sig = bandpass_filter(sig, filtered, self.fs)

            ax.plot(time, sig + i * offset, color="black", lw=0.8, label=f"Ch {ch_idx}" if i == 0 else "")

        min_y, max_y = ax.get_ylim()

        # ---------------------
        # Helper function to plot intervals safely within window
        # ---------------------
        def plot_intervals(intervals, color, label):
            if window is not None:
                mask = (intervals[:, 1] / self.fs > start_s) & (intervals[:, 0] / self.fs < end_s)
                intervals = intervals[mask]
            for i, r in enumerate(intervals):
                ax.fill_between(r / self.fs, min_y, max_y, color=color, alpha=0.3, label=label if i == 0 else "")

        # ---------------------
        # Ground truth ripples (yellow)
        # ---------------------
        if hasattr(self.annotated, "ripples_GT") and self.annotated.ripples_GT is not None:
            plot_intervals(self.annotated.ripples_GT, "yellow", "Ground truth")

        # ---------------------
        # Predicted ripples (blue)
        # ---------------------
        if hasattr(self.from_data, "snn_predicts") and self.from_data.snn_predicts is not None:
            # plot_intervals(self.from_data.snn_predicts, "tab:blue", "Predicted")
            starts = self.from_data.snn_predicts[:, 0] / self.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="tab:blue", alpha=0.5, lw=0.5,label="Predicted")

        if hasattr(self.annotated, "snn_predicts") and self.annotated.snn_predicts is not None:
            starts = self.annotated.snn_predicts[:, 0] / self.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="green", alpha=0.5, lw=0.5,label="Annotated Predicted")

        # ---------------------
        # Light stimulation (red)
        # ---------------------
        if hasattr(self.from_data, "light_stim") and self.from_data.light_stim is not None:
            plot_intervals(self.from_data.light_stim, "red", "Light stim")
            starts = self.from_data.light_stim[:, 0] / self.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="red", alpha=0.5, lw=0.5)

        if hasattr(self.annotated, "light_stim") and self.annotated.light_stim is not None:
            plot_intervals(self.annotated.light_stim, "orange", "Annotated light stim")
            starts = self.annotated.light_stim[:, 0] / self.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="orange", alpha=0.5, lw=0.5)

        # ---------------------
        # Offline detections (purple dashed)
        # ---------------------
        if hasattr(self, "offline_detections") and self.offline_detections.size > 0:
            offline_starts = self.offline_detections / self.fs
            if window is not None:
                offline_starts = offline_starts[(offline_starts >= start_s) & (offline_starts <= end_s)]
            ax.vlines(offline_starts, min_y, max_y, color="purple", alpha=0.5, lw=0.5, ls='--', label="Offline detections")
        # ---------------------
        # Beautify
        # ---------------------
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_offline(self, ch=None, offset=0, filtered=None, extend=0.5, title='Overview', window=None):
        """
        Plot data with overlays for:
        - Ground truth ripples (yellow)
        - Model predicted ripples (blue)
        - Light stimulation TTLs (red)

        Parameters
        ----------
        ch : int or list of ints, optional
            Channel(s) to plot. If None, plots the first channel.
        offset : float, optional
            Vertical offset between channels.
        filtered : tuple (low, high), optional
            Bandpass filter range (Hz).
        extend : float, optional
            Extra context (in seconds) around each event.
        title : str, optional
            Title for the plot.
        window : tuple (start, end), optional
            Time window in seconds to plot (e.g., (10, 20)).
        """

        # ---------------------
        # Select channels
        # ---------------------
        if ch is None:
            ch = [0]
        elif isinstance(ch, int):
            ch = [ch]

        n_samples = self.data.shape[0]
        time = np.arange(n_samples) / self.fs

        # ---------------------
        # Apply time window
        # ---------------------
        if window is not None:
            start_s, end_s = window
            start_idx = int(start_s * self.fs)
            end_idx = int(end_s * self.fs)
            time = time[start_idx:end_idx]
            data_slice = self.data[start_idx:end_idx, :]
        else:
            data_slice = self.data

        fig, ax = plt.subplots(figsize=(15, 6))
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (normalized units)")

        # ---------------------
        # Plot data (optionally filtered)
        # ---------------------
        for i, ch_idx in enumerate(ch):
            sig = deepcopy(data_slice[:, ch_idx])
            if filtered:
                from signal_aid import bandpass_filter
                sig = bandpass_filter(sig, filtered, self.fs)

            ax.plot(time, sig + i * offset, color="black", lw=0.8, label=f"Ch {ch_idx}" if i == 0 else "")

        min_y, max_y = ax.get_ylim()

        # ---------------------
        # Helper function to plot intervals safely within window
        # ---------------------
        def plot_intervals(intervals, color, label):
            if window is not None:
                mask = (intervals[:, 1] / self.fs > start_s) & (intervals[:, 0] / self.fs < end_s)
                intervals = intervals[mask]
            for i, r in enumerate(intervals):
                ax.fill_between(r / self.fs, min_y, max_y, color=color, alpha=0.3, label=label if i == 0 else "")

        # ---------------------
        # Ground truth ripples (yellow)
        # ---------------------
        if hasattr(self.annotated, "ripples_GT") and self.annotated.ripples_GT is not None:
            plot_intervals(self.annotated.ripples_GT, "yellow", "Ground truth")

        # ---------------------
        # Predicted ripples (blue)
        # ---------------------
        # if hasattr(self.from_data, "snn_predicts") and self.from_data.snn_predicts is not None:
        #     # plot_intervals(self.from_data.snn_predicts, "tab:blue", "Predicted")
        #     starts = self.from_data.snn_predicts[:, 0] / self.fs
        #     if window is not None:
        #         starts = starts[(starts >= start_s) & (starts <= end_s)]
        #     ax.vlines(starts, min_y, max_y, color="tab:blue", alpha=0.5, lw=0.5,label="Predicted")

        # if hasattr(self.annotated, "snn_predicts") and self.annotated.snn_predicts is not None:
        #     starts = self.annotated.snn_predicts[:, 0] / self.fs
        #     if window is not None:
        #         starts = starts[(starts >= start_s) & (starts <= end_s)]
        #     ax.vlines(starts, min_y, max_y, color="green", alpha=0.5, lw=0.5,label="Annotated Predicted")

        # ---------------------
        # Light stimulation (red)
        # ---------------------
        if hasattr(self.from_data, "light_stim") and self.from_data.light_stim is not None:
            plot_intervals(self.from_data.light_stim, "red", "Light stim")
            starts = self.from_data.light_stim[:, 0] / self.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="red", alpha=0.5, lw=0.5)

        if hasattr(self.annotated, "light_stim") and self.annotated.light_stim is not None:
            plot_intervals(self.annotated.light_stim, "orange", "Annotated light stim")
            starts = self.annotated.light_stim[:, 0] / self.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="orange", alpha=0.5, lw=0.5)

        # ---------------------
        # Offline detections (purple dashed)
        # ---------------------
        if hasattr(self, "offline_detections") and self.offline_detections.size > 0:
            offline_starts = self.offline_detections / self.fs
            if window is not None:
                offline_starts = offline_starts[(offline_starts >= start_s) & (offline_starts <= end_s)]
            ax.vlines(offline_starts, min_y, max_y, color="purple", alpha=0.5, lw=0.5, ls='--', label="Offline detections")
        # ---------------------
        # Beautify
        # ---------------------
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class RippleEvents():
    def __init__(self,offset):
        self.ripples_GT = None
        self.snn_predicts = None
        self.light_stim = None
        self.offset=offset

    def seconds_to_samples(self, fs, start, fs_conv_fact):
        """
        Convert event times from seconds to sample indices.

        Parameters
        ----------
        fs : float
            Sampling frequency (Hz).
        start : int
            Start sample index of the chunk.
        fs_conv_fact : float
            Conversion factor if data were downsampled.
        """
        def convert(arr):
            return (arr * fs - start / fs_conv_fact).astype(int)

        if isinstance(self.ripples_GT, np.ndarray) and self.ripples_GT.size > 0:
            self.ripples_GT = convert(self.ripples_GT)
        if isinstance(self.snn_predicts, np.ndarray) and self.snn_predicts.size > 0:
            self.snn_predicts = convert(self.snn_predicts-self.offset)
        if isinstance(self.light_stim, np.ndarray) and self.light_stim.size > 0:
            self.light_stim = convert(self.light_stim)


def plot_difference(liset):
    """
    Compare timing differences between light stim starts and SNN predicted starts
    for both 'from_data' and 'annotated' sources.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    def compute_and_plot_diff(source, label, color,expected_offset=0.2, max_delay=0.5):
        snn_predicts = getattr(source, "snn_predicts", None)
        light_stim = getattr(source, "light_stim", None)

        if snn_predicts is None or light_stim is None:
            print(f"⚠️ Missing data in {label}: ensure both 'snn_predicts' and 'light_stim' exist.")
            return None

        snn_starts = snn_predicts[:, 0] / liset.fs
        light_starts = light_stim[:, 0] / liset.fs

        matched_diffs = []
        used_snn_idx = set()

        for t_light in light_starts:
            # expected snn time (light + expected_offset)
            expected_time = t_light + expected_offset
            
            # find nearest snn time to that expected value
            idx = np.argmin(np.abs(snn_starts - expected_time))
            nearest_snn = snn_starts[idx]
            diff = nearest_snn - t_light

            # only accept if within window and not already used
            if abs(diff - expected_offset) < max_delay and idx not in used_snn_idx:
                matched_diffs.append(diff)
                used_snn_idx.add(idx)
            else:
                # skip if no reasonable match found
                continue

        if len(matched_diffs) == 0:
            print(f"⚠️ {label}: No valid matches found within {max_delay}s window.")
            return None

        diffs = np.array(matched_diffs)
        print(f"✅ {label}: matched {len(diffs)} events "
            f"(light={len(light_starts)}, snn={len(snn_starts)})")

        # Plot histogram of timing differences
        ax.hist(diffs, bins=20, alpha=0.6, label=f"{label} (n={len(diffs)})", color=color, edgecolor="black")

        # Print summary stats
        print(f"📊 {label} timing differences:")
        print(f"  Mean Δt = {np.mean(diffs):.4f} s")
        print(f"  Std Δt  = {np.std(diffs):.4f} s")
        print(f"  Median  = {np.median(diffs):.4f} s\n")

        return diffs

    # Compute and plot both sets
    diffs_data = compute_and_plot_diff(liset.from_data, "From data", "tab:blue")
    diffs_annot = compute_and_plot_diff(liset.annotated, "Annotated", "tab:orange")

    # Format figure
    ax.axvline(np.mean(diffs_data), color='gray', lw=1, ls='--',label='Mean From Data') if diffs_data is not None else None
    ax.axvline(np.mean(diffs_annot), color='black', lw=1, ls='--',label='Mean Annotated') if diffs_annot is not None else None
    ax.set_xlabel("Δt = Predicted start − Light start [s]")
    ax.set_ylabel("Count")
    ax.set_title("Timing Differences: SNN Prediction vs Light Stimulation")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return {"from_data": diffs_data, "annotated": diffs_annot}



# data_path=r"D:\Madrid_tests"
# name="2025-09-24_14-37-17"

# invert=True

# liset=read_data(data_path, name, downsample=1000, normalize=True, invert=invert,offset=0)
# # print(liset.offline_detections[:10]/liset.fs)
# liset.plot_all(ch=[20,],offset=0.3,filtered=(150,250))
# # dic=plot_difference(liset)

# liset.plot_offline(ch=[20,],offset=0.3,)