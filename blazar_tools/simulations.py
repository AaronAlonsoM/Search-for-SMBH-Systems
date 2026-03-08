import numpy as np
import pandas as pd
import colorednoise as cn
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from .ssanalysis_pyts import SSA_pyts
from .periodogram import Periodogram
from .utils import time_splits, mjd_to_year, year_to_mjd


def random_holes(time, signal):
    
    ratio = np.random.uniform(0, 0.44) # Random ratio of empty bins
    mask = np.random.rand(len(signal)) > ratio

    # Now, the mask is applied and we are only left with the appropiate bins
    time_filt = time[mask] 
    signal_filt = signal[mask]
    
    return time_filt, signal_filt

class LC_sim:

    def  __init__ (self, time, day_bins = 28):
        """
        Generates a simulated blazar LC with noise or a noisy signal.
        It simulates flux received in bins, similarly to Fermi-Lat LC Repository.

        Inputs
        ------
        time: in years
        day_bins: the size on the bins. By dafault, 28-day bins

        Returns: time, signal
        """

        days = time * 365
        self.n_bins = int(days / day_bins) # 28 day bins
        self.t = np.linspace(0, time, self.n_bins)

        
    def noise(self):
        
        A = np.random.uniform(1, 20)
        noise_comp = [cn.powerlaw_psd_gaussian(i, self.n_bins) for i in (0, 1, 2)]
        noise_component = A * (  noise_comp[0] +  noise_comp[1] + noise_comp[2])
        
        min_signal = abs(np.min(noise_component) * 1.1)
        O = np.random.uniform( max(min_signal, 0) ,  150)
        signal = noise_component + O

        time, flux = random_holes(self.t, signal)

        return time, flux

    def curve(self):
        
        A = np.random.uniform(1, 20)
        T = np.random.uniform(1.0, 4.5)
        theta = np.random.uniform(0, 2 * np.pi)

        snr = np.random.uniform(0.8, 2.5) # Random SNR
        noise_comp = [cn.powerlaw_psd_gaussian(i, self.n_bins) for i in (0, 1, 2)]
        noise_component = (noise_comp[0] + noise_comp[1] + noise_comp[2])
        
        self.real_oscillatory = A * (  np.sin((2 * np.pi *  self.t / T) + theta))

        periodic = A * ( snr *  np.sin((2 * np.pi *  self.t / T) + theta) + noise_component)

        min_signal = abs(np.min(periodic) * 1.1)
        O = np.random.uniform( max(min_signal, 0) ,  150)
        signal = periodic + O

        time, flux = random_holes(self.t, signal)

        return time, flux


def features_extraction(time, signal, freq_bound = 0.03, c_bound = 0.9):
    '''
    Returns a dictionary with the selected features used to train the model.
    '''
    d = {}
    
    # First, we get features from the full signal
    # SSA decomposition is performed
    n =len(signal)
    L = int(0.2 * n)
    df_ssa = SSA_pyts(year_to_mjd(time), signal, L = L, freq_bound = freq_bound, c_bound = c_bound)
    trend = df_ssa.trend.values
    oscillatory = df_ssa.oscillatory.values
    noise = df_ssa.noise.values

    d['r_std'] = np.std(signal) / np.std(oscillatory) 
    d['SNR'] = np.std(signal) / np.std(noise) 

    
    per = Periodogram(df_ssa)
    pgram_dict = per.LSP(plot = False)
    pgram = pgram_dict['pgram'].values
    
    period_full = pgram_dict['period']
    fwhm_full = pgram_dict['fwhm']

    

    d['period'] = period_full
    d['fwhm'] = fwhm_full

    # Splits analysis
    df_full  = pd.DataFrame({'time': time, 'signal': signal})
    split_list = time_splits(df_full, splits = 2) 

    pgram_dict_split_list = []
    for i in (0, 1):
        df_split = split_list[i]
        time_split = df_split.time.values
        signal_split = df_split.signal.values
        n = len(time_split)
        L = int(0.2*n)
        df_ssa_split = SSA_pyts(year_to_mjd(time_split), signal_split, L = L, freq_bound = freq_bound, c_bound = c_bound)
        osc_values = df_ssa_split.oscillatory.values

        # In the case the analysis results in no oscillatory component
        if np.all(osc_values == 0) or np.dot(osc_values, osc_values) == 0:
            pgram_dict_split_list.append({'period': 0, 'fwhm': 0, 'pgram': pd.Series([np.nan])})
        else:
            per_split = Periodogram(df_ssa_split)
            pgram_dict_split = per_split.LSP(plot=False)
            pgram_dict_split_list.append(pgram_dict_split)


    if len(pgram_dict_split_list)==0:
        d['r_period_splits']=np.nan
    else:
        d['r_period_splits'] = abs(pgram_dict_split_list[0]['period'] - pgram_dict_split_list[1]['period']) / period_full
        
        

    peak_power_full = pgram.max() 
    
    d['r_period_fwhm'] = period_full /  fwhm_full
    d['r_power_std'] = peak_power_full / np.std(pgram)

    
    peaks_idx, prop = find_peaks(pgram, distance=10, prominence=0) 
    
    if len(peaks_idx) > 0:
        # Peaks sorting
        sorted_prominences = np.sort(prop['prominences'])[::-1]
        
        d['main_peak_prominence'] = sorted_prominences[0] / np.std(pgram)
        
        if len(sorted_prominences) >= 2:
            d['prominence_ratio'] = sorted_prominences[0] / sorted_prominences[1]
        else:
            d['prominence_ratio'] = sorted_prominences[0] # Only if there is no second peak
    else:
        d['main_peak_prominence'] = 0
        d['prominence_ratio'] = 0
    
    
    autocorr = acf(oscillatory, nlags=len(oscillatory)//2)
    d['autocorrelation_osc'] = np.max(autocorr[10:]) 
    
    autocorr = acf(signal, nlags=len(signal)//2)
    d['autocorrelation_signal'] = np.max(autocorr[10:])

    return d