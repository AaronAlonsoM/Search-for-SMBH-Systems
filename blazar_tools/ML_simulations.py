import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from ftperiodogram.modeler import FastTemplatePeriodogram
from ftperiodogram.template import Template
from scipy.interpolate import interp1d
from scipy.stats import skew
from scipy.sparse.linalg import svds
from astropy.timeseries import LombScargle
import colorednoise as cn
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

# ==========================================
# 1. CORE FUNCTIONS (Preserved from original)
# ==========================================
class SSAFast(object):
    def __init__(self, tseries, L, t, n_components=11):
        self.ts_values = np.asarray(tseries, dtype=float)
        self.t_index = t
        self.N = len(self.ts_values)
        self.L = int(L)
        self.K = self.N - self.L + 1
        
        shape = (self.L, self.K)
        strides = (self.ts_values.strides[0], self.ts_values.strides[0])
        self.X = np.lib.stride_tricks.as_strided(self.ts_values, shape=shape, strides=strides).copy()
        k_svd = min(n_components, self.L - 1, self.K - 1)
        U, Sigma, VT = svds(self.X, k=k_svd)
        
        self.U = U[:, ::-1]
        self.Sigma = Sigma[::-1]
        self.VT = VT[::-1, :]
        self.d = len(self.Sigma)
        self.eigvals = (self.Sigma ** 2) / float(self.K)
        
    def reconstruct(self, indices):
        valid_indices = [i for i in indices if i < self.d]
        if not valid_indices: return pd.Series(np.zeros(self.N), index=self.t_index)
        U_sub = self.U[:, valid_indices]
        Sigma_sub = np.diag(self.Sigma[valid_indices])
        VT_sub = self.VT[valid_indices, :]
        X_elem_sum = np.dot(U_sub, np.dot(Sigma_sub, VT_sub))
        X_rev = X_elem_sum[::-1]
        reconstructed_vals = np.array([X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])])
        return pd.Series(reconstructed_vals, index=self.t_index)

def _count_zero_crossings(x, eps=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2: return 0
    x = x - np.nanmedian(x)
    if eps is None: eps = 1e-12 * (np.nanstd(x) + 1.0)
    x[np.abs(x) <= eps] = 0.0
    nz = x != 0.0
    x = x[nz]
    if x.size < 2: return 0
    s = np.sign(x)
    return int(np.sum(s[:-1] * s[1:] < 0))

def find_trend(ssa, time_mjd, period_max_yr=6):
    total_years = (time_mjd[-1] - time_mjd[0]) / 365.2425
    baseline_rcs = ssa.reconstruct([0])
    for k in range(1, min(11, len(ssa.eigvals))):
        rc_series = ssa.reconstruct([k])
        n_cross = _count_zero_crossings(rc_series.values)
        timescale = np.inf if n_cross == 0 else (total_years * 2.0) / n_cross
        if timescale > period_max_yr:
            baseline_rcs += rc_series
        else:
            break
    return baseline_rcs

def gauss_template(sigma_template, x):
    return (1 / (sigma_template * np.sqrt(2 * np.pi)))*np.exp(-(x-0.5)**2 / (2*sigma_template**2) )

# ==========================================
# 2. FEATURE EXTRACTION LOGIC
# ==========================================
def extract_10_features(time_mjd, flux, nan_mask, ftp_models, freqs):
    ''' Extract the 10 hardened features from a light curve '''
    
    features = {}
    periods = 1 / (freqs * 365.25) # in years
    total_obs_years = (time_mjd[-1] - time_mjd[0]) / 365.25
    
    # MINIMAL CHANGE: Mask to force feature extraction only for periods <= 5.0 years
    valid_mask = periods <= 5.0
    
    # 1. Fill holes for SSA
    idx_nans = np.flatnonzero(nan_mask)
    idx_valid = np.flatnonzero(~nan_mask)
    flux_filled = flux.copy()
    flux_filled[nan_mask] = np.interp(idx_nans, idx_valid, flux[~nan_mask])
    
    # Calculate Autocorr Raw (Feature 3)
    features['autocorrelation_raw'] = pd.Series(flux_filled).autocorr(lag=1)
    
    # 2. SSA Detrending
    L = int(0.4 * len(flux_filled))
    ssa = SSAFast(flux_filled, L, time_mjd)
    trend = find_trend(ssa, time_mjd, period_max_yr=6)
    
    osc_array_filled = flux_filled - trend.values
    
    # 3. Calculate SSA Features (Group A)
    noise_rc = ssa.reconstruct(range(ssa.d - 5, ssa.d)).values # Estimate noise from tail components
    features['SNR_osc'] = np.std(osc_array_filled) / (np.std(noise_rc) + 1e-9)
    features['autocorrelation_osc'] = pd.Series(osc_array_filled).autocorr(lag=1)
    
    # Re-apply nan mask for FTP
    time_valid = time_mjd[~nan_mask]
    flux_detrended = osc_array_filled[~nan_mask]
    flux_err = np.ones_like(flux_detrended)
    
    # 4. Standard Lomb-Scargle
    ls = LombScargle(time_valid, flux_detrended)
    power_ls = ls.power(freqs)
    # MINIMAL CHANGE: Apply mask
    max_ls = np.max(power_ls[valid_mask])
    
    # 5. Run all FTP Templates to find the BEST
    best_power = 0.0
    best_pgram = None
    best_ftp = None
    
    for ftp in ftp_models:
        ftp.fit(time_valid, flux_detrended, flux_err)
        pgram = ftp.power(freqs, save_best_model=True)
        # MINIMAL CHANGE: Apply mask
        max_p = np.max(pgram[valid_mask])
        if max_p > best_power:
            best_power = max_p
            best_pgram = pgram
            best_ftp = ftp

    # MINIMAL CHANGE: Make sure we get the index of the max valid peak
    valid_indices = np.where(valid_mask)[0]
    best_idx_in_valid = np.argmax(best_pgram[valid_mask])
    best_idx = valid_indices[best_idx_in_valid]
    
    best_freq = freqs[best_idx]
    best_period_yr = periods[best_idx]
    
    # 6. Frequency Domain Features (Group B)
    features['Power_Ratio_FTP_vs_LSP'] = best_power / (max_ls + 1e-9)
    
    # FWHM
    half_max = best_power / 2.0
    left_idx = best_idx
    while left_idx > 0 and best_pgram[left_idx] > half_max: left_idx -= 1
    right_idx = best_idx
    while right_idx < len(best_pgram)-1 and best_pgram[right_idx] > half_max: right_idx += 1
    features['periodogram_fwhm'] = periods[left_idx] - periods[right_idx] # Width in frequency/period space

    # Prominence (H1/H2) - Look for highest peak separated by at least 10 frequency bins
    mask_h2 = np.ones(len(best_pgram), dtype=bool)
    mask_h2[max(0, best_idx - 10):min(len(best_pgram), best_idx + 10)] = False
    # MINIMAL CHANGE: ensure prominence also only looks at valid mask
    mask_h2 = mask_h2 & valid_mask
    secondary_power = np.max(best_pgram[mask_h2]) if np.any(mask_h2) else 1e-9
    features['FTP_prominence_ratio'] = best_power / (secondary_power + 1e-9)
    
    # 7. Temporal Stability (Group C) - r_period_splits
    mid_idx = len(time_valid) // 2
    if mid_idx > 10:
        t1, f1 = time_valid[:mid_idx], flux_detrended[:mid_idx]
        t2, f2 = time_valid[mid_idx:], flux_detrended[mid_idx:]
        
        best_ftp.fit(t1, f1, np.ones_like(f1))
        # MINIMAL CHANGE: apply valid_mask to splits
        pgram1 = best_ftp.power(freqs, save_best_model=False)
        p1_idx = valid_indices[np.argmax(pgram1[valid_mask])]
        p1 = periods[p1_idx]
        
        best_ftp.fit(t2, f2, np.ones_like(f2))
        pgram2 = best_ftp.power(freqs, save_best_model=False)
        p2_idx = valid_indices[np.argmax(pgram2[valid_mask])]
        p2 = periods[p2_idx]
        
        features['r_period_splits'] = abs(p1 - p2) / best_period_yr
    else:
        features['r_period_splits'] = 0.0

    # 8. Phase-Domain Features (Group D)
    phase_grid = np.linspace(0, 1, 100)
    time_grid = phase_grid / best_freq
    folded_model = best_ftp.best_model(time_grid)
    
    features['Folded_Skewness'] = skew(folded_model)
    
    threshold = np.max(folded_model) * 0.5
    features['Duty_Cycle'] = np.sum(folded_model >= threshold) / len(folded_model)
    
    # Number of cycles
    features['number_of_cycles'] = total_obs_years / best_period_yr

    return features

# ==========================================
# 3. DATASET GENERATOR
# ==========================================
def generate_samples(base_blazar_id, df_blazar, real_alphas, ftp_models, freqs, t_sin, t_emp, t_gauss_list, n_samples_per_class=10):
    ''' Generates N noise (y=0) and N injected QPO (y=1) samples '''
    
    results = []
    time_mjd = df_blazar.t_mjd.values
    nan_mask = df_blazar.flux.isna().values
    n_bins = len(df_blazar)
    
    # Y=0: PURE RED NOISE
    for _ in range(n_samples_per_class):
        alpha = np.random.choice(real_alphas)
        sim_noise = cn.powerlaw_psd_gaussian(alpha, size=n_bins)
        sim_noise[nan_mask] = np.nan
        
        feats = extract_10_features(time_mjd, sim_noise, nan_mask, ftp_models, freqs)
        feats['label'] = 0
        feats['injected_alpha'] = alpha
        feats['base_mask'] = base_blazar_id
        results.append(feats)

    # Y=1: RED NOISE + INJECTED QPO
    x_phase = np.linspace(0, 1, 500)
    
    # Pre-build numpy arrays for the 3 injection shapes
    shape_sin = np.sin(2 * np.pi * x_phase)
    
    # Empirical PG1553 shape
    try:
        shape_emp = np.loadtxt('data/PG1553_template_shape.csv', delimiter=',')
        shape_emp = np.interp(x_phase, np.linspace(0,1,len(shape_emp)), shape_emp)
    except:
        shape_emp = shape_sin # Fallback
        
    for _ in range(n_samples_per_class):
        alpha = np.random.choice(real_alphas)
        sim_noise = cn.powerlaw_psd_gaussian(alpha, size=n_bins)
        
        # QPO Parameters
        true_period_yr = np.random.uniform(1.5, 5.0)
        true_freq = 1 / (true_period_yr * 365.25)
        phase_shift = np.random.uniform(0, 1)
        
        # Choose Shape
        shape_choice = np.random.choice(['sin', 'gauss', 'emp'])
        if shape_choice == 'sin':
            y_shape = shape_sin
        elif shape_choice == 'emp':
            y_shape = shape_emp
        else:
            w = np.random.choice([0.05, 0.10, 0.15])
            y_shape = gauss_template(w, x_phase)
            
        y_shape = (y_shape - np.mean(y_shape)) / np.std(y_shape)
        
        # Map shape to time array
        phases = (time_mjd * true_freq + phase_shift) % 1.0
        qpo_signal = np.interp(phases, x_phase, y_shape)
        
        # Add Signal to Noise (Randomize SNR amplitude)
        # Standard deviation of noise is ~1.0 by default from colorednoise
        amplitude = np.random.uniform(0.5, 2.5) # The challenge zone
        final_lc = sim_noise + (qpo_signal * amplitude)
        final_lc[nan_mask] = np.nan
        
        feats = extract_10_features(time_mjd, final_lc, nan_mask, ftp_models, freqs)
        feats['label'] = 1
        feats['injected_alpha'] = alpha
        feats['base_mask'] = base_blazar_id
        results.append(feats)
        
    return results

def main():
    # 1. Setup Models
    x = np.linspace(0, 1, 500)
    try:
        forma_template = np.loadtxt('data/PG1553_template_shape.csv', delimiter=',')
        t_emp = Template.from_sampled(forma_template, nharmonics=4, template_id='Emp')
    except:
        t_emp = None
        
    t_sin = Template.from_sampled(np.sin(2 * np.pi * x), nharmonics=1, template_id='Sin')
    
    t_gauss_list = []
    for w in [0.05, 0.10, 0.15]:
        y = gauss_template(w, x)
        y = (y - np.mean(y)) / np.std(y)
        t_gauss_list.append(Template.from_sampled(y, nharmonics=int(6 - (w*20)), template_id=f'Gauss_{w}'))

    # Build FTP instances
    ftp_models = [FastTemplatePeriodogram(template=t_sin, allow_negative_amplitudes=False)]
    if t_emp is not None:
        ftp_models.append(FastTemplatePeriodogram(template=t_emp, allow_negative_amplitudes=False))
    for t in t_gauss_list:
        ftp_models.append(FastTemplatePeriodogram(template=t, allow_negative_amplitudes=False))

    freqs = np.linspace(1/(5.5*365.25), 1/(1*365.25), 200) # Slightly reduced resolution for ML generation speed

    # 2. Load Data & Alphas
    raw_features = pd.read_hdf('data/raw_curves_features.h5', key='data')
    real_alphas = raw_features[raw_features['alpha_psd'] >= 0.1]['alpha_psd'].values

    # 3. Parallel Execution
    SAMPLES_PER_BLAZAR = 250 # Creates 250 noise + 250 QPO per blazar mask -> 50,000 total rows for 100 blazars
    results_master = []
    
    with pd.HDFStore('data/df_candidates_bin7_filter30.h5', mode='r') as reader:
        candidates = [k[1:] for k in reader.keys()]
        cores = os.cpu_count()
        print(f"Generating ML Dataset using {cores} cores...")
        
        with ProcessPoolExecutor(max_workers=cores) as executor:
            futures = {}
            for blazar in candidates:
                df_blazar = reader.get(blazar)
                f = executor.submit(generate_samples, blazar, df_blazar, real_alphas, ftp_models, freqs, t_sin, t_emp, t_gauss_list, SAMPLES_PER_BLAZAR)
                futures[f] = blazar
                
            for i, f in enumerate(as_completed(futures), 1):
                blazar = futures[f]
                try:
                    res = f.result()
                    results_master.extend(res)
                    print(f"[{i}/{len(candidates)}] Processed gap mask from {blazar} (+{len(res)} rows)")
                except Exception as e:
                    print(f"Error on {blazar}: {e}")

    # 4. Save Final ML Dataset
    df_ml = pd.DataFrame(results_master)
    df_ml.to_csv('data/ML_training_set.csv', index=False)
    print(f"Finished! Saved {len(df_ml)} rows to ML_training_set.csv")

if __name__ == '__main__':
    main()
