import os
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import joblib
from scipy.interpolate import interp1d
from scipy.stats import skew
from scipy.sparse.linalg import svds
from astropy.timeseries import LombScargle
from ftperiodogram.modeler import FastTemplatePeriodogram
from ftperiodogram.template import Template

# ==========================================
# 1. CORE FUNCTIONS (Must match training EXACTLY)
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

def extract_10_features(time_mjd, flux, nan_mask, ftp_models, freqs):
    features = {}
    periods = 1 / (freqs * 365.25)
    total_obs_years = (time_mjd[-1] - time_mjd[0]) / 365.25
    
    # EXACT MASK FROM TRAINING: Ignore periods > 5.0 years
    valid_mask = periods <= 5.0
    valid_indices = np.where(valid_mask)[0]
    
    idx_nans = np.flatnonzero(nan_mask)
    idx_valid = np.flatnonzero(~nan_mask)
    flux_filled = flux.copy()
    flux_filled[nan_mask] = np.interp(idx_nans, idx_valid, flux[~nan_mask])
    
    features['autocorrelation_raw'] = pd.Series(flux_filled).autocorr(lag=1)
    
    L = int(0.4 * len(flux_filled))
    ssa = SSAFast(flux_filled, L, time_mjd)
    trend = find_trend(ssa, time_mjd, period_max_yr=6)
    
    osc_array_filled = flux_filled - trend.values
    
    noise_rc = ssa.reconstruct(range(ssa.d - 5, ssa.d)).values 
    features['SNR_osc'] = np.std(osc_array_filled) / (np.std(noise_rc) + 1e-9)
    features['autocorrelation_osc'] = pd.Series(osc_array_filled).autocorr(lag=1)
    
    time_valid = time_mjd[~nan_mask]
    flux_detrended = osc_array_filled[~nan_mask]
    flux_err = np.ones_like(flux_detrended)
    
    ls = LombScargle(time_valid, flux_detrended)
    power_ls = ls.power(freqs)
    max_ls = np.max(power_ls[valid_mask])
    
    best_power = 0.0
    best_pgram = None
    best_ftp = None
    best_template_name = ""
    
    for ftp in ftp_models:
        ftp.fit(time_valid, flux_detrended, flux_err)
        pgram = ftp.power(freqs, save_best_model=True)
        max_p = np.max(pgram[valid_mask])
        if max_p > best_power:
            best_power = max_p
            best_pgram = pgram
            best_ftp = ftp
            best_template_name = ftp.template.template_id

    best_idx_in_valid = np.argmax(best_pgram[valid_mask])
    best_idx = valid_indices[best_idx_in_valid]
    best_freq = freqs[best_idx]
    best_period_yr = periods[best_idx]
    
    features['Power_Ratio_FTP_vs_LSP'] = best_power / (max_ls + 1e-9)
    
    half_max = best_power / 2.0
    left_idx = best_idx
    while left_idx > 0 and best_pgram[left_idx] > half_max: left_idx -= 1
    right_idx = best_idx
    while right_idx < len(best_pgram)-1 and best_pgram[right_idx] > half_max: right_idx += 1
    features['periodogram_fwhm'] = periods[left_idx] - periods[right_idx] 

    mask_h2 = np.ones(len(best_pgram), dtype=bool)
    mask_h2[max(0, best_idx - 10):min(len(best_pgram), best_idx + 10)] = False
    mask_h2 = mask_h2 & valid_mask
    secondary_power = np.max(best_pgram[mask_h2]) if np.any(mask_h2) else 1e-9
    features['FTP_prominence_ratio'] = best_power / (secondary_power + 1e-9)
    
    mid_idx = len(time_valid) // 2
    if mid_idx > 10:
        t1, f1 = time_valid[:mid_idx], flux_detrended[:mid_idx]
        t2, f2 = time_valid[mid_idx:], flux_detrended[mid_idx:]
        
        best_ftp.fit(t1, f1, np.ones_like(f1))
        pgram1 = best_ftp.power(freqs, save_best_model=False)
        p1 = periods[valid_indices[np.argmax(pgram1[valid_mask])]]
        
        best_ftp.fit(t2, f2, np.ones_like(f2))
        pgram2 = best_ftp.power(freqs, save_best_model=False)
        p2 = periods[valid_indices[np.argmax(pgram2[valid_mask])]]
        
        features['r_period_splits'] = abs(p1 - p2) / best_period_yr
    else:
        features['r_period_splits'] = 0.0

    phase_grid = np.linspace(0, 1, 100)
    time_grid = phase_grid / best_freq
    folded_model = best_ftp.best_model(time_grid)
    
    features['Folded_Skewness'] = skew(folded_model)
    threshold = np.max(folded_model) * 0.5
    features['Duty_Cycle'] = np.sum(folded_model >= threshold) / len(folded_model)
    features['number_of_cycles'] = total_obs_years / best_period_yr

    # Return features AND metadata for the final output table
    return features, best_period_yr, best_template_name

# ==========================================
# 2. MAIN INFERENCE EXECUTION
# ==========================================
def main():
    print("Loading Machine Learning Model...")
    try:
        rf_model = joblib.load('data/qpo_rf_model.joblib')
        # Extract the exact feature names the model expects
        expected_features = rf_model.feature_names_in_ 
    except FileNotFoundError:
        print("Error: 'data/qpo_rf_model.joblib' not found. Run the training script first.")
        return

    # Setup Templates
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

    ftp_models = [FastTemplatePeriodogram(template=t_sin, allow_negative_amplitudes=False)]
    if t_emp is not None:
        ftp_models.append(FastTemplatePeriodogram(template=t_emp, allow_negative_amplitudes=False))
    for t in t_gauss_list:
        ftp_models.append(FastTemplatePeriodogram(template=t, allow_negative_amplitudes=False))

    freqs = np.linspace(1/(5.5*365.25), 1/(1*365.25), 200)

    # Targets to evaluate
    robust_targets = [
        'J1555.7+1111', 'J0102.8+5824', 'J0809.8+5218', 
        'J1058.4+0133', 'J0211.2+1051', 'J0303.4-2407', 'J1427.9-4206', 
        'J1048.4+7143', 'J0739.2+0137'
    ]
    
    results = []
    
    print("\nExtracting features and predicting probabilities for real candidates...")
    with pd.HDFStore('data/df_candidates_bin7_filter30.h5', mode='r') as reader:
        for target in robust_targets:
            if target not in reader:
                print(f"Warning: {target} not found in HDF5 file.")
                continue
                
            df_blazar = reader.get(target)
            time_mjd = df_blazar.t_mjd.values
            flux = df_blazar.flux.values
            nan_mask = df_blazar.flux.isna().values
            
            # Extract Features
            feats_dict, period, best_temp = extract_10_features(time_mjd, flux, nan_mask, ftp_models, freqs)
            
            # Convert to DataFrame ensuring column order matches training exactly
            X_real = pd.DataFrame([feats_dict])[expected_features]
            
            # Predict
            prob_qpo = rf_model.predict_proba(X_real)[0][1] # Probability of Class 1 (QPO)
            prediction = "QPO" if prob_qpo >= 0.50 else "Noise"
            
            results.append({
                'Source': target,
                'ML_Probability': prob_qpo,
                'Classification': prediction,
                'Period (yr)': round(period, 2),
                'Best Template': best_temp
            })
            
            print(f"[{target}] Prob: {prob_qpo*100:.1f}% | Pred: {prediction} | Per: {period:.2f}yr")

    # Save and Display Final Results
    df_results = pd.DataFrame(results).sort_values(by='ML_Probability', ascending=False)
    df_results.to_csv('data/final_ML_predictions.csv', index=False)
    
    print("\n" + "="*60)
    print("FINAL MACHINE LEARNING EVALUATION")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    print("Saved complete predictions to 'data/final_ML_predictions.csv'")

if __name__ == '__main__':
    main()
