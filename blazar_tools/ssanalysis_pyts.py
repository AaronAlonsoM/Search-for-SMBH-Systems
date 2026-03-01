from pyts.decomposition import SingularSpectrumAnalysis
import pandas as pd

def SSA_pyts(t_mjd, flux, L, freq_bound = 0.05, c_bound = 0.85):
    ssa = SingularSpectrumAnalysis(window_size= L, groups='auto', lower_frequency_bound = freq_bound, lower_frequency_contribution = c_bound)
    X = (flux, t_mjd) # ( signal, time )
    X_ssa = ssa.fit_transform(X)
    d = {'trend': X_ssa[0,0], 'oscillatory': X_ssa[0, 1], 'noise': X_ssa[0,2]}
    df_ssa = pd.DataFrame( d, index = X_ssa[1][0])
    return df_ssa