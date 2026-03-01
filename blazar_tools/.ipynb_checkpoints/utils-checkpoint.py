import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_analysis(df_data, df_ssa, blazar, split = None, save = False):
    '''
    Plots the flux data and the SSA components.

    Inputs:
    ------
    df_ssa: DataFrame containing the columns trend, oscillatory and noise, with the time (MJD) as index.
    df_data : DataFrame containing de Fermi LCR data for an specific blazar.
    blazar: blazar name
    split: split number 
    save: for saving the plot
    
    '''
    
    df_data_cleaned = df_data[['t_year', 't_mjd' ,'flux', 'flux_error']].dropna()
    time_mjd = df_data_cleaned.t_mjd.values
    time_years = df_data_cleaned.t_year.values

    scale = 1e-8
    
    flux = df_data_cleaned.flux.values  / scale
    flux_error = df_data_cleaned.flux_error.values  / scale

    trend = df_ssa.trend.values  / scale
    osc = df_ssa.oscillatory.values / scale
    noise = df_ssa.noise.values / scale

    osc_idx = df_ssa.attrs.get('osc_idx')

    f_size=12
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.errorbar(time_mjd, flux, xerr = 0, fmt = 'o', yerr = flux_error, ms=4, label=r'$Fermi$-LAT data')
    
    
    if 'upper_limits' in list(df_data.columns):
        
        df_limits = df_data[['t_mjd', 'upper_limits']].dropna()
        upper_limits = pd.Series(df_limits.upper_limits.values, index = df_limits.t_mjd.values)
        
        time_mjd = df_data.t_mjd.values

        ax1.plot(time_mjd, noise, alpha = 0.6, label = 'Noise')
        ax1.plot(time_mjd, osc, linestyle = 'dashed', lw = 2, label = f'Periodicity {osc_idx}')
        ax1.plot(time_mjd, trend,  linestyle = 'dashed', lw =2,label = f'Trend')
        ax1.scatter(upper_limits.index, upper_limits / scale, marker = 'v', c = 'black', s = 10, label = 'Upper limits')
            
    else:
        ax1.plot(time_mjd, noise, alpha = 0.6, label = 'Noise')
        ax1.plot(time_mjd, osc, linestyle = 'dashed', lw = 2, label = f'Periodicity {osc_idx}')
        ax1.plot(time_mjd, trend,  linestyle = 'dashed', lw =2,label = f'Trend')
        
    if split is None:
        ax1.text(0.5, 0.95, blazar, transform=ax1.transAxes, fontsize=f_size, ha='center', va='top')
    else :
        ax1.text(0.5, 0.95, f'{blazar} (Split {split})', transform=ax1.transAxes, fontsize=f_size, ha='center', va='top')
    # 5. AÑADIMOS EL EJE SUPERIOR (usando el objeto ax1 directamente)
    secax = ax1.secondary_xaxis('top', functions=(mjd_to_year, year_to_mjd))
    secax.set_xlabel('Time (Year)', fontsize=f_size)
    secax.tick_params(labelsize=f_size)
    
    ax1.set_ylabel(r"Photon Flux $\;( \times 10^{-8} \, \mathrm{ph\,cm^{-2}\,s^{-1}})$", fontsize = f_size)
    ax1.set_xlabel(r'Time (MJD)', fontsize = f_size)
    ax1.tick_params(axis='both', which='major', labelsize=f_size)
    ax1.legend(fontsize=f_size, loc='upper right')
    fig.tight_layout()
    
    if save:
        if split is None:
            fig.savefig(f"Reconstructed Components of {blazar}.pdf")
        else :
            fig.savefig(f"Reconstructed Components of {blazar} Split {split}.pdf")
    plt.show()


def time_splits(df_data, splits = 1, overlap = 0):
    '''
    Splits the df containing the LCR data of a blazar into the number of splits give. 
    If overlap is introduced as a float from 0 to 1, then the splits will overlap with a step given by that amount.
    
    Returns a list containing each split's df
    '''
    list = []
    n = len(df_data)
    split_size = n // splits 
    step = int( (1 - overlap) * split_size )
    dict = {}
    for i in range(splits):
        idx_0 = i * step
        if i == splits - 1:
            idx_1 = n
        else:
            idx_1 = idx_0 + split_size
    
        df_split = df_data.iloc[idx_0:idx_1].copy()
        list.append(df_split)
    return list


def plot_split_pgram(pgram_list, blazar):
    '''
    For a given blazar, plots the full and each of the splits periodograms together.
    pgram_list must be a list of dictionaries containing the keys pgram, period and fwhm of each split.
    
    '''

    fig, ax1 = plt.subplots(figsize=(10, 6))
    max_power = np.max([np.max(item['pgram'].values) for item in pgram_list])
    
    for i in range(len(pgram_list)):
        pgram_info = pgram_list[i]
        pgram = pgram_info['pgram'] / max_power
        period = pgram_info['period']
        fwhm  = pgram_info['fwhm']
    
        if i == 0:
            label = f'Full. {period:.2f} $\pm$ {fwhm:.2f} yr'
        else :
            label = f'Split {i}. {period:.2f} $\pm$ {fwhm:.2f} yr'
            
        ax1.plot(pgram.index, pgram, label = label)
        
    ax1.text(0.5, 0.95, blazar, transform=ax1.transAxes, fontsize = 12, ha='center', va='top')
    ax1.legend( loc='upper right')
    fig.tight_layout()
    plt.show()

def mjd_to_year(mjd):
        return 2000 + (mjd - 51544) / 365.25

def year_to_mjd(year):
    return (year - 2000) * 365.25 + 51544