import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lombscargle, detrend
from scipy.optimize import curve_fit
import datetime
from ftperiodogram.modeler import FastTemplatePeriodogram, FastMultiTemplatePeriodogram, TemplateModel
from ftperiodogram.template import Template

from .utils import mjd_to_year

class Periodogram:
    '''
    Computes the periodogram of a temporal series. 
    
    Parameters
        ----------
        df_ssa : DataFrame from SSA, inluding the trend, oscillatory and noise components with time (MJD) as index. 
        T_min, T_max: Minimum and maximum periods to look for. By default, set to 1 and 6 years respectively.
        signal_error: relevant only for FTP as it is used to compute different weights for the points
        split: number of the split of the analysis

        Functions
        ----------
        LSP, FTP_sin, FTP_peak
    
    '''

    def  __init__ (self, df_ssa, blazar = 'None', signal_error = None, T_min = 1, T_max = 6, split = None):
        
        self.blazar = blazar
        self.df = df_ssa
        self.time = mjd_to_year(df_ssa.index.values)
        self.signal_error = signal_error
        self.periods = np.linspace(max(T_min, 0), T_max, 500)  # Periods in years
        self.T_max = T_max
        self.freqs = 1 / self.periods           # Angular frequencies
        self.ang_freqs = 2 * np.pi / self.periods
        self.split = split

        
        
    def find_adaptive_window(self, max_window=50, min_window=8):
        idx_peak = self.max_idx
        pgram = self.pgram
        
        start = idx_peak
        while start > 0 and (idx_peak - start) < max_window:
            # Si el valor actual es mayor que el siguiente (yendo hacia atrás), 
            # significa que hemos encontrado un mínimo y empieza a subir otro pico
            if pgram[start - 1] > pgram[start]:
                break
            start -= 1
            
        # 3. Buscar hacia la DERECHA el primer valle
        end = idx_peak
        while end < len(pgram) - 1 and (end - idx_peak) < max_window:
            if pgram[end + 1] > pgram[end]:
                break
            end += 1
    
        # 4. Asegurar un ancho mínimo para que curve_fit tenga puntos suficientes
        if (idx_peak - start) < min_window: start = max(0, idx_peak - min_window)
        if (end - idx_peak) < min_window: end = min(len(pgram), idx_peak + min_window)
    
        self.x_data = self.periods[start:end]
        self.y_data = self.pgram[start:end]
        
        return start, end

        
        
    def gaussian(self, x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        
    def gaussian_fit(self):

        # 3. Recortar datos alrededor del pico para facilitar el ajuste
        # (Tomamos una ventana alrededor del pico para que el ajuste no se distraiga con otros picos)
        
        self.find_adaptive_window()
        
        try:
            # 4. Ajustar Gaussiana
            # p0 son los valores iniciales [altura, centro, ancho]
            self.popt, pcov = curve_fit(self.gaussian, self.x_data, self.y_data, p0=[self.h_peak - np.min(self.pgram) , self.p_peak, 0.2])
            
            sigma_fit = abs(self.popt[2]) # El sigma ajustado
            self.fwhm = 2.355 * sigma_fit
        except (RuntimeError, ValueError):
            self.fwhm = np.nan
            return np.nan

    
    def plot(self, save = False):

        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S%MS")

        if self.template_name is not None:
            # Visualizacion
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [1, 1.5]})
            
            # Gráfico del Periodograma
            ax1.plot(self.periods , self.pgram, color='royalblue', label = 'Periodogram')
            if self.fwhm is not np.nan:
                pass
                # ax1.plot(self.x_data, gaussian(self.x_data, *self.popt), 'r--', label=f"Gaussian fit. Peak at: {self.p_peak:.3f}. FWHM: {self.fwhm:.3f}", alpha=0.75)
            ax1.text(0.85, 0.95, self.blazar, transform=ax1.transAxes, fontsize=12, ha='center', va='top')    
            ax1.text(0.85, 0.88, f'{self.p_peak:.2f} $\pm$ {self.fwhm:.2f} yr', transform=ax1.transAxes, fontsize=12, ha='center', va='top')
            ax1.set_xlabel("Period")
            ax1.set_ylabel("Normalized Power")
            
            # Gráfico de la curva de luz 
            ax2.scatter(self.time, self.signal, marker='o', s = 5, label='Oscillatory component')
            #ax2.plot(self.time, self.signal, c = 'royalblue', alpha = 0.7)
            if self.sigma_best_fit is None:
                ax2.plot(self.time, self.best_model(self.time), c = 'g', lw=1, alpha = 0.7, label='Fitted Template')
            else :
                ax2.plot(self.time, self.best_model(self.time), c = 'g', lw=1, alpha = 0.7, label= f'Fitted Template ($\sigma$: {self.sigma_best_fit})')
            
            ax2.set_xlabel("Time")
            ax2.set_ylabel(r"Amplitude $\;( \times 10^{-8} \, \mathrm{ph\,cm^{-2}\,s^{-1}})$")
            
            ax2.legend()
                
            fig.subplots_top: 0.88
            plt.tight_layout()
            if save:
                if self.split is None:
                    plt.savefig(f'Full FTP {self.template_name } for {self.blazar} {timestamp}.pdf')
                else:
                    plt.savefig(f'FTP {self.template_name } Split {self.split} for {self.blazar} {timestamp}.pdf')
            plt.show()
            
        else:
            # Gráfico del Periodograma
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(self.periods, self.pgram / np.max(self.pgram), color='royalblue', label = 'Periodogram')
            if self.fwhm is not np.nan:
                pass
                # ax1.plot(self.x_data, self.gaussian(self.x_data, *self.popt), 'r--', label=f"Peak: {self.p_peak:.3f}. FWHM: {self.fwhm:.3f}")
            if self.split is None: 
                # ax1.set_title(f'Full LSP Periodogram for {self.blazar}')
                pass
            else:
                # ax1.set_title(f'LSP Periodogram Split {self.split} for {sef.blazar}')
                pass
            ax1.text(0.9, 0.95, self.blazar, transform=ax1.transAxes, fontsize=12, ha='center', va='top')    
            ax1.text(0.9, 0.9, f'{self.p_peak:.2f} $\pm$ {self.fwhm:.2f} yr', transform=ax1.transAxes, fontsize=12, ha='center', va='top')
            ax1.set_xlabel(r'Period')
            ax1.set_ylabel("Normalized Power")
            if save:
                if self.split is None:
                    plt.savefig(f'Full LSP for {self.blazar} {timestamp}.pdf')
                else:
                    plt.savefig(f'LSP Split {self.split} for {self.blazar} {timestamp}.pdf')
            plt.show()


    def LSP(self,  save = False, plot = True):
        '''
        Options: plot = False, save = False.
        Returns a dictionary containing pgram, p_peak and fwhm.
        
        '''

        self.template_name = None
        self.signal = self.df.oscillatory.values
        self.pgram = lombscargle(self.time, self.signal, self.ang_freqs, normalize=True)
        self.max_idx = np.argmax(self.pgram)
        self.p_peak = self.periods[self.max_idx]
        self.h_peak = self.pgram[self.max_idx]

        self.gaussian_fit()

        if plot:
            if self.split is None:
                print(f"Full LSP Period detected: {self.p_peak:.2f} +- {self.fwhm:.2f} yr")
            else:
                print(f" Split {self.split} LSP Period detected: {self.p_peak:.2f} +- {self.fwhm:.2f} yr")
            self.plot(save = save)
        s = pd.Series(self.pgram, index = self.periods) 
        d = {'pgram': s, 'period': self.p_peak, 'fwhm': self.fwhm}
        return d

    def FTP(self, template, save = False, plot = True):

        self.sigma_best_fit = None # This will take a value for a gaussian peak template only
        self.periods = np.linspace(8, 50, 500)
        self.freqs = 1/self.periods
        
        scale = 1e-8
        self.signal = self.df.oscillatory.values / scale# From pyts
        
        # Template definition
        self.template_name = template.template_id
        
        if self.signal_error is None: # Equal weights for every point if no error is provided
            error = np.ones_like(self.signal)
 
        ftp = FastTemplatePeriodogram(template)
        ftp.fit(self.time, self.signal, error)
        self.pgram = ftp.power(self.freqs)
        self.best_model = ftp.best_model

        self.periods *= 2.01/ 17.33 
        self.max_idx = np.argmax(self.pgram)
        self.p_peak = self.periods[self.max_idx]
        self.h_peak = self.pgram[self.max_idx]
    
        self.gaussian_fit()
    
        if self.split is None:
            print(f"Full FTP {self.template_name } Period detected: {self.p_peak:.2f} +- {self.fwhm:.2f} yr")
        else :
            print(f"FTP {self.template_name } Split: {self.split} Period detected: {self.p_peak:.2f} +- {self.fwhm:.2f} yr")
            
        if plot:
            self.plot(save = save)
        s = pd.Series(self.pgram, index = self.periods) 
        d = {'pgram': s, 'period': self.p_peak, 'fwhm': self.fwhm}
        return d