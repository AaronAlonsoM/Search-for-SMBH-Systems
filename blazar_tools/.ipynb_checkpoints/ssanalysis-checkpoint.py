import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SSA(object):
    
    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, t, tseries, L, save_mem=False):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list. 
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.

        Methods
        ----------
        components_to_df, reconstruct, plot_wcorr, grouping
        
        """
        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.orig_TS.index=t
        self.K = self.N - self.L + 1
        
        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)
        
        self.TS_comps = np.zeros((self.N, self.d))
        
        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # Diagonally average the elementary matrices, store them as columns in array.           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."
        
        # Calculate the w-correlation matrix.
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        #self.orig_TS.index=t
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            
    
    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
    
    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.calc_wcorr()
        
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\widetilde{Y}_N^{(i)}$")
        plt.ylabel(r"$\widetilde{Y}_N^{(j)}$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label(r'$\rho_w(\widetilde{Y}_N^{(i)}, \widetilde{Y}_N^{(j)})$')
        plt.clim(0,1)
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)

    def auto_reconstruction(self, max_search = 4):
        """
        Reconstructs the components as trend, oscillatory and noise.
        
        It selects the two more correlated components among the first n = max_search as the oscillatory components.
        Trend and noise cointain the components before and after the oscillatory group.

        Returns a df with the components and the time as index, with max_corr and the tuple (i_osc, j_osc) as attributes.
   
        """
        corr_matrix = self.Wcorr
        i_osc = None  # Valor centinela para saber si encontramos algo
        j_osc = None
        self.max_corr = 0
        # Buscar el par oscilatorio (evitando salirnos del rango con -1)
        for i in range(max_search):
            corr = abs(corr_matrix[i][i+1])
            if corr > self.max_corr:
              i_osc = i
              j_osc = i+1
              self.max_corr = corr
            if self.max_corr > 0.8:
              break
        # Reconstrucción correcta
        # Tendencia: Todo lo anterior al par encontrado
        # Si i_osc es 0, slice(0,0) devuelve vacío, lo cual es correcto (no hay tendencia)
        self.trend = self.reconstruct(slice(0, i_osc))
        
        # Oscilación: El par encontrado (i, i+1)
        # Nota: slice(start, end) el 'end' no se incluye, así que es i_osc + 2
        self.osc = self.reconstruct(slice(i_osc, j_osc+1))
        
        # Ruido: Todo lo que va después del par hasta el final absoluto
        self.noise = self.reconstruct(slice(j_osc+1, -1))
        t = self.orig_TS.index

        dict = {'trend': self.trend.values , 'oscillatory': self.osc.values , 'noise' : self.noise.values}
        df = pd.DataFrame(dict)
        df.index = t
        osc_idx = (i_osc, j_osc)

        df.attrs['osc_idx'] = (i_osc, j_osc)
        df.attrs['max_corr'] = self.max_corr
        
        return df