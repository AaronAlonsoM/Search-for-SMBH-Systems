import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

class GetLCRData:
    """
    Gets blazar flux data from Fermi LAT Light Curve Repository (LCR)
    Rquest package is needed

    Inputs
    ---
    path: path of the txt file with the names of the blazars and links (JSON)
    
    """
    def __init__(self, path):
        self.links = dict( np.loadtxt(path, dtype = 'str') )
        self.blazar_list = list(self.links.keys())
        self.df = None
        self.df_dict = {}
        for blazar in self.blazar_list:   
            
            url = self.links[blazar]
            # Fetch data from the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an error if the request failed
            # Parse the JSON response
            data_dict = response.json()
            df = self.to_df(data_dict)
            self.df_dict[blazar] = df

    def mjd_to_years(self, mjd):
        return 2000 + (mjd - 51544) / 365.25


    def to_df(self, data_dict):
        # Extract 'flux' and 'flux_error' data
        # self.t_met = np.array(data_dict['flux'])[:,0]
        flux_data = np.array(data_dict['flux'])
        upper_limits_data = np.array(data_dict['flux_upper_limits'])

        flux = flux_data[:,1]
        t_met = flux_data[:,0]
        flux_error = np.array([(item[2] - item[1]) / 2 for item in data_dict['flux_error']]) 

        dict = {'t_met': t_met,'flux':flux, 'flux_error':flux_error}   
        df = pd.DataFrame(dict)
        
        
        if len(upper_limits_data) > 0:
            
            t_met_upper_limits = upper_limits_data[:,0] 
            upper_limits = upper_limits_data[:,1]
            
            dict = {'upper_limits': upper_limits , 't_met' : t_met_upper_limits}
            df_limits = pd.DataFrame(dict) # New df with upper limits data only
            
            df = df.merge(df_limits, on = 't_met', how = 'outer', sort = True)
              
                
        # Fermi LAT reference MJD for MET = 0
        mjd_reference = 51910  
        
        # Convert MET to MJD
        t_mjd = mjd_reference + (df['t_met'].values / 86400)  # Convert seconds to days and add to reference MJD
        df['t_mjd'] = t_mjd
        # Convert MJD to decimal year 
        
        df['t_year'] = self.mjd_to_years(t_mjd)

        return df

    def plot(self, blazar):

        df = self.df_dict[blazar]
        
        df_cleaned = df[['t_year', 't_mjd' ,'flux', 'flux_error']].dropna()

        t_mjd = df_cleaned.t_mjd.values
        t_years = df_cleaned.t_year.values

        scale = 1e-8
        flux = df_cleaned.flux.values /scale 
        flux_error = df_cleaned.flux_error.values/scale 

        plt.figure(figsize=(12, 4))
        plt.errorbar(t_mjd, flux, xerr=0, yerr=flux_error, fmt = 'o', ms = 3)

        if 'upper_limits' in df.columns:
            t_mjd = df.t_mjd
            upper_limits = df['upper_limits'].values /scale
            flux_with_limits = df[['flux', 'upper_limits']].sum(axis = 1).values /scale
            plt.scatter(t_mjd, upper_limits, marker = 'v' , s = 14 , c= 'black', label='Upper limits')
            plt.plot(t_mjd, flux_with_limits, c = 'royalblue' ,alpha = 0.35)
            
        else:
            plt.plot(t_mjd, flux, c = 'royalblue' ,alpha = 0.35)

        plt.ylabel(r"Photon Flux $\;( \times 10^{-8} \, \mathrm{ph\,cm^{-2}\,s^{-1}})$")
        plt.xlabel('Date (MJD)')
        plt.title(f'Raw Light Curve of {blazar}')
        plt.show()   