"""Collection of functions to apply the time-temperature superposition
principle to create a master curve from measurements performed at different
temperatures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



"""----------------------------------------------------------------------------
Shift functions
"""

def WLF(Temp, RefT, WLF_C1, WLF_C2):
    return -WLF_C1*(Temp - RefT)/(WLF_C2+(Temp - RefT))

def poly1(x,C0,C1):
    return C0+C1*x

def poly2(x,C0,C1,C2):
    return C0+C1*x+C2*x**2

def poly3(x,C0,C1,C2,C3):
    return C0+C1*x+C2*x**2+C3*x**3

def poly4(x,C0,C1,C2,C3,C4):
    return C0+C1*x+C2*x**2+C3*x**3+C4*x**4

def fit_WLF(RefT, df_aT):

    xdata = df_aT['T'].values+273.15
    ydata = df_aT['log_aT'].values

    popt, _pcov = curve_fit(lambda x, C1, C2: WLF(x, RefT+273.15, C1, C2), 
        xdata, ydata, p0 = [1E3, 5E3], bounds=(0, 5000))

    #Put fitted WLF shift function in df
    df = pd.DataFrame(data = np.insert(popt,0,RefT)).T 
    df.columns = ['RefT', 'C1', 'C2']

    return df

def fit_poly(df_aT):

    df_K = pd.DataFrame(np.zeros((5, 4)), 
        index = ['C0', 'C1', 'C2', 'C3', 'C4'], 
        columns=['D4', 'D3', 'D2', 'D1'],
        dtype=float)

    #Kelvin
    xdata = df_aT['T'].values+273.15
    ydata = df_aT['log_aT'].values

    df_K['D4'][['C0', 'C1', 'C2', 'C3', 'C4']], _pcov = curve_fit(poly4, xdata, ydata)
    df_K['D3'][['C0', 'C1', 'C2', 'C3']], _pcov = curve_fit(poly3, xdata, ydata)
    df_K['D2'][['C0', 'C1', 'C2']], _pcov = curve_fit(poly2, xdata, ydata)
    df_K['D1'][['C0', 'C1']], _pcov = curve_fit(poly1, xdata, ydata)
#
    ##Celsius

    df_C = pd.DataFrame(np.zeros((5, 4)), 
    index = ['C0', 'C1', 'C2', 'C3', 'C4'], 
    columns=['D4', 'D3', 'D2', 'D1'],
    dtype=float)

    xdata = df_aT['T'].values
    ydata = df_aT['log_aT'].values
#
    df_C['D4'][['C0', 'C1', 'C2', 'C3', 'C4']], _pcov = curve_fit(poly4, xdata, ydata)
    df_C['D3'][['C0', 'C1', 'C2', 'C3']], _pcov = curve_fit(poly3, xdata, ydata)
    df_C['D2'][['C0', 'C1', 'C2']], _pcov = curve_fit(poly2, xdata, ydata)
    df_C['D1'][['C0', 'C1']], _pcov = curve_fit(poly1, xdata, ydata)

    df_C = df_C.T
    df_K = df_K.T

    coeff_C = df_C.columns.get_level_values(0)
    C = ['°C', '-', '°C^-1', '°C^-2', '°C^-3']
    df_C.columns = pd.MultiIndex.from_tuples(zip(coeff_C, C), names = ['Degree', '-'])

    coeff_K = df_K.columns.get_level_values(0)
    K = ['K', '-', 'K^-1', 'K^-2', 'K^-3']
    df_K.columns = pd.MultiIndex.from_tuples(zip(coeff_K, K), names = ['Degree', '-'])

    return df_C, df_K


def plot(df_aT, df_WLF, df_poly):

    x = df_aT['T'].values
    y_aT = df_aT['log_aT'].values

    y_WLF = WLF(df_aT['T'], df_WLF['RefT'].values, df_WLF['C1'].values, df_WLF['C2'].values)

    df_p = df_poly.copy()
    df_p.columns = df_poly.columns.droplevel(1)
    df_p = df_p.T

    y_poly4 = poly4(df_aT['T'], *df_p['D4'])
    y_poly3 = poly3(df_aT['T'], *df_p['D3'][0:4])
    y_poly2 = poly2(df_aT['T'], *df_p['D2'][0:3])
    y_poly1 = poly1(df_aT['T'], *df_p['D1'][0:2])

    fig, ax = plt.subplots()
    ax.plot(x, y_aT, ls='', marker='o', c='r', label='Shift factors',zorder=10)
    ax.plot(x, y_WLF, c='k', label='WLF', ls='-')
    ax.plot(x, y_poly4, label='D4 polynomial', c='0.2', ls='--')
    ax.plot(x, y_poly3, label='D3 polynomial', c='0.4', ls=':')
    ax.plot(x, y_poly2, label='D2 polynomial', c='0.6', ls='-.')
    ax.plot(x, y_poly1, label='D1 polynomial', c='0.8', ls='-')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel(r'$\log(a_{\mathrm{T}})$')
    ax.legend()

    fig.show()
  
    #Collect figure data in dataframe
    df_shift = pd.DataFrame(np.zeros((x.shape[0], 7)), 
        columns=(['T', 'log_aT', 'WLF', 'Poly1', 'Poly2', 'Poly3', 'Poly4']))
    df_shift['T'] = x
    df_shift['log_aT'] = y_aT
    df_shift['WLF'] = y_WLF
    df_shift['Poly1'] = y_poly1
    df_shift['Poly2'] = y_poly2
    df_shift['Poly3'] = y_poly3
    df_shift['Poly4'] = y_poly4

    return fig, df_shift



