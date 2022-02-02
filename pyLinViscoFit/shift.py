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

    xdata = df_aT['Temp'].values+273.15
    ydata = df_aT['log_aT'].values

    popt, _pcov = curve_fit(lambda x, C1, C2: WLF(x, RefT+273.15, C1, C2), 
        xdata, ydata, p0 = [1E3, 5E3], bounds=(0, 5000))

    #Put fitted WLF shift function in df
    df = pd.DataFrame(data = np.insert(popt,0,RefT)).T 
    df.columns = ['RefT', 'C1', 'C2']

    return df

def fit_poly(df_aT):

    df = pd.DataFrame(np.zeros((5, 8)), 
        index = ['C0', 'C1', 'C2', 'C3', 'C4'], 
        columns=['P4 (K)', 'P3 (K)', 'P2 (K)', 'P1 (K)', 'P4 (C)', 'P3 (C)','P2 (C)','P1 (C)'],
        dtype=float)

    #Kelvin
    xdata = df_aT['Temp'].values+273.15
    ydata = df_aT['log_aT'].values

    df['P4 (K)'][['C0', 'C1', 'C2', 'C3', 'C4']], _pcov = curve_fit(poly4, xdata, ydata)
    df['P3 (K)'][['C0', 'C1', 'C2', 'C3']], _pcov = curve_fit(poly3, xdata, ydata)
    df['P2 (K)'][['C0', 'C1', 'C2']], _pcov = curve_fit(poly2, xdata, ydata)
    df['P1 (K)'][['C0', 'C1']], _pcov = curve_fit(poly1, xdata, ydata)
#
    ##Celsius
    xdata = df_aT['Temp'].values
#
    df['P4 (C)'][['C0', 'C1', 'C2', 'C3', 'C4']], _pcov = curve_fit(poly4, xdata, ydata)
    df['P3 (C)'][['C0', 'C1', 'C2', 'C3']], _pcov = curve_fit(poly3, xdata, ydata)
    df['P2 (C)'][['C0', 'C1', 'C2']], _pcov = curve_fit(poly2, xdata, ydata)
    df['P1 (C)'][['C0', 'C1']], _pcov = curve_fit(poly1, xdata, ydata)

    return df


def plot(df_aT, df_WLF, df_poly):

    x = df_aT['Temp'].values
    y_aT = df_aT['log_aT'].values

    y_WLF = WLF(df_aT['Temp'], df_WLF['RefT'].values, df_WLF['C1'].values, df_WLF['C2'].values)

    y_poly4 = poly4(df_aT['Temp'], *df_poly['P4 (C)'])
    y_poly3 = poly3(df_aT['Temp'], *df_poly['P3 (C)'][0:4])
    y_poly2 = poly2(df_aT['Temp'], *df_poly['P2 (C)'][0:3])
    y_poly1 = poly1(df_aT['Temp'], *df_poly['P1 (C)'][0:2])

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



