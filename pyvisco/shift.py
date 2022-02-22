"""
Collection of functions to apply the time-temperature superposition
principle to create a master curve from measurements performed at different
temperatures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

def WLF(Temp, RefT, WLF_C1, WLF_C2):
    """
    Calculate the Williams-Landel-Ferry (WLF) equation [1].
    
    Parameters
    ----------
    Temp : numeric
        Evaluation temperature of the shift factor.
    RefT : numeric
        Reference temperature chosen to construct the master curve.
    WLF_C1, WLF_C2 : numeric
        Empirical constants. (Obtained from fitting the shift factor a_T)

    Returns
    -------
    log_aT : numeric
        The decadic logarithm of the WLF shift factor.

    References
    ----------
    [1] Williams, Malcolm L.; Landel, Robert F.; Ferry, John D. (1955). 
    "The Temperature Dependence of Relaxation Mechanisms in Amorphous Polymers 
    and Other Glass-forming Liquids". J. Amer. Chem. Soc. 77 (14): 3701-3707. 
    doi:10.1021/ja01619a008
    """
    log_aT = -WLF_C1*(Temp - RefT)/(WLF_C2+(Temp - RefT))
    return log_aT


def poly1(x,C0,C1):
    """
    Calculate a polynomial function of degree 1 with a single variable 'x'.
    
    Parameters
    ----------
    x : numeric
        Input variable.
    C0, C1 : numeric
        Polynomial coefficients

    Returns
    -------
    numeric
        Result of the polynomial function.
    """    
    return C0+C1*x


def poly2(x,C0,C1,C2):
    """
    Calculate a polynomial function of degree 2 with a single variable 'x'.
    
    Parameters
    ----------
    x : numeric
        Input variable.
    C0, C1, C2 : numeric
        Polynomial coefficients

    Returns
    -------
    numeric
        Result of the polynomial function.
    """   
    return C0+C1*x+C2*x**2


def poly3(x,C0,C1,C2,C3):
    """
    Calculate a polynomial function of degree 3 with a single variable 'x'.
    
    Parameters
    ----------
    x : numeric
        Input variable.
    C0, C1, C2, C3 : numeric
        Polynomial coefficients

    Returns
    -------
    numeric
        Result of the polynomial function.
    """  
    return C0+C1*x+C2*x**2+C3*x**3


def poly4(x,C0,C1,C2,C3,C4):
    """
    Calculate a polynomial function of degree 4 with a single variable 'x'.
    
    Parameters
    ----------
    x : numeric
        Input variable.
    C0, C1, C2, C3, C4 : numeric
        Polynomial coefficients

    Returns
    -------
    numeric
        Result of the polynomial function.
    """  
    return C0+C1*x+C2*x**2+C3*x**3+C4*x**4


def fit_WLF(RefT, df_aT):
    """
    Fit the Williams-Landel-Ferry (WLF) equation [1] to a set of shift factors.
    
    Parameters
    ----------
    RefT : numeric
        Reference temperature chosen to construct the master curve.
    df_aT : pandas.DataFrame
        Contains the decadic logarithm of the shift factors 'log_aT'
        and the corresponding temperature values 'T' in degree Celsius.

    Returns
    -------
    df : pandas.DataFrame
        Contains the necessary parameters to calculate the WLF
        equation (RefT, WLF_C1, WLF_C2).

    See also
    --------
    shift.WLF : Calculates the WLF equation.

    Note:
    -----
    Too avoid large WLF coefficients in cases where the shift factors are nearing
    a straight line, an upper bound of 5000 is set in the fitting routine for the
    empirical coefficients. In this case, only the ratio between C1 and C2
    is of importance and should not affect the goodness of the fit significantly.

    References
    ----------
    [1] Williams, Malcolm L.; Landel, Robert F.; Ferry, John D. (1955). 
    "The Temperature Dependence of Relaxation Mechanisms in Amorphous Polymers 
    and Other Glass-forming Liquids". J. Amer. Chem. Soc. 77 (14): 3701-3707. 
    doi:10.1021/ja01619a008
    """
    xdata = df_aT['T'].values
    ydata = df_aT['log_aT'].values

    popt, _pcov = curve_fit(lambda x, C1, C2: WLF(x, RefT, C1, C2),
        xdata, ydata, p0 = [1E3, 5E3], bounds=(0, 5000))

    df = pd.DataFrame(data = np.insert(popt,0,RefT)).T 
    df.columns = ['RefT', 'C1', 'C2']
    return df


def fit_poly(df_aT):
    """
    Fit polynomial functions of degree 1 to 4 to a set of shift factors.
    
    Parameters
    ----------
    df_aT : pandas.DataFrame
        Contains the decadic logarithm of the shift factors 'log_aT'
        and the corresponding temperature values 'T' in degree Celsius.

    Returns
    -------
    df_C : pandas.DataFrame
        Contains the coefficients to calculate the polynomial 
        shift functions of degree 1 to 4 for temperatures in degree **Celsius**.

    df_K : pandas.DataFrame
        Contains the coefficients to calculate the polynomial 
        shift functions of degree 1 to 4 for temperatures in **Kelvin**.

    Note:
    -----
    The coefficients of the polynomial shift funtions are dependent on the
    Temperature unit. Hence, two different dataframes are provided for
    temperatures described in Celsius and Kelvin. For temperatures in Kelvin,
    at least 5 significant figures should be used for the polynomial 
    coefficients to obtain accurate results for the polynomial shift functions. 

    The interconversion from degree Celsius (T_C) to Kelvin (T_K) is performed 
    as: T_K = T_C + 273.15.
    """
    #Kelvin
    df_K = pd.DataFrame(np.zeros((5, 4)), 
        index = ['C0', 'C1', 'C2', 'C3', 'C4'], 
        columns=['D4', 'D3', 'D2', 'D1'],
        dtype=float)

    xdata = df_aT['T'].values+273.15
    ydata = df_aT['log_aT'].values

    df_K['D4'][['C0', 'C1', 'C2', 'C3', 'C4']], _pcov = curve_fit(poly4, xdata, ydata)
    df_K['D3'][['C0', 'C1', 'C2', 'C3']], _pcov = curve_fit(poly3, xdata, ydata)
    df_K['D2'][['C0', 'C1', 'C2']], _pcov = curve_fit(poly2, xdata, ydata)
    df_K['D1'][['C0', 'C1']], _pcov = curve_fit(poly1, xdata, ydata)

    #Celsius
    df_C = pd.DataFrame(np.zeros((5, 4)), 
    index = ['C0', 'C1', 'C2', 'C3', 'C4'], 
    columns=['D4', 'D3', 'D2', 'D1'],
    dtype=float)

    xdata = df_aT['T'].values
    ydata = df_aT['log_aT'].values

    df_C['D4'][['C0', 'C1', 'C2', 'C3', 'C4']], _pcov = curve_fit(poly4, xdata, ydata)
    df_C['D3'][['C0', 'C1', 'C2', 'C3']], _pcov = curve_fit(poly3, xdata, ydata)
    df_C['D2'][['C0', 'C1', 'C2']], _pcov = curve_fit(poly2, xdata, ydata)
    df_C['D1'][['C0', 'C1']], _pcov = curve_fit(poly1, xdata, ydata)

    #Prep dataframes and add units for output
    df_C = df_C.T
    df_K = df_K.T

    coeff_C = df_C.columns.get_level_values(0)
    C = ['°C', '-', '°C^-1', '°C^-2', '°C^-3']
    df_C.columns = pd.MultiIndex.from_tuples(zip(coeff_C, C), names = ['Degree', '-'])

    coeff_K = df_K.columns.get_level_values(0)
    K = ['K', '-', 'K^-1', 'K^-2', 'K^-3']
    df_K.columns = pd.MultiIndex.from_tuples(zip(coeff_K, K), names = ['Degree', '-'])
    
    return df_C, df_K


def plot(df_aT, df_WLF, df_C):
    """
    Plot shift factors and shift functions.
    
    Parameters
    ----------
    df_aT : pandas.DataFrame
        Contains the decadic logarithm of the shift factors 'log_aT'
        and the corresponding temperature values 'T' in degree Celsius.

    df_WLF : pandas.DataFrame
        Contains the necessary parameters to calculate the WLF
        equation (RefT, WLF_C1, WLF_C2) in degree Celsius.

    df_C : pandas.DataFrame
        Contains the coefficients to calculate the polynomial 
        shift functions of degree 1 to 4 for temperatures in degree **Celsius**.
    
    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure instance.

    df_shift: pandasDataFrame
        Contains the data used to create the plot.
    """
    x = df_aT['T'].values
    y_aT = df_aT['log_aT'].values

    y_WLF = WLF(df_aT['T'], df_WLF['RefT'].values, df_WLF['C1'].values, df_WLF['C2'].values)

    df_p = df_C.copy()
    df_p.columns = df_C.columns.droplevel(1)
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