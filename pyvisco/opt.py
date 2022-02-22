"""
Collection of functions to minimize the number of Prony series terms used
in the Generalized Maxwell model.
"""

import pandas as pd
import matplotlib.pyplot as plt

from . import prony

#Find optimal number of Prony terms for FEM
#-----------------------------------------------------------------------------
def nprony(df_master, prony_series, window='min', opt = 1.5):
    """
    Minimize number of Prony terms used in Generalized Maxwell model.

    The number of Prony terms is gradually decreased and the new Prony series
    parameters are identified. The goodness of fit is evaluated based on the R^2
    measure. An optimal number of Prony terms is suggested.

    Parameters
    ----------
    df_master : pandas.DataFrame
        Contains the master curve data.
    prony_series : dict
        Contains the Prony series parameters of the initial fit.
    window : {'min', 'round', 'exact'}
        Defines the location of the discretization of the relaxation times.
        - 'exact' : use whole window of the experimental data and logarithmically 
        space the relaxation times between
        - 'round' : round the minimum and maximum values of the experimental data
        to the nearest base 10 number and logarithmically space the 
        remaining relaxation times between the rounded numbers
        - 'min'   : Position of relaxation times is optimized during minimization
        routine to reduce the number of Prony terms.
    opt : numeric
        Multiplier for the inital least squares residual to suggest an optimal 
        number of Prony terms: (R_opt)^2 = opt * (R_0)^2

    Return
    ------
    dict_prony : dict{N : prony_series}
        Contains all prony_series parameters for each number of
        calculated Prony terms, N.
    N_opt : int
        Optimal number of Prony terms.
    err : pandas.DataFrame
        Contains the least sqare residuals for each calculated Prony series.
    """
    dict_prony = {}
    nprony = prony_series['df_terms'].shape[0]
    for i in range(1, nprony-2):
        N = nprony - i
        if not (N>20 and N%2 == 1): 
            # Above 20 Prony terms only compute every 2nd series 
            # to reduce computational time
            df_dis = prony.discretize(df_master, window, N)
            if df_master.domain == 'time':
                prony_series = prony.fit_time(df_dis, df_master, opt=True)
            elif df_master.domain == 'freq':
                prony_series = prony.fit_freq(df_dis, df_master, opt=True)
            dict_prony[N] = prony_series 
        
    # Get least square residuals
    err = pd.DataFrame()
    for key, item in dict_prony.items():
        err.at[key, 'res'] = item['err']
    err.modul = df_master.modul
        
    # Find optimal number of Prony terms
    err_opt = opt*err['res'].min()
    N_opt = (err['res']-err_opt).abs().sort_values().index[0]
    return dict_prony, N_opt, err


def plot_fit(df_master, dict_prony, N, units):
    """
    Calculate and plot the optimized Prony series fit.

    Parameters
    ----------
    df_master : pandas.DataFrame
        Contains the master curve data.
    dict_prony : dict{N : prony_series}
        Contains all Prony series parameters for each number of
        calculated Prony terms, N.
    N : int
        Number of Prony terms for plot.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    Returns
    -------
    df_GMaxw : pandas.DataFrame
        Contains the calculated Generalized Maxwell model data for the Prony
        series with N terms.
    fig : matplotlib.pyplot.figure
        Plot of optimized Prony series fit.
    """
    df_GMaxw = prony.calc_GMaxw(**dict_prony[N])
    fig = prony.plot_fit(df_master, df_GMaxw, units)
    return df_GMaxw, fig


def plot_residual(err):
    """
    Plot the least squares residual of the Prony series fits.

    Parameters
    ----------
    err : pandas.DataFrame
        Contains the least sqare residuals for each calculated Prony series.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Matplotlib figure instance.

    See also
    --------
    opt.nprony : Return the err dataframe.
    """
    m = err.modul
    fig, ax = plt.subplots()
    err.plot(y=['res'], ax=ax, c='k', label=['Least squares residual'], 
        marker='o', ls='--', markersize=4, lw=1)
    ax.set_xlabel('Number of Prony terms')
    ax.set_ylabel(r'$R^2 = \sum \left[{{{0}}}_{{meas}} - {{{0}}}_{{Prony}} \right]^2$'.format(m)) 
    ax.set_xlim(0,)
    ax.set_ylim(-0.01, max(2*err['res'].min(), 0.25))
    ax.legend()
    fig.show()
    return fig