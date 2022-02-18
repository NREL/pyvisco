"""
Collection of function to compare and verify the Python implementation 
within this module with the curve fitting routine of Ansys APDL 2021 R1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_prony_ANSYS(filepath):
    """
    Load Prony series parameters from ANSYS material card file.

    Parameters
    ----------
    filepath : str
        Location and file name of ANSYS material card file (*.MPL)

    Returns
    -------
    df_prony : pandas.DataFrame
        Contains the Prony series parameter.
    """
    alpha_i = []
    tau_i = []
    mod = True
    with open(filepath) as f:
        for line in f:
            lsplit = line.rstrip('\n').split(',')
            if lsplit[0] == 'TB':
                nterms = int(lsplit[4])
            if lsplit[0] == 'TBDATA':
                if len(lsplit) == 5:
                    if mod:
                        alpha_i.append(lsplit[2])
                        mod = False
                    else:
                        tau_i.append(lsplit[2])
                        mod = True
                    if mod:
                        alpha_i.append(lsplit[3])
                        mod = False
                    else:
                        tau_i.append(lsplit[3])
                        mod = True
                    if mod:
                        alpha_i.append(lsplit[4])
                        mod = False
                    else:
                        tau_i.append(lsplit[4])
                        mod = True
                if len(lsplit) == 4:
                    if mod:
                        alpha_i.append(lsplit[2])
                        mod = False
                    else:
                        tau_i.append(lsplit[2])
                        mod = True
                    if mod:
                        alpha_i.append(lsplit[3])
                        mod = False
                    else:
                        tau_i.append(lsplit[3])
                        mod = True
                if len(lsplit) == 3:
                    if mod:
                        alpha_i.append(lsplit[2])
                        mod = False
                    else:
                        tau_i.append(lsplit[2])
                        mod = True
    #f.close()
    alpha_i = np.array(alpha_i, dtype=float)
    tau_i = np.array(tau_i, dtype=float)
    df_prony = pd.DataFrame([tau_i, alpha_i]).T
    df_prony.columns = ['tau_i', 'alpha_i']
    df_prony.index += 1 
    return df_prony


def prep_prony_ANSYS(df_prony, prony, E_0 = None):
    """
    Prepare ANSYS Prony series parameters for further processing.

    The ANSYS curve fitting routine for viscoelastic materials only stores
    the Prony series parameters ('tau_i', 'alpha_i') in the material card file.
    To calculate the master curve from the Prony series parameters the 
    instantenous modulus and frequency range are required and added to the
    dataframe of the ANSYS Prony series parameters.

    Parameters
    ----------
    df_prony : pandas.DataFrame
        Contains the ANSYS Prony series parameter.
    prony : dict
        Contains the Python Prony series parameter
    E_0 : float, default = None
        Instantaneous storage modulus; for either tensile (E_0) or shear (G_0) 
        loading. If E_0 is not provided the instantaneous storage modulus 
        identified during the Python curve fitting process will be used to 
        create the master curve with the ANSYS Prony series parameters.

    Returns
    -------
    prony_ANSYS : dict
        Contains the ANSYS Prony series parameter in the same format as the 
        Python implementation provides (see 'prony' Parameter above).
    """
    m = prony['modul']
    if E_0 == None:
        E_0 = prony['E_0'] #use same estimate as Python curve fitting
    f_min = prony['f_min'] #use same frequency range as Python curve fitting
    f_max = prony['f_max']
    prony_ANSYS = {'E_0': E_0, 'df_terms':df_prony, 'f_min':f_min, 
        'f_max':f_max, 'label':'ANSYS', 'modul' : m}
    return prony_ANSYS


def plot_fit_ANSYS(df_master, df_GMaxw, df_GMaxw_ANSYS, units):
    """
    Plot master curve, Prony series fit of Python implementation, and Prony 
    series fit of ANSYS curve fitting routine.

    Parameters
    ----------
    df_master : pandas.DataFrame
        Contains the master curve data.
    df_GMaxw : pandas.DataFrame
        Contains the calculated Generalized Maxwell model data for the Prony
        series obtained with the **Python** implementation.
    df_GMaxw_ANSYS : pandas.DataFrame
        Contains the calculated Generalized Maxwell model data for the Prony
        series obtained with the **ANSYS** implementation.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Plot of master curve and Prony fits.
    """
    modul = df_master.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)

    if df_master.domain == 'freq':
        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=[stor, loss], ax=ax1, logx=True, 
            color=['C0', 'C1'], ls='', marker='o', markersize=3, alpha=0.5,
            label=["{}'(Master curve)".format(modul), "{}''(Master curve)".format(modul)])
        df_GMaxw.plot(x='f', y=[stor, loss], ax=ax1, logx=True,  
            color=['C0', 'C1'], ls='-', lw=2,
            label=["{}'(Python)".format(modul), "{}''(Python)".format(modul)])
        df_GMaxw_ANSYS.plot(x='f', y=[stor, loss], ax=ax1, logx=True, 
            color=['C2', 'C3'], ls='-', lw=2, 
            label=["{}'(ANSYS)".format(modul), "{}''(ANSYS)".format(modul)])
        ax1.set_xlabel('Frequency ({})'.format(units['f']))
        ax1.set_ylabel('Storage and loss modulus ({})'.format(units[stor]))
        ax1.legend()
        fig.show()
        return fig
    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=[relax], label=['Master curve'], ax=ax1,
            logx=True, color=['C0'], alpha=0.5, ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='t', y=[relax], label=['Prony fit (Python)'], ax=ax1, 
            logx=True, ls='-', lw=2, color=['C0'])
        df_GMaxw_ANSYS.plot(x='t', y=[relax], label=['Prony fit (ANSYS)'], ax=ax1, 
            logx=True, ls='-', lw=2, color=['C2'])
        ax1.set_xlabel('Time ({})'.format(units['t']))
        ax1.set_ylabel('Relaxation modulus ({})'.format(units[relax]))
        ax1.legend()
        fig.show()
        return fig
