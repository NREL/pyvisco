"""Collection of function to compare and verify the Python implementation 
within this module with the curve fitting routine of Ansys APDL 2021 R1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Verification - Compare with ANSYS material fitting routine
#-----------------------------------------------------------------------------
def load_prony_ANSYS(path):

    alpha_i = []
    tau_i = []

    mod = True

    with open(path) as f:
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

    f.close()

    alpha_i = np.array(alpha_i, dtype=float)
    tau_i = np.array(tau_i, dtype=float)

    df_prony = pd.DataFrame([tau_i, alpha_i]).T
    df_prony.columns = ['tau', 'alpha']
    df_prony.index += 1 

    return df_prony


def prep_prony_ANSYS(df_prony_ANSYS, prony, E_0 = None):
    if E_0 == None:
        E_0 = prony['E_0'] #use same estimate as for GUSTL fit
    f_min = prony['f_min'] #use same frequency range as for fit
    f_max = prony['f_max']

    prony_ANSYS = {'E_0':E_0, 'df_terms':df_prony_ANSYS, 'f_min':f_min, 
        'f_max':f_max, 'label':'ANSYS'}

    return prony_ANSYS


def plot_fit_ANSYS(df_master, df_GMaxw, df_GMaxw_ANSYS):

    if df_master.domain == 'freq':

        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=['E_stor', 'E_loss'], label=["E'(exp.)", "E''(exp.)"], 
            ax=ax1, logx=True, color=['C0', 'C1'], alpha=0.5, ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='f', y=['E_stor', 'E_loss'], label=["E'(Python)", "E''(Python)"], 
            ax=ax1, logx=True, ls='-', lw=2, color=['C0', 'C1'])
        df_GMaxw_ANSYS.plot(x='f', y=['E_stor', 'E_loss'], label=["E'(ANSYS)", "E''(ANSYS)"], 
            ax=ax1, logx=True, ls='-', lw=2, color=['C2', 'C3'])

        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Moduli (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig

    elif df_master.domain == 'time':

        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=['E_relax'], label=['E(exp.)'], 
            ax=ax1, logx=True, color=['C0'], alpha=0.5, ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='t', y=['E_relax'], label=['E(Python)'], 
            ax=ax1, logx=True, ls='-', lw=2, color=['C0'])
        df_GMaxw_ANSYS.plot(x='t', y=['E_relax'], label=['E(ANSYS)'], 
            ax=ax1, logx=True, ls='-', lw=2, color=['C2'])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig

