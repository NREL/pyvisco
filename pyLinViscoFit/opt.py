import pandas as pd
import matplotlib.pyplot as plt

from . import prony

#Find optimal number of Prony terms for FEM
#-----------------------------------------------------------------------------
def nprony(df_master, prony_series, window='min', err_opt = 0.025):
    dict_prony = {}
    nprony = prony_series['df_terms'].shape[0]
    
    for i in range(1, nprony-2):
        N = nprony - i
        if not (N>20 and N%2 == 1): #above 20 Prony terms only compute every 2nd series
            df_dis = prony.discretize(df_master, window, N)
            if df_master.domain == 'time':
                prony_series = prony.fit_time(df_dis, df_master, opt=True)
            elif df_master.domain == 'freq':
                prony_series = prony.fit_freq(df_dis, df_master, opt=True)

            dict_prony[N] = prony_series 
        
    err = pd.DataFrame()
    for key, item in dict_prony.items():
        err.at[key, 'res'] = item['err']
        
        N_opt = (err['res']-err_opt).abs().sort_values().index[0]
            
    return dict_prony, N_opt, err


def plot_fit(df_master, dict_prony, N):
    
    df_GMaxw = prony.calc_GMaxw(**dict_prony[N])
    fig = prony.plot_fit(df_master, df_GMaxw)

    return df_GMaxw, fig


def plot_residual(N_opt_err):

    fig, ax = plt.subplots()
    N_opt_err.plot(y=['res'], ax=ax, c='k', label=['Least squares residual'], 
        marker='o', ls='--', markersize=4, lw=1)
    ax.set_xlabel('Number of Prony terms')
    ax.set_ylabel(r'$R^2 = \sum \left[E_{meas} - E_{Prony} \right]^2$') 
    ax.set_xlim(0,)
    ax.set_ylim(-0.01,0.25)
    ax.legend()

    fig.show()
    return fig