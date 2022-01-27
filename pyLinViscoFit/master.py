import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



def plot(df_master):
    """Simple function to plot the master curve."""
    if df_master.domain == 'freq':

        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=['E_stor', 'E_loss'], ax=ax1, logx=True)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage and loss modulus (MPa)') #TODO: Make sure it makes sense to include units here...

        fig.show()
        return fig

    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=['E_relax'], ax=ax1, logx=True)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...

        fig.show()
        return fig


"""----------------------------------------------------------------------------
Shift factors
"""

#Power law function shift
def log_func_pwr(x, a, b):
    return a*x**b



# #Linear function shift
# def log_func(x, k, d):
#     return k*np.log(x)+d


# def get_at(df, gb_ref, gb_shift, remove=False):

#     if remove:
#         low = 1
#         upp = -1
#     else:
#         low=0
#         upp=None

#     gb = df.groupby('Set')

#     refx = gb.get_group(gb_ref)["f_set"]
#     refy = gb.get_group(gb_ref)["E_stor"]

#     refpopt, ref_pcov = curve_fit(log_func, refx[low:upp], refy[low:upp])

#     shiftx = gb.get_group(gb_shift)["f_set"]
#     shifty = gb.get_group(gb_shift)["E_stor"]

#     shiftpopt, shift_pcov = curve_fit(log_func, shiftx[low:upp], shifty[low:upp])

#     k1 = refpopt[0]
#     d1 = refpopt[1]
#     k2 = shiftpopt[0]
#     d2 = shiftpopt[1]

#     xrefint = log_func(gb.get_group(gb_ref)["f_set"], *refpopt).min()
#     xshiftint = log_func(gb.get_group(gb_shift)["f_set"], *shiftpopt).max()

#     yint = (xrefint+xshiftint)/2
#     refxi = np.exp((yint-d1)/k1)
#     shiftxi = np.exp((yint-d2)/k2)
#     log_aT = np.log10(refxi/shiftxi)

#     return log_aT, refpopt, shiftpopt


# def get_shift_lin(df, RefT, remove=False):

#     gb = df.groupby('Set')

#     arrshift_fit = np.zeros((gb.ngroups,2))
#     popt = np.zeros((gb.ngroups,2))

#     for group, df_set in gb:
#         if RefT <= df_set["T_round"].iloc[0]+1 and RefT >= df_set["T_round"].iloc[0]-1:
#             gb_ref = group
#             arrshift_fit[int(group),0] = df_set["T_round"].iloc[0]

#     for i in range(int(gb_ref), 0, -1):
#         arrshift_fit[i-1,0] = gb.get_group(i-1)["T_round"].iloc[0]
#         arrshift_fit[i-1,1] = get_at(df, i, i-1, remove)[0] + arrshift_fit[i,1]

#     for i in range(int(gb_ref), gb.ngroups-1, 1):
#         arrshift_fit[i+1,0] = gb.get_group(i+1)["T_round"].iloc[0]
#         arrshift_fit[i+1,1] = get_at(df, i, i+1, remove)[0] + arrshift_fit[i,1]

#     df_aT = pd.DataFrame(arrshift_fit, columns=['Temp', 'aT']).sort_values(by=['Temp'], ascending=False).reset_index(drop=True)

#     return df_aT


def get_at_pwr(df, gb_ref, gb_shift, remove=False):

    if remove:
        low = 1
        upp = -1
    else:
        low=0
        upp=None

    if df.domain == 'freq':
        _domain = 'f_set'
        _modul = 'E_stor'
    elif df.domain == 'time':
        _domain = 'f_set'
        _modul = 'E_relax'

    gb = df.groupby('Set')

    refx = gb.get_group(gb_ref)[_domain]
    refy = gb.get_group(gb_ref)[_modul]

    refpopt, ref_pcov = curve_fit(log_func_pwr, refx[low:upp], refy[low:upp])

    shiftx = gb.get_group(gb_shift)[_domain]
    shifty = gb.get_group(gb_shift)[_modul]

    shiftpopt, shift_pcov = curve_fit(log_func_pwr, shiftx[low:upp], shifty[low:upp])

    a1 = refpopt[0]
    b1 = refpopt[1]
    a2 = shiftpopt[0]
    b2 = shiftpopt[1]

    xrefint = log_func_pwr(gb.get_group(gb_ref)[_domain], *refpopt).min()
    xshiftint = log_func_pwr(gb.get_group(gb_shift)[_domain], *shiftpopt).max()

    yint = (xrefint+xshiftint)/2
    refxi = (yint/a1)**(1/b1)
    shiftxi = (yint/a2)**(1/b2)
    log_aT = np.log10(refxi/shiftxi)

    return log_aT, refpopt, shiftpopt


def get_aT(df, RefT, remove=False):

    gb = df.groupby('Set')

    arrshift_fit = np.zeros((gb.ngroups,2))
    popt = np.zeros((gb.ngroups,2))

    for group, df_set in gb:
        if RefT <= df_set["T_round"].iloc[0]+1 and RefT >= df_set["T_round"].iloc[0]-1:
            gb_ref = group
            arrshift_fit[int(group),0] = df_set["T_round"].iloc[0]

    for i in range(int(gb_ref), 0, -1):
        arrshift_fit[i-1,0] = gb.get_group(i-1)["T_round"].iloc[0]
        arrshift_fit[i-1,1] = get_at_pwr(df, i, i-1, remove)[0] + arrshift_fit[i,1]

    for i in range(int(gb_ref), gb.ngroups-1, 1):
        arrshift_fit[i+1,0] = gb.get_group(i+1)["T_round"].iloc[0]
        arrshift_fit[i+1,1] = get_at_pwr(df, i, i+1, remove)[0] + arrshift_fit[i,1]

    df_aT = pd.DataFrame(arrshift_fit, columns=['Temp', 'aT']).sort_values(by=['Temp'], ascending=False).reset_index(drop=True)

    return df_aT


def get_curve(df_raw, df_aT, RefT):

    gb = df_raw.groupby('Set')
    _num = int(df_raw.shape[0]/gb.ngroups)

    _shift = np.array([])

    for index, rows in df_aT.iterrows():
        _shift = np.append(_shift, np.flip(df_raw['f_set'][index*_num:(index+1)*_num].values*10**(rows['aT'])))

    df_raw['f'] = np.flip(_shift)

    if df_raw.domain == 'freq':
        df_master = df_raw[["f", "E_stor", "E_loss", "Set"]].copy()
        if "E_comp" in df_raw:
            df_master['E_comp'] = df_raw['E_comp']
            df_master['tan_del'] = df_raw['tan_del']
        df_master = df_master.sort_values(by=['f']).reset_index(drop=True)
        df_master.RefT = RefT
        df_master.domain = df_raw.domain
        df_master['omega'] = 2*np.pi*df_master['f']
        df_master['t'] = 1/df_master['f'] 

        
    elif df_raw.domain == 'time':
        df_raw['t'] = 1/df_raw['f']

        df_master = df_raw[["t", "E_relax", "Set"]].copy()
        df_master = df_master.sort_values(by=['t']).reset_index(drop=True)
        df_master.RefT = RefT
        df_master.domain = df_raw.domain
        df_master['f'] = 1/df_master['t']
        df_master['omega'] = 2*np.pi*df_master['f'] 

        df_raw = df_raw.drop(['t'], axis=1)

    df_raw = df_raw.drop(['f'], axis=1)

    return df_master


def plot_shift(df_raw, df_master):

    if df_master.domain == 'freq':

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,0.75*4))
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Loss modulus (MPa)') #TODO: Make sure it makes sense to include units here...

        gb_raw = df_raw.groupby('Set')
        gb_master = df_master.groupby('Set')
        colors1 = np.flip(plt.cm.Blues(np.linspace(0,1,int(gb_raw.ngroups*1.25))),axis=0)
        colors2 = np.flip(plt.cm.Oranges(np.linspace(0,1,int(gb_raw.ngroups*1.25))),axis=0)

        for group, df_set in gb_master:
            ax1.semilogx(df_set["f"], df_set["E_stor"], ls='', marker='.', color=colors1[int(group)])
            ax2.semilogx(df_set["f"], df_set["E_loss"], ls='', marker='.', color=colors2[int(group)])
        for group, df_set in gb_raw:
            ax1.semilogx(df_set["f_set"], df_set["E_stor"], ls='', marker='.', color=colors1[int(group)])
            ax2.semilogx(df_set["f_set"], df_set["E_loss"], ls='', marker='.', color=colors2[int(group)])

        fig.show()
        return fig

    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        
        gb_raw = df_raw.groupby('Set')
        gb_master = df_master.groupby('Set')
        colors1 = np.flip(plt.cm.Blues(np.linspace(0,1,int(gb_raw.ngroups*1.25))),axis=0)

        for group, df_set in gb_master:
            ax1.semilogx(df_set["t"], df_set["E_relax"], ls='', marker='.', color=colors1[int(group)])
        for group, df_set in gb_raw:
            ax1.semilogx(df_set["t_set"], df_set["E_relax"], ls='', marker='.', color=colors1[int(group)])


        fig.show()
        return fig


def smooth(df_master, win):
    if df_master.domain == 'freq':
        df_master["E_stor_filt"] = df_master["E_stor"].rolling(win, center=True, min_periods=1).median()
        df_master["E_loss_filt"] = df_master["E_loss"].rolling(win, center=True, min_periods=1).median()

    elif df_master.domain == 'time':
        df_master["E_relax_filt"] = df_master["E_relax"].rolling(win, center=True, min_periods=1).median()

    return df_master


def plot_smooth(df_master):
    if df_master.domain == 'freq':

        fig, ax = plt.subplots()
        df_master.plot(x='f', y=['E_stor'], label=["E'(raw)"], 
            ax=ax, logx=True, color=['C0'], marker='o', ls='', alpha=0.5)
        df_master.plot(x='f', y=['E_stor_filt'], label=["E'(filt)"], 
            ax=ax, logx=True, color=['C0'])
        df_master.plot(x='f', y=['E_loss'], label=["E''(raw)"], 
            ax=ax, logx=True, color=['C1'], marker='o', ls='', alpha=0.5)
        df_master.plot(x='f', y=['E_loss_filt'], label=["E'(filt)"], 
            ax=ax, logx=True, color=['C1'])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Storage and loss modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax.legend()

        fig.show()
        return fig


    elif df_master.domain == 'time':
        
        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=['E_relax'], label = ['E_relax'], 
            ax=ax1, logx=True, ls='', marker='o', color=['gray'])
        df_master.plot(x='t', y=['E_relax_filt'], label=['filter'], 
            ax=ax1, logx=True, color=['r'])
        ax1.set_xlabel('Time (s)')                  #TODO: Make sure it makes sense to include units here...
        ax1.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig
