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
def pwr_y(x, a, b, e):
    return a*x**b+e

def pwr_x(y, a, b, e):
    return ((y-e)/a)**(1/b)

def fit_at_pwr(df_raw, gb_ref, gb_shift):
    if df_raw.domain == 'freq':
        _modul = 'E_stor'
    elif df_raw.domain == 'time':
        _modul = 'E_relax'

    gb = df_raw.groupby('Set')

    ref_xdata   = gb.get_group(gb_ref)['f_set']
    ref_ydata   = gb.get_group(gb_ref)[_modul]
    shift_xdata = gb.get_group(gb_shift)['f_set']
    shift_ydata = gb.get_group(gb_shift)[_modul]

    #Curve fit power law
    ref_popt, ref_pcov = curve_fit(pwr_y, ref_xdata, ref_ydata, maxfev=10000)
    shift_popt, shift_pcov = curve_fit(pwr_y, shift_xdata, shift_ydata, maxfev=10000)

    #Check and remove first measurement point if outlier
    ref_popt_rem, ref_pcov_rem = curve_fit(pwr_y, ref_xdata[1:], ref_ydata[1:], maxfev=10000)
    perr = np.sqrt(np.abs(np.diag(ref_pcov)))
    perr_rem = np.sqrt(np.abs(np.diag(ref_pcov_rem)))
    if all(perr_rem < perr):
        ref_popt = ref_popt_rem
        ref_xdata = ref_xdata[1:] 
        ref_ydata = ref_ydata[1:]

    shift_popt_rem, shift_pcov_rem = curve_fit(pwr_y, shift_xdata[1:], shift_ydata[1:], maxfev=10000)
    perr = np.sqrt(np.abs(np.diag(shift_pcov)))
    perr_rem = np.sqrt(np.abs(np.diag(shift_pcov_rem)))
    if all(perr_rem < perr):
        shift_popt = shift_popt_rem
        shift_xdata = shift_xdata[1:] 
        shift_ydata = shift_ydata[1:]

    #Calculate fit
    ref_ydata_fit = pwr_y(ref_xdata, *ref_popt)
    shift_ydata_fit = pwr_y(shift_xdata, *shift_popt)

    #Get interpolation or extrapolation range
    if ref_ydata_fit.max() > shift_ydata_fit.max():
        #Ref is on top
        top_xdata = ref_xdata
        top_ydata = ref_ydata
        top_popt = ref_popt
        bot_xdata = shift_xdata
        bot_ydata = shift_ydata
        bot_popt = shift_popt
        sign = 1
    else:
        #Shift is on top
        top_xdata = shift_xdata
        top_ydata = shift_ydata   
        top_popt = shift_popt
        bot_xdata = ref_xdata
        bot_ydata = ref_ydata
        bot_popt = ref_popt
        sign = -1
        
    if top_ydata.min() < bot_ydata.max():
        #overlap range
        ymin = top_ydata.min()
        ymax = bot_ydata.max()
    else:
        #gap range
        ymin = bot_ydata.max()
        ymax = top_ydata.min()   
        
    #Define three points along inter/extrapolation range
    ymid = (ymin+ymax)/2
    y = np.array([ymin, ymid, ymax])

    #Compute average shift factor for the three points
    x_top = pwr_x(y, *top_popt)
    x_bot = pwr_x(y, *bot_popt)

    log_aT = sign * np.log10(x_top/x_bot).mean()

    return log_aT


def get_aT(df_raw, RefT):
    #Create df_aT
    Temp = []
    for i, df_set in df_raw.groupby('Set'):
        T = df_set['T_round'].iloc[0]
        Temp.append(T)
        if T == RefT:
            idx = i
    df_aT = pd.DataFrame(Temp, columns=['Temp'])
    df_aT['log_aT'] = np.nan 

    #Set shift factor at RefT
    df_aT.loc[idx]['log_aT'] = 0

    #Shift below RefT
    for i in range(idx, 0, -1):
        df_aT.loc[i-1]['log_aT'] = fit_at_pwr(df_raw, i, i-1) + df_aT.loc[i]['log_aT']
        
    #Shift above RefT
    for i in range(idx, df_aT.shape[0]-1, 1):
        df_aT.loc[i+1]['log_aT'] = fit_at_pwr(df_raw, i, i+1) + df_aT.loc[i]['log_aT']

    return df_aT


def get_curve(df_raw, df_aT, RefT):
    df_shift = pd.DataFrame() 
    for S, df in df_raw.groupby('Set'):  
        aT = 10**(df_aT[df_aT['Temp'] == df['T_round'].iloc[0]]['log_aT'].values)
        fshift = aT * df['f_set']
        df_shift = df_shift.append(fshift.to_frame())

    if df_raw.domain == 'freq':
        df_master = df_raw[["E_stor", "E_loss", "Set"]].copy()
        df_master['f'] = df_shift
        if "E_comp" in df_raw:
            df_master['E_comp'] = df_raw['E_comp']
            df_master['tan_del'] = df_raw['tan_del']
        df_master = df_master.sort_values(by=['f']).reset_index(drop=True)
        df_master.RefT = RefT
        df_master.domain = df_raw.domain
        df_master['omega'] = 2*np.pi*df_master['f']
        df_master['t'] = 1/df_master['f'] 

    elif df_raw.domain == 'time':
        df_master = df_raw[["E_relax", "Set"]].copy()
        df_master['t'] = 1/df_shift
        df_master = df_master.sort_values(by=['t']).reset_index(drop=True)
        df_master.RefT = RefT
        df_master.domain = df_raw.domain
        df_master['f'] = 1/df_master['t']
        df_master['omega'] = 2*np.pi*df_master['f'] 

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
