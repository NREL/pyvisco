import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



def plot(df_master, units):
    """Simple function to plot the master curve."""
    m = df_master.modul
    if df_master.domain == 'freq':
        
        fig, ax1 = plt.subplots()
        df_master.plot(x='f', 
            y=['{}_stor'.format(m), '{}_loss'.format(m)], ax=ax1, logx=True)
        ax1.set_xlabel('Frequency ({})'.format(units['f']))
        ax1.set_ylabel('Storage and loss modulus ({})'.format(units['{}_stor'.format(m)])) 

        fig.show()
        return fig

    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=['{}_relax'.format(m)], ax=ax1, logx=True)
        ax1.set_xlabel('Time ({})'.format(units['t']))
        ax1.set_ylabel('Relaxation modulus ({})'.format(units['{}_relax'.format(m)]))

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
    m = df_raw.modul
    if df_raw.domain == 'freq':
        _modul = '{}_stor'.format(m)
    elif df_raw.domain == 'time':
        _modul = '{}_relax'.format(m)

    gb = df_raw.groupby('Set')

    ref_xdata   = gb.get_group(gb_ref)['f_set'].values
    ref_ydata   = gb.get_group(gb_ref)[_modul].values
    shift_xdata = gb.get_group(gb_shift)['f_set'].values
    shift_ydata = gb.get_group(gb_shift)[_modul].values

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
    df_aT = pd.DataFrame(Temp, columns=['T'])
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
    m = df_raw.modul
    df_shift = pd.DataFrame() 
    for S, df in df_raw.groupby('Set'):  
        aT = 10**(df_aT[df_aT['T'] == df['T_round'].iloc[0]]['log_aT'].values)
        fshift = aT * df['f_set']
        df_shift = df_shift.append(fshift.to_frame())

    if df_raw.domain == 'freq':
        df_master = df_raw[[
            "{}_stor".format(m), 
            "{}_loss".format(m), 
            "Set"]].copy()
        df_master['f'] = df_shift
        if "{}_comp".format(m) in df_raw:
            df_master['{}_comp'.format(m)] = df_raw['{}_comp'.format(m)]
            df_master['tan_del'] = df_raw['tan_del']
        df_master = df_master.sort_values(by=['f']).reset_index(drop=True)
        df_master.RefT = RefT
        df_master.domain = df_raw.domain
        df_master.modul = m
        df_master['omega'] = 2*np.pi*df_master['f']
        df_master['t'] = 1/df_master['f'] 

    elif df_raw.domain == 'time':
        df_master = df_raw[["{}_relax".format(m), "Set"]].copy()
        df_master['t'] = 1/df_shift
        df_master = df_master.sort_values(by=['t']).reset_index(drop=True)
        df_master.RefT = RefT
        df_master.domain = df_raw.domain
        df_master.modul = m
        df_master['f'] = 1/df_master['t']
        df_master['omega'] = 2*np.pi*df_master['f'] 

    return df_master


def plot_shift(df_raw, df_master, units):
    m = df_raw.modul
    if df_master.domain == 'freq':

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,0.75*4))
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage modulus ({})'.format(units["{}_stor".format(m)]))
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Loss modulus ({})'.format(units["{}_stor".format(m)])) 

        gb_raw = df_raw.groupby('Set')
        gb_master = df_master.groupby('Set')
        colors1 = np.flip(plt.cm.Blues(np.linspace(0,1,int(gb_raw.ngroups*1.25))),axis=0)
        colors2 = np.flip(plt.cm.Oranges(np.linspace(0,1,int(gb_raw.ngroups*1.25))),axis=0)

        lax1 = []
        lax2 = []
        for i, (group, df_set) in enumerate(gb_master):
            line1, = ax1.semilogx(df_set["f"], df_set["{}_stor".format(m)], ls='', marker='.', 
                color=colors1[int(group)])
            line2, = ax2.semilogx(df_set["f"], df_set["{}_loss".format(m)], ls='', marker='.', 
                color=colors2[int(group)])
            lax1.append(line1)
            lax2.append(line2)
        for i, (group, df_set) in enumerate(gb_raw):
            if i in np.linspace(0, gb_raw.ngroups-1, num=5, dtype=int):
                ax1.semilogx(df_set["f_set"], df_set["{}_stor".format(m)], ls='', marker='.', 
                color=colors1[int(group)], label = '$T$={}\N{DEGREE SIGN}C'.format(df_set['T_round'].iloc[0]))
                ax2.semilogx(df_set["f_set"], df_set["{}_loss".format(m)], ls='', marker='.', 
                color=colors2[int(group)], label = '$T$={}\N{DEGREE SIGN}C'.format(df_set['T_round'].iloc[0]))
            else:
                ax1.semilogx(df_set["f_set"], df_set["{}_stor".format(m)], ls='', marker='.', 
                color=colors1[int(group)])
                ax2.semilogx(df_set["f_set"], df_set["{}_loss".format(m)], ls='', marker='.', 
                color=colors2[int(group)])

        legend1 = ax1.legend(handlelength=1, handletextpad=0.1, fontsize=8)
        legend2 = ax2.legend(handlelength=1, handletextpad=0.1, fontsize=8)
        for legend_handle in legend1.legendHandles:
            legend_handle._legmarker.set_markersize(8)
        for legend_handle in legend2.legendHandles:
            legend_handle._legmarker.set_markersize(8)


        fig.show()
        return fig, (ax1, lax1, ax2, lax2)

    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Relaxation modulus ({})'.format(units["{}_relax".format(m)])) 
        
        gb_raw = df_raw.groupby('Set')
        gb_master = df_master.groupby('Set')
        colors1 = np.flip(plt.cm.Blues(np.linspace(0,1,int(gb_raw.ngroups*1.25))),axis=0)

        lax1 = []
        for i, (group, df_set) in enumerate(gb_master):
            line1, = ax1.semilogx(df_set["t"], df_set["{}_relax".format(m)], ls='', marker='.', 
                color=colors1[int(group)])
            lax1.append(line1)
        for i, (group, df_set) in enumerate(gb_raw):
            if i in np.linspace(0, gb_raw.ngroups-1, num=5, dtype=int):
                ax1.semilogx(df_set["t_set"], df_set["{}_relax".format(m)], ls='', marker='.', 
                color=colors1[int(group)], label = '$T$={}\N{DEGREE SIGN}C'.format(df_set['T_round'].iloc[0]))
            else:
                ax1.semilogx(df_set["t_set"], df_set["{}_relax".format(m)], ls='', marker='.', 
                color=colors1[int(group)])

        legend = ax1.legend(handlelength=1, handletextpad=0.1, fontsize=8)
        for legend_handle in legend.legendHandles:
            legend_handle._legmarker.set_markersize(8)

        fig.show()
        return fig, (ax1, lax1)


def plot_shift_update(df_master, fig, lax):
    m = df_master.modul

    gb_master = df_master.groupby('Set')
    if len(lax) == 2:
        ax1, lax1 = lax
    elif len(lax) == 4:
        ax1, lax1, ax2, lax2 = lax

    for i, (group, df_set) in enumerate(gb_master):
        line1 = lax1[i]
        line1.set_xdata(df_set["f"])
        line1.set_ydata(df_set["{}_stor".format(m)])

        if len(lax) == 4:
            line2 = lax2[i]
            line2.set_xdata(df_set["f"])
            line2.set_ydata(df_set["{}_loss".format(m)])
        
    ax1.relim()
    ax1.autoscale_view()
    if len(lax) == 4:
        ax2.relim()
        ax2.autoscale_view()

    fig.canvas.draw_idle()

    #fig.canvas.draw()
    fig.canvas.flush_events()

    return fig



def smooth(df_master, win):
    m = df_master.modul
    if df_master.domain == 'freq':
        df_master["{}_stor_filt".format(m)] = df_master[
            "{}_stor".format(m)].rolling(win, center=True, min_periods=1).median()
        df_master["{}_loss_filt".format(m)] = df_master[
            "{}_loss".format(m)].rolling(win, center=True, min_periods=1).median()

    elif df_master.domain == 'time':
        df_master["{}_relax_filt".format(m)] = df_master[
            "{}_relax".format(m)].rolling(win, center=True, min_periods=1).median()

    return df_master

def plot_smooth(df_master, units):
    m = df_master.modul
    if df_master.domain == 'freq':
        fig, ax = plt.subplots()
        df_master.plot(x='f', y=['{}_stor'.format(m)], label=["{}'(raw)".format(m)], 
            ax=ax, logx=True, color=['C0'], marker='o', ls='', alpha=0.5)
        df_master.plot(x='f', y=['{}_stor_filt'.format(m)], label=["{}'(filt)".format(m)], 
            ax=ax, logx=True, color=['C0'])
        df_master.plot(x='f', y=['{}_loss'.format(m)], label=["{}''(raw)".format(m)], 
            ax=ax, logx=True, color=['C1'], marker='o', ls='', alpha=0.5)
        df_master.plot(x='f', y=['{}_loss_filt'.format(m)], label=["{}'(filt)".format(m)], 
            ax=ax, logx=True, color=['C1'])
        ax.set_xlabel('Frequency ({})'.format(units['f']))
        ax.set_ylabel('Storage and loss modulus ({})'.format(units['{}_stor'.format(m)])) 
        ax.legend()

        fig.show()
        return fig

    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=['{}_relax'.format(m)], label = ['{}_relax'.format(m)], 
            ax=ax1, logx=True, ls='', marker='o', color=['gray'])
        df_master.plot(x='t', y=['{}_relax_filt'.format(m)], label=['filter'], 
            ax=ax1, logx=True, color=['r'])
        ax1.set_xlabel('Time ({})'.format(units['t']))
        ax1.set_ylabel('Relaxation modulus ({})'.format(units['{}_relax'.format(m)])) 
        ax1.legend()

        fig.show()
        return fig
