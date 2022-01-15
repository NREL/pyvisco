import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import os
import io
#import seaborn as sns
import zipfile
import ipywidgets as widgets

#import matplotlib as mpl
#import sys
#import subprocess
#from shutil import copyfile
#from scipy.signal import savgol_filter
#from scipy.signal import medfilt

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import nnls
from ipympl.backend_nbagg import Canvas
from base64 import b64encode
from IPython.display import display, clear_output, HTML

def load_file(path):
    with open(path, 'rb') as file:  
        data = file.read() 
    return data

def load_Eplexor_raw(data):
    df = pd.read_excel(io.BytesIO(data), 'Exported Data', header=[0,1])
    df.columns = df.columns.droplevel(1)
    df.rename(columns={"f":"f_set", "E'":'E_stor', "E''":'E_loss', "|E*|":'E_comp',
        "tan delta":'tan_del'}, inplace=True, errors='raise')


    df_raw = df[['f_set', 'E_stor', 'E_loss', 'E_comp', 'tan_del', 'T']].copy()
    #df_raw['omega'] = 2*np.pi*df_raw['f']
    #df_raw['t'] = 1/df_raw['omega']

    df_raw = get_sets(df_raw, num=0)
    arr_RefT = df_raw.groupby('Set')['T'].mean().round()
    df_raw = df_raw.join(arr_RefT, on='Set', rsuffix='_round')
    df_raw.domain = 'freq'

    return df_raw, arr_RefT


def load_user_raw(data, domain):
    df_raw = pd.read_csv(io.BytesIO(data))

    if domain == 'freq':
        df_raw.rename(columns={"f":"f_set", "E_stor":'E_stor', 
            "E_loss":'E_loss', "T":"T", 'Set':'Set'}, inplace=True, errors='raise')
    elif domain == 'time':
        df_raw.rename(columns={"t":"t_set", "E_relax":'E_relax', 
            "T":"T", 'Set':'Set'}, inplace=True, errors='raise')
        df_raw['f_set'] = 1/df_raw['t_set']

    arr_RefT = df_raw.groupby('Set')['T'].mean().round()
    df_raw = df_raw.join(arr_RefT, on='Set', rsuffix='_round')
    df_raw.domain = domain
    df_raw.domain = domain
    
    return df_raw, arr_RefT

def load_Eplexor_master(data):
    df = pd.read_excel(io.BytesIO(data), 'Shiftlist',header=[0,1,2])
    df_master_raw = pd.read_excel(io.BytesIO(data), 'Exported Data', header=[0,1])

    #Prep Shift factors into df
    RefT = float(df.columns.values[1][0][:-3])
    C1 = float(df.columns.values[1][1][5:])
    C2 = float(df.columns.values[2][1][5:-2])
    WLF = [RefT, C1, C2]

    df.columns = ['Temp', 'aT', 'DEL']
    df.drop(['DEL'], axis = 1, inplace = True)
    df_aT = df.round({'Temp': 0})

    #Put fitted WLF shift function in df
    df_WLF = pd.DataFrame(data = WLF).T #
    df_WLF.columns = ['RefT', 'C1', 'C2']

    #Prep Master curve data into df
    df_master_raw.columns = df_master_raw.columns.droplevel(1)
    df_master_raw.rename(columns={"E'":'E_stor', "E''":'E_loss', "|E*|":'E_comp',
        "tan delta":'tan_del'}, inplace=True, errors='raise')

    df_master = df_master_raw[['f', 'E_stor', 'E_loss', 'E_comp', 'tan_del']].copy()
    df_master['omega'] = 2*np.pi*df_master['f']
    df_master['t'] = 1/df_master['f']  
    df_master.RefT = RefT
    df_master.domain = 'freq'

    return df_master, df_aT, df_WLF



def load_user_master(data, domain, RefT):
    df_master = pd.read_csv(io.BytesIO(data))

    if domain == 'freq':
        df_master.rename(columns = {'f':'f', 'E_stor':'E_stor', 
            'E_loss':'E_loss'}, inplace=True, errors='raise')
        df_master['omega'] = 2*np.pi*df_master['f']
        df_master['t'] = 1/df_master['f'] 
    elif domain == 'time':
        df_master.rename(columns = {'t':'t', 'E_relax':'E_relax'}, 
            inplace=True, errors='raise')
        df_master['f'] = 1/df_master['t']
        df_master['omega'] = 2*np.pi*df_master['f']

    df_master.RefT = RefT
    df_master.domain = domain

    return df_master

def load_user_shift(data_shift):
    df_aT = pd.read_csv(io.BytesIO(data_shift))
    df_aT.rename(columns = {'Temp':'Temp', 'aT':'aT'}, inplace=True, errors='raise')

    return df_aT



#Shift factors
#-----------------------------------------------------------------------------
def get_sets(df_raw, num=0):
    """Group raw DMTA data into measurements conducted at same temperature.
    num: Number of measurements at one temperature level
    If num is not provided the number of measurements is evaluated based on 
    the frequency range and the index of the first occurance of the max. frequency
    is used to group the data frame.
    """
    iset = -1
    lset = []
    if num == 0: #Identify measurement sets based on frequency range
        num = df_raw['f_set'].idxmax()+1
            
    for i in range(df_raw.shape[0]):
        if i%num == 0:
            iset += 1
        lset.append(iset)
        
    df_raw['Set'] = lset
    
    return df_raw

#Power law function shift
def log_func_pwr(x, a, b):
    return a*x**b

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

    refpopt, refpcov = curve_fit(log_func_pwr, refx[low:upp], refy[low:upp])

    shiftx = gb.get_group(gb_shift)[_domain]
    shifty = gb.get_group(gb_shift)[_modul]

    shiftpopt, shiftpcov = curve_fit(log_func_pwr, shiftx[low:upp], shifty[low:upp])

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

def get_shift_pwr(df, RefT, remove=False):

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


#Linear function shift
def log_func(x, k, d):
    return k*np.log(x)+d

def get_at(df, gb_ref, gb_shift, remove=False):

    if remove:
        low = 1
        upp = -1
    else:
        low=0
        upp=None

    gb = df.groupby('Set')

    refx = gb.get_group(gb_ref)["f_set"]
    refy = gb.get_group(gb_ref)["E_stor"]

    refpopt, refpcov = curve_fit(log_func, refx[low:upp], refy[low:upp])

    shiftx = gb.get_group(gb_shift)["f_set"]
    shifty = gb.get_group(gb_shift)["E_stor"]

    shiftpopt, shiftpcov = curve_fit(log_func, shiftx[low:upp], shifty[low:upp])

    k1 = refpopt[0]
    d1 = refpopt[1]
    k2 = shiftpopt[0]
    d2 = shiftpopt[1]

    xrefint = log_func(gb.get_group(gb_ref)["f_set"], *refpopt).min()
    xshiftint = log_func(gb.get_group(gb_shift)["f_set"], *shiftpopt).max()

    yint = (xrefint+xshiftint)/2
    refxi = np.exp((yint-d1)/k1)
    shiftxi = np.exp((yint-d2)/k2)
    log_aT = np.log10(refxi/shiftxi)

    return log_aT, refpopt, shiftpopt

def get_shift_lin(df, RefT, remove=False):

    gb = df.groupby('Set')

    arrshift_fit = np.zeros((gb.ngroups,2))
    popt = np.zeros((gb.ngroups,2))

    for group, df_set in gb:
        if RefT <= df_set["T_round"].iloc[0]+1 and RefT >= df_set["T_round"].iloc[0]-1:
            gb_ref = group
            arrshift_fit[int(group),0] = df_set["T_round"].iloc[0]

    for i in range(int(gb_ref), 0, -1):
        arrshift_fit[i-1,0] = gb.get_group(i-1)["T_round"].iloc[0]
        arrshift_fit[i-1,1] = get_at(df, i, i-1, remove)[0] + arrshift_fit[i,1]

    for i in range(int(gb_ref), gb.ngroups-1, 1):
        arrshift_fit[i+1,0] = gb.get_group(i+1)["T_round"].iloc[0]
        arrshift_fit[i+1,1] = get_at(df, i, i+1, remove)[0] + arrshift_fit[i,1]

    df_aT = pd.DataFrame(arrshift_fit, columns=['Temp', 'aT']).sort_values(by=['Temp'], ascending=False).reset_index(drop=True)

    return df_aT

#Shift master
def df_shift_master(df_raw, df_aT, RefT):

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


def plot_master_shift(df_raw, df_master):

    if df_master.domain == 'freq':

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,0.75*4.5))
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




#Shift function
#-----------------------------------------------------------------------------

def WLF(Temp, RefT, WLF_C1, WLF_C2):
    return -WLF_C1*(Temp - RefT)/(WLF_C2+(Temp - RefT))

def fit_shift_WLF(RefT, df_aT):

    xdata = df_aT['Temp'].values+273.15
    ydata = df_aT['aT'].values

    popt, pcov = curve_fit(lambda x, C1, C2: WLF(x, RefT+273.15, C1, C2), xdata, ydata, p0 = [1E3, 5E3], bounds=(0, 5000))

    #Put fitted WLF shift function in df
    df_WLF = pd.DataFrame(data = np.insert(popt,0,RefT)).T 
    df_WLF.columns = ['RefT', 'C1', 'C2']

    return df_WLF

def poly1(x,C0,C1):
    return C0+C1*x

def poly2(x,C0,C1,C2):
    return C0+C1*x+C2*x**2

def poly3(x,C0,C1,C2,C3):
    return C0+C1*x+C2*x**2+C3*x**3

def poly4(x,C0,C1,C2,C3,C4):
    return C0+C1*x+C2*x**2+C3*x**3+C4*x**4

def fit_shift_poly(df_aT):

    df = pd.DataFrame(np.zeros((5, 8)), 
        index = ['C0', 'C1', 'C2', 'C3', 'C4'], 
        columns=['P4 (K)', 'P3 (K)', 'P2 (K)', 'P1 (K)', 'P4 (C)', 'P3 (C)','P2 (C)','P1 (C)'],
        dtype=float)

    #Kelvin
    xdata = df_aT['Temp'].values+273.15
    ydata = df_aT['aT'].values

    df['P4 (K)'][['C0', 'C1', 'C2', 'C3', 'C4']], pcov = curve_fit(poly4, xdata, ydata)
    df['P3 (K)'][['C0', 'C1', 'C2', 'C3']], pcov = curve_fit(poly3, xdata, ydata)
    df['P2 (K)'][['C0', 'C1', 'C2']], pcov = curve_fit(poly2, xdata, ydata)
    df['P1 (K)'][['C0', 'C1']], pcov = curve_fit(poly1, xdata, ydata)
#
    ##Celsius
    xdata = df_aT['Temp'].values
#
    df['P4 (C)'][['C0', 'C1', 'C2', 'C3', 'C4']], pcov = curve_fit(poly4, xdata, ydata)
    df['P3 (C)'][['C0', 'C1', 'C2', 'C3']], pcov = curve_fit(poly3, xdata, ydata)
    df['P2 (C)'][['C0', 'C1', 'C2']], pcov = curve_fit(poly2, xdata, ydata)
    df['P1 (C)'][['C0', 'C1']], pcov = curve_fit(poly1, xdata, ydata)

    return df

def plot_shift_func(df_aT, df_WLF, df_poly):

    x = df_aT['Temp'].values
    y_aT = df_aT['aT'].values

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



#Master curve
#-----------------------------------------------------------------------------
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
        df_master.plot(x='f', y=['E_stor'], label=["E'(raw)"], ax=ax, logx=True, color=['C0'], marker='o', ls='', alpha=0.5)
        df_master.plot(x='f', y=['E_stor_filt'], label=["E'(filt)"], ax=ax, logx=True, color=['C0'])
        df_master.plot(x='f', y=['E_loss'], label=["E''(raw)"], ax=ax, logx=True, color=['C1'], marker='o', ls='', alpha=0.5)
        df_master.plot(x='f', y=['E_loss_filt'], label=["E'(filt)"], ax=ax, logx=True, color=['C1'])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Storage and loss modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax.legend()

        fig.show()
        return fig

   
    elif df_master.domain == 'time':
        
        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=['E_relax'], label = ['E_relax'], ax=ax1, logx=True, ls='', marker='o', color=['gray'])
        df_master.plot(x='t', y=['E_relax_filt'], label=['filter'], ax=ax1, logx=True, color=['r'])
        ax1.set_xlabel('Time (s)')                  #TODO: Make sure it makes sense to include units here...
        ax1.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig


def discretize(df_master, window='round', nprony=0):

    #Get relaxation times
    a = 1 #[Tschoegl 1989]
    #omega = (1/(a*tau)) #[Kraus 2017, Eq. 25]
       
    _tau = 1/(a*df_master['omega'])

    #Window Time Domain
    if df_master.domain == 'freq':
        exp_inf = int(np.floor(np.log10(_tau.iloc[0])))  #get highest time domain exponent
        exp_0 = int(np.ceil(np.log10(_tau.iloc[-1])))    #get lowest time domain exponent
        val_inf = _tau.iloc[0]
        val_0 = _tau.iloc[-1]

    elif df_master.domain == 'time':
        exp_inf = int(np.floor(np.log10(_tau.iloc[-1])))  #get highest time domain exponent
        exp_0 = int(np.ceil(np.log10(_tau.iloc[0])))    #get lowest time domain exponent
        val_inf = _tau.iloc[-1]
        val_0 = _tau.iloc[0]

    decades = exp_inf - exp_0
    
    #Space evenly on a log scale in time domain
    if nprony == 0:
        nprony = exp_inf - exp_0 + 1 #one prony term per decade 

    if window == 'round':
        tau = np.flip(np.geomspace(float(10**exp_0), float(10**exp_inf), nprony)) 

    elif window == 'exact':
        tau = np.flip(np.geomspace(val_0, val_inf, nprony)) 

    elif window == 'min':
        tau = np.flip(np.geomspace(val_0, val_inf, nprony+2))[1:-1]


    #Get dataframe with discretized values
    omega_dis = (1/(a*tau)) #[Kraus 2017, Eq. 25]
    freq_dis = omega_dis/(2*np.pi) #convert to cycles per second [Hz] 
    t_dis = 1/freq_dis


    if df_master.domain == 'freq':

        #Interpolate E_stor and E_loss at discretization poins
        E_stor_dis = np.interp(freq_dis, df_master['f'], df_master['E_stor_filt'])
        E_loss_dis = np.interp(freq_dis, df_master['f'], df_master['E_loss_filt'])

        #Estimate instantenous (E_0) and equilibrium (E_inf) modulus
        E_0 = df_master['E_stor_filt'].iloc[-1]
        E_inf = df_master['E_stor_filt'].iloc[0]

        #Assembly data frame
        df_dis = pd.DataFrame([freq_dis, E_stor_dis, E_loss_dis, omega_dis, tau]).T
        df_dis.columns = ['f', 'E_stor', 'E_loss', 'omega', 'tau']

    elif df_master.domain == 'time':
    
        #Interpolate E_stor and E_loss at discretization poins
        E_relax_dis = np.interp(t_dis, df_master['t'], df_master['E_relax_filt'])

        #Estimate instantenous (E_0) and equilibrium (E_inf) modulus
        E_0 = df_master['E_relax_filt'].iloc[0]
        E_inf = df_master['E_relax_filt'].iloc[-1]

        #Assembly data frame
        df_dis = pd.DataFrame([tau, t_dis, E_relax_dis, omega_dis, freq_dis]).T
        df_dis.columns = ['tau', 't', 'E_relax', 'omega', 'f']

    #Add df attributes    
    df_dis.index += 1 
    df_dis.nprony = nprony
    df_dis.E_0 = E_0
    df_dis.E_inf = E_inf
    df_dis.RefT = df_master.RefT
    df_dis.f_min = df_master['f'].min()
    df_dis.f_max = df_master['f'].max()
    df_dis.decades = decades

    return df_dis


def plot_master(df_master):

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


def plot_dis(df_master, df_dis):

    if df_master.domain == 'freq':

        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=['E_stor', 'E_loss'], ax=ax1, logx=True, color=['C0', 'C1'], alpha=0.5)
        df_dis.plot(x='f', y=['E_stor', 'E_loss'], label=['tau_i', 'tau_i'], ax=ax1, logx=True, ls='', marker='o', color=['C0', 'C1'])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage and loss modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig

    elif df_master.domain == 'time':

        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=['E_relax'], ax=ax1, logx=True, color=['k'])
        df_dis.plot(x='t', y=['E_relax'], label = ['tau_i'], ax=ax1, logx=True, ls='', marker='o', color=['red'])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig



#Prony series - Frequency domain
#-----------------------------------------------------------------------------

def E_freq(omega, alpha_i, tau_i):
    A = (omega*tau_i[:,None])
    A2 = A**2
    
    E_stor = 1-np.sum(alpha_i) + np.dot(alpha_i, A2/(A2+1))
    E_loss = np.dot(alpha_i, A/(A2+1))
    E_norm = np.concatenate((E_stor, E_loss))
    
    return E_norm

def res_freq(alpha_i, tau_i, E_freq_meas, omega_meas):
    res = np.sum((E_freq_meas - E_freq(omega_meas, alpha_i, tau_i))**2)
    return res

def opt_freq(x0, E_freq_meas, omega_meas):
    alpha_i = x0[0:int(x0.shape[0]/2)]
    tau_i = x0[int(x0.shape[0]/2):]
    return res_freq(alpha_i, tau_i, E_freq_meas, omega_meas)


def fit_prony_freq(df_dis, df_master=None, opt=False):
    #Assembly 'K_global' matrix [Kraus 2017, Eq. 22]

    N = df_dis.nprony #number of prony terms

    K_stor = np.tril(np.ones((N,N)), -1) + np.diag([0.5] * N)

    K_loss = (np.diag([0.5] * N) 
        + np.diag([0.1] * (N-1), 1) + np.diag([0.1] * (N-1), -1) 
        + np.diag([0.01] * (N-2), 2) + np.diag([0.01] * (N-2), -2)
        + np.diag([0.001] * (N-3), 3) + np.diag([0.001] * (N-3), -3))

    K_global = np.vstack([K_stor, K_loss, np.ones((1,N))])

    #Estimate instantenous (E_0) and equilibrium (E_inf) modulus
    E_0 = df_dis.E_0
    E_inf = df_dis.E_inf


    #Assembly right-hand vector
    E = np.concatenate((df_dis['E_stor']/(E_0-E_inf), df_dis['E_loss']/(E_0-E_inf), np.array([1])))

    #Solve equation system
    alpha_i, err = nnls(K_global, E)

    #use initial fit and try to optimize both alpha_i and tau_i
    if opt:
        #Get measurement data
        E_freq_meas = np.concatenate((df_master['E_stor']/E_0, df_master['E_loss']/E_0))
        omega_meas = df_master['omega'].values

        #get Prony series
        tau_i = df_dis['tau']
        x0 = np.hstack((alpha_i, tau_i))

        #Define bounds
        tau_max = 1/(2*np.pi*df_dis.f_min)
        tau_min = 1/(2*np.pi*df_dis.f_max)
        bnd_t = ((tau_min, tau_max),)*alpha_i.shape[0]
        bnd_a = ((0,1),)*alpha_i.shape[0]
        bnd = bnd_a + bnd_t

        #find optimal Prony parameters
        res = minimize(opt_freq, x0, args=(E_freq_meas, omega_meas), bounds=bnd,  method='L-BFGS-B', options={'maxls':100})
        alpha_i = res.x[0:int(res.x.shape[0]/2)]
        df_dis['tau'] = res.x[int(res.x.shape[0]/2):]
        err = res.fun
        print(res.success)

    #Ensure that Sum(alpha_i) < 1 (otherwise can lead to numerical difficulties in FEM)
    if alpha_i.sum() >= 1:
        df_dis['alpha'] = 0.99/alpha_i.sum()*alpha_i #Normalize alpha values to 0.99
    else:
        df_dis['alpha'] = alpha_i

    df_prony = df_dis[['tau', 'alpha']].copy()
    df_prony = df_prony.iloc[::-1].reset_index(drop=True)
    df_prony.index += 1 
    df_prony['E_0'] = E_0
    df_prony['E_i'] = E_0 * df_prony['alpha']
    df_prony.RefT = df_dis.RefT

    prony = {'E_0':E_0, 'df_terms':df_prony, 'f_min':df_dis.f_min, 'f_max':df_dis.f_max, 'label':'equi.', 'err' : err, 'decades':df_dis.decades}
    
    return prony



#Prony series - Time domain
#-----------------------------------------------------------------------------

def E_relax(time, alpha_i, tau_i):
    return (1-np.dot(alpha_i, 1-np.exp(-time/tau_i[:,None])))
    
    #y = np.zeros(time.shape[0])
    #for i, t in enumerate(time):
    #    y[i] = E_0 * (1 - np.sum(alpha_i*(1-np.exp(-t/tau_i))))
    #return y

def res_time(alpha_i, tau_i, E_relax_meas, time_meas):
    return np.sum((E_relax_meas - E_relax(time_meas, alpha_i, tau_i))**2)


def opt_time(x0, E_relax_meas, time_meas):
    alpha_i = x0[0:int(x0.shape[0]/2)]
    tau_i = x0[int(x0.shape[0]/2):]
    return res_time(alpha_i, tau_i, E_relax_meas, time_meas)


def fit_prony_time(df_dis, df_master, opt=False):

    alpha_i = np.ones(df_dis['tau'].values.shape) #start all a_i = 1
    tau_i = df_dis['tau'].values
    E_0 = df_dis.E_0
    E_relax_meas = df_master['E_relax_filt'].values / E_0
    time_meas = df_master['t'].values
    #N = df_dis.nprony
    bnd_a = ((0,1),)*alpha_i.shape[0]

    res = minimize(res_time, alpha_i, args=(tau_i, E_relax_meas, time_meas), method='L-BFGS-B', bounds=bnd_a)
    
    alpha_i = res.x

    #use initial fit and try to optimize both alpha_i and tau_i
    if opt:
        x0 = np.hstack((alpha_i, tau_i))
        tau_max = 1/(2*np.pi*df_dis.f_min)
        tau_min = 1/(2*np.pi*df_dis.f_max)
        bnd_t = ((tau_min, tau_max),)*alpha_i.shape[0]
        bnd = bnd_a + bnd_t
        res = minimize(opt_time, x0, args=(E_relax_meas, time_meas), method='L-BFGS-B' , bounds=bnd) 

        alpha_i = res.x[0:int(res.x.shape[0]/2)]
        df_dis['tau'] = res.x[int(res.x.shape[0]/2):]
     

    #Ensure that Sum(alpha_i) < 1 (otherwise can lead to numerical difficulties in FEM)
    if alpha_i.sum() >= 1:
        df_dis['alpha'] = 0.99/alpha_i.sum()*alpha_i #Normalize alpha values to 0.99
    else:
        df_dis['alpha'] = alpha_i

    df_prony = df_dis[['tau', 'alpha']].copy()
    df_prony = df_prony.iloc[::-1].reset_index(drop=True)
    df_prony.index += 1 
    df_prony['E_0'] = E_0
    df_prony['E_i'] = E_0 * df_prony['alpha']
    df_prony.RefT = df_dis.RefT

    prony = {'E_0':E_0, 'df_terms':df_prony, 'f_min':df_dis.f_min, 'f_max':df_dis.f_max, 'label':'equi.', 'err' : res.fun, 'decades':df_dis.decades}
    
    return prony







#Generalized Maxwell model
#-----------------------------------------------------------------------------

def calc_GenMaxw(E_0, df_terms, f_min, f_max, decades, **kwargs):

    alpha_i = df_terms['alpha'].values
    tau_i = df_terms['tau'].values

    #Define angular frequency range for plotting
    omega_min = 2*np.pi*f_min
    omega_max = 2*np.pi*f_max
    omega_len = 10*decades #number of datapoints along x-axis (approx. 10 per decade)

    #Define dataframe
    df_GMaxw = pd.DataFrame(np.zeros((omega_len, 8)), 
        columns=(['f', 'omega', 'E_stor', 'E_loss', 'E_comp', 'tan_del', 't', 'E_relax']))

    #Fill frequency and time axis
    df_GMaxw['omega'] = np.geomspace(omega_min, omega_max, omega_len)
    df_GMaxw['f'] = df_GMaxw['omega']/(2*np.pi)
    df_GMaxw['t'] = 1/df_GMaxw['f'] 

    #Calculate frequency domain
    #for i, omega in enumerate(df_GMaxw['omega']): #TODO: use linear algebra
    #    df_GMaxw['E_stor'][i] = E_0 * (1-np.sum(alpha_i-((alpha_i*(tau_i*omega)**2)/(1+(tau_i*omega)**2))))
    #    df_GMaxw['E_loss'][i] = E_0 * (np.sum((alpha_i*tau_i*omega)/(1+(tau_i*omega)**2)))

    E_inf = E_0*(1-np.sum(alpha_i))
    A = (df_GMaxw['omega'].values*tau_i[:,None])
    A2 = (df_GMaxw['omega'].values*tau_i[:,None])**2
    df_GMaxw['E_stor'] = E_inf + np.dot(E_0*alpha_i, A2/(A2+1))
    df_GMaxw['E_loss'] = np.dot(E_0*alpha_i, A/(A2+1))
    df_GMaxw['E_comp'] = (df_GMaxw['E_stor']**2 + df_GMaxw['E_loss']**2)**0.5  
    df_GMaxw['tan_del'] = df_GMaxw['E_loss']/df_GMaxw['E_stor']

    #Calculate time domain
    df_GMaxw['E_relax'] = E_0 * E_relax(df_GMaxw['t'].values, alpha_i, tau_i)

    #for i, t in enumerate(df_GMaxw['t']):
    #    df_GMaxw['E_relax'][i] = E_0 * (1 - np.sum(alpha_i*(1-np.exp(-t/tau_i))))

    return df_GMaxw


def plot_GMaxw(df_GMaxw):

    fig1, ax1 = plt.subplots()
    df_GMaxw.plot(x='f', y=['E_stor'], ax=ax1, logx=True, ls='-', lw=2, color=['C0'])
    df_GMaxw.plot(x='f', y=['E_loss'], ax=ax1, logx=True, ls=':', lw=2, color=['C1'])
    df_GMaxw.plot(x='f', y=['E_relax'], ax=ax1, logx=True, ls='--', lw=2, color=['C2'])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Relaxation, storage and \n loss modulus (MPa)') #TODO: Make sure it makes sense to include units here...
    fig1.show()
    

    #fig2, ax2 = plt.subplots()
    #df_GMaxw.plot(x='t', y=['E_relax'], ax=ax2, logx=True, ls='-', lw=2, color=['C0'])
    #ax2.set_xlabel('Time (s)')
    #ax2.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...
    #fig2.show()

    return fig1 #, fig2


def plot_fit(df_master, df_GMaxw):

    if df_master.domain == 'freq':

        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=['E_stor', 'E_loss'], ax=ax1, logx=True, color=['C0', 'C1'], alpha=0.5, ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='f', y=['E_stor', 'E_loss'], ax=ax1, logx=True, ls='-', lw=2, color=['C0', 'C1'])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage and loss modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig

    if df_master.domain == 'time':

        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=['E_relax'], ax=ax1, logx=True, color=['gray'], ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='t', y=['E_relax'], label=['fit'], ax=ax1, logx=True, ls='-', lw=2, color=['r'])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig



#Find optimal number of Prony terms for FEM
#-----------------------------------------------------------------------------
def find_nprony(df_master, prony, window='min', err_opt = 0.025):
    dict_prony = {}
    nprony = prony['df_terms'].shape[0]
    
    for i in range(nprony-2):
        N = nprony - i
        df_dis = discretize(df_master, window, N)
        if df_master.domain == 'time':
            prony = fit_prony_time(df_dis, df_master, opt=True)
        elif df_master.domain == 'freq':
            prony = fit_prony_freq(df_dis, df_master, opt=True)

        dict_prony[N] = prony 
        
    err = pd.DataFrame()
    for key, item in dict_prony.items():
        err.at[key, 'res'] = item['err']
        
        N_opt = (err['res']-err_opt).abs().sort_values().index[0]
               
    return dict_prony, N_opt, err


def plot_nprony(df_master, dict_prony, N):
    
    df_GMaxw = calc_GenMaxw(**dict_prony[N])
    fig = plot_fit(df_master, df_GMaxw)

    return df_GMaxw, fig

def plot_residual(N_opt_err):

    fig, ax = plt.subplots()
    N_opt_err.plot(y=['res'], ax=ax, c='k', label=['Least squares residual'], marker='o', ls='--', markersize=4, lw=1)
    ax.set_xlabel('Number of Prony terms')
    ax.set_ylabel(r'$R^2 = \sum \left[E_{meas} - E_{Prony} \right]^2$') 
    ax.set_xlim(0,)
    ax.set_ylim(-0.01,0.25)
    ax.legend()

    fig.show()
    return fig




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


def prep_prony_ANSYS(df_prony_ANSYS, prony):
    E_0 = prony['E_0'] #use same estimate as for GUSTL fit
    f_min = prony['f_min'] #use same frequency range as for fit
    f_max = prony['f_max']

    prony_ANSYS = {'E_0':E_0, 'df_terms':df_prony_ANSYS, 'f_min':f_min, 'f_max':f_max, 'label':'ANSYS'}

    return prony_ANSYS


def plot_fit_ANSYS(df_master, df_GMaxw, df_GMaxw_ANSYS):

    if df_master.domain == 'freq':

        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=['E_stor', 'E_loss'], label=["E'(master)", "E''(master)"], ax=ax1, logx=True, color=['C0', 'C1'], alpha=0.5, ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='f', y=['E_stor', 'E_loss'], label=["E'(GUSTL)", "E''(GUSTL)"], ax=ax1, logx=True, ls='-', lw=2, color=['C0', 'C1'])
        df_GMaxw_ANSYS.plot(x='f', y=['E_stor', 'E_loss'], label=["E'(ANSYS)", "E''(ANSYS)"], ax=ax1, logx=True, ls='-', lw=2, color=['C2', 'C3'])

        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Moduli (MPa)') #TODO: Make sure it makes sense to include units here...

        fig.show()
        return fig

    elif df_master.domain == 'time':

        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=['E_relax'], label=['Measurement'], ax=ax1, logx=True, color=['C0'], alpha=0.5, ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='t', y=['E_relax'], label=['GenMaxw(scipy)'], ax=ax1, logx=True, ls='-', lw=2, color=['C0'])
        df_GMaxw_ANSYS.plot(x='t', y=['E_relax'], label=['GenMaxw(ANSYS)'], ax=ax1, logx=True, ls='-', lw=2, color=['C2'])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...

        fig.show()
        return fig

def plot_prony_coeff(prony_list):
    df_list = []
    for prony in prony_list:
        df = prony['df_terms'][['tau', 'alpha']].copy()
        df = df.set_index('tau')
        df.columns = [prony['label']]
        
        df_list.append(df)

    df_bar = pd.concat(df_list, axis=1)

    fig, ax1 = plt.subplots()
    df_bar.plot.bar(ax=ax1)

    xticklabels = [("{:.0e}".format(a)) for a in df_bar.index.tolist()]
    ax1.set_xticklabels(xticklabels)
    ax1.set_xlabel(r'$\tau_i$')
    ax1.set_ylabel(r'$\alpha_i$')

    fig.show()
    return fig


#Convenience functions
#-----------------------------------------------------------------------------

def generate_zip(files):
    mem_zip = io.BytesIO()

    with zipfile.ZipFile(mem_zip, mode="w",compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            pre = name.split(sep='_')[0]
            if pre == 'df':
                fname = name + '.csv'
            elif pre == 'fig':
                fname = name + '.png'
            else:
                fname = name
            zf.writestr(fname, data)

    return mem_zip.getvalue()

def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, dpi = 600)
    return buf.getvalue()


#Define figure style
#-----------------------------------------------------------------------------
def format_fig():
    #Set default colors
    #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette("colorblind")) #colorblind

    #Activate Grid
    plt.rcParams['axes.grid'] = True

    #Set default figure size to one column
    plt.rcParams['figure.figsize'] = (4.5,0.75*4.5)

    #Increase default resolution
    #plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600

    #Default use of Latex
    #plt.rcParams['text.usetex'] = True
    #plt.rcParams['font.family'] = 'serif' #sans-serif, monospace
    plt.rcParams['font.size'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9

    #Change grid line properties
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.4

    #Change marker properties
    plt.rcParams['lines.markersize'] = 3.0

    #Change tick direction to inwards
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    #Define default legend options
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.edgecolor'] = '0'
    plt.rcParams['legend.handlelength'] = 2.2
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.fontsize'] = 9

    #Use constraint layout
    plt.rcParams['figure.constrained_layout.use'] = True
    #plt.rcParams['figure.constrained_layout.w_pad'] = 0.05
    #plt.rcParams['figure.constrained_layout.h_pad'] = 0.05

    #ipympl duplicate plot issue
    #https://issueexplorer.com/issue/matplotlib/ipympl/402
    plt.ioff()

    #Jupyter widget specific
    Canvas.header_visible.default_value = False
    Canvas.footer_visible.default_value = False
    #Canvas.resizable.default_value = False



#Classes for Jupyter notebook
#-----------------------------------------------------------------------------
class Widgets():
    """Create widgets for GUI"""
    def __init__(self):
        #self.notebook_width()
        self.ini_variables()
        self.widgets()
        self.layout()
        #self.show()

    def notebook_width(self):
        """Use full screen width for notebook."""
        display(HTML(
            '<style>'
                '#notebook { padding-top:0px !important; } '
                '.container { width:100% !important; } '
                '.end_space { min-height:0px !important; } '
            '</style>'
        ))


    def ini_variables(self):
        self.RefT = 0
        self.nprony = 0

    def widgets(self):
        """Define GUI widgets."""      
        
        ###INPUTS###
        #--------------------------------------

        _height = 'auto'
        _width = 'auto'
        _width_b = '200px'
        
        #Radio buttons -----------------------------
        self.rb_eplexor = widgets.RadioButtons(
            options=['Eplexor', 'user'],
            value='Eplexor', 
            description='Instrument:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_eplexor.observe(self.set_ftype, 'value')
        
        self.rb_type = widgets.RadioButtons(
            options=['master', 'raw'],
            value='master',
            description='Type:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_type.observe(self.set_mastern, 'value')
            
        self.rb_domain = widgets.RadioButtons(
            options=['freq', 'time'],
            value='freq', 
            description='Domain:',
            disabled=False,
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_domain.observe(self.set_instr, 'value')

        self.rb_dis = widgets.RadioButtons(
            options=['default', 'manual'],
            value='default', 
            description='Discretization:',
            disabled=False,
            style = {'description_width' : 'initial'},
            layout = widgets.Layout(height = _height, width = _width))
        self.rb_dis.observe(self.set_dis, 'value')

        self.rb_dis_win = widgets.RadioButtons(
            options=['round', 'exact'],
            value='round', 
            description='Window:',
            disabled=True,
            layout = widgets.Layout(height = _height, width = _width))

        
        #Check box -------------------------------
        self.cb_shift = widgets.Checkbox(
            value=False, 
            description='user shift factors',
            disabled=False,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        self.cb_shift.observe(self.set_shift, 'value')

        self.cb_aT = widgets.Checkbox(
            value=False, 
            description='fit and overwrite provided shift factors?',
            disabled=True,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        #self.cb_WLF.observe(self.set_shift, 'value')

        self.cb_WLF = widgets.Checkbox(
            value=False, 
            description='fit and overwrite provided WLF?',
            disabled=True,
            indent = True,
            layout = widgets.Layout(height = _height, width = _width),
            style = {'description_width' : 'initial'})
        #self.cb_WLF.observe(self.set_shift, 'value')
        
        #Text field -------------------------------
        self.ft_RefT = widgets.FloatText(
            value=self.RefT,
            description='Reference temperature (C):',
            disabled=False,
            layout = widgets.Layout(height = _height, width = '220px'),
            style = {'description_width' : 'initial'})
        self.ft_RefT.observe(self.set_RefT, 'value')

        self.it_nprony = widgets.BoundedIntText(
            value=self.nprony,
            min = 0,
            max = 1000,
            step = 1,
            description='Number of Prony terms:',
            disabled=True,
            layout = widgets.Layout(height = _height, width = '220px'),
            style = {'description_width' : 'initial'})
        #self.it_nprony.observe(self.set_nprony, 'value')
        
        #Upload buttons ---------------------------
        self.up_inp = widgets.FileUpload(
            accept= '.xls, .xlsx',
            multiple=False,
            layout = widgets.Layout(height = _height, width = _width_b))
        
        self.up_shift = widgets.FileUpload(
            accept='.csv, .xls', 
            multiple=False, 
            disabled=True,
            layout = widgets.Layout(height = _height, width = _width_b))


        #Valid indicator ---------------------------------
        self.v_modulus = widgets.Valid(
            value=False,
            description='modulus data',
            continuous_update=True,
            readout = '(N/P)', #string.whitespace
            layout = widgets.Layout(height = _height, width = _width))

        self.v_aT = widgets.Valid(
            value=False,
            description='shift factors',
            continuous_update=True,
            readout = '(N/P)', #string.whitespace
            layout = widgets.Layout(height = _height, width = _width))

        self.v_WLF = widgets.Valid(
            value=False,
            description='WLF shift func. (Eplexor master)',
            continuous_update=True,
            readout = '(N/P)', #string.whitespace
            style = {'description_width' : 'initial'},
            layout = widgets.Layout(height = _height, width = _width))
    
        #Buttons and outputs ---------------------------------
        
        #Load
        self.b_load = widgets.Button(
            description='Load data',
            button_style='success',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_load.on_click(self.inter_load_master)
        
        self.out_load = widgets.Output()

                
        #fit shift factors
        self.b_aT = widgets.Button(
            description='master raw data',
            button_style='info',
            disabled = True,
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_aT.on_click(self.inter_aT)

        self.out_aT = widgets.Output()
        
        
        #fit shift functions
        self.b_shift = widgets.Button(
            description='(fit) & plot shift functions',
            button_style='info',
            disabled = True,
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_shift.on_click(self.inter_shift)

        self.out_shift = widgets.Output()
        
        #Smooth
        self.b_smooth = widgets.Button(
            description='smooth master curve',
            button_style='info',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_smooth.on_click(self.inter_smooth_fig)

        self.out_smooth = widgets.Output()
        
        #Discretization
        self.b_dis = widgets.Button(
            description='plot discretization',
            button_style='info', 
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_dis.on_click(self.inter_dis)

        self.out_dis = widgets.Output()
        
        #Prony fit
        self.b_fit = widgets.Button(
            description='fit Prony series',
            button_style='danger',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_fit.on_click(self.inter_fit)

        self.out_fit = widgets.Output()
        
        self.out_html = widgets.Output()
        
        #Generalized Maxwell
        self.b_GMaxw = widgets.Button(
            description='plot Generalized Maxwell',
            button_style='info',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_GMaxw.on_click(self.inter_GMaxw)
        
        self.out_GMaxw = widgets.Output()

        #Minimize nprony
        self.b_opt = widgets.Button(
            description='minimize Prony terms',
            button_style='info',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.b_opt.on_click(self.inter_opt)
        
        self.out_opt = widgets.Output()
        self.out_res = widgets.Output()

        
        #Download/HTML buttons -----------------------
        self.db_prony = widgets.Button(
            description='Download Prony series',
            button_style='warning',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.db_prony.on_click(self.down_prony)
        
        self.db_zip = widgets.Button(
            description='Download zip',
            button_style='warning',
            layout = widgets.Layout(height = _height, width = _width_b))
        self.db_zip.on_click(self.down_zip)

        self.b_reload = widgets.Button(
            description='Clear notebook!',
            button_style='danger',
            layout = widgets.Layout(height = 'auto', width = _width_b))
        self.b_reload.on_click(self.reload)


    def layout(self):

        self.w_inp_gen = widgets.HBox([
            self.rb_domain,
            self.rb_eplexor,
            self.rb_type,
            self.up_inp,],
            layout = widgets.Layout(width = '100%', justify_content='space-between'))

        self.w_inp_shift = widgets.HBox([
            self.ft_RefT,
            self.cb_shift,
            self.up_shift],
            layout = widgets.Layout(width = '100%', justify_content='space-between'))

        #self.w_inp_load = widgets.HBox([

        _load = widgets.HBox([
            self.b_load,
            self.v_modulus,
            self.v_aT,
            self.v_WLF], 
            layout = widgets.Layout(width = '100%', justify_content='space-between'))
            
        self.w_inp_load = widgets.VBox([
            _load,
            self.out_load],
            layout = widgets.Layout(width = '100%')) #, align_items='center'

        self.w_aT = widgets.VBox([
            widgets.HBox([self.b_aT, self.cb_aT]),
            self.out_aT])


        self.w_shift = widgets.VBox([
            widgets.HBox([self.b_shift, self.cb_WLF]),
            self.out_shift])

        _dis = widgets.HBox([
            self.rb_dis, 
            self.rb_dis_win, 
            self.it_nprony],  
            layout = widgets.Layout(width = '100%', justify_content='space-between'))

        self.w_dis = widgets.VBox([
            _dis,
            self.b_dis,
            self.out_dis])
        


         
class GUIControl(Widgets):
    """GUI Controls"""
    def __init__(self):
        super().__init__()
        self.collect_files()
        self.create_loading_bar()

    def collect_files(self):
        self.files = {}

    def create_loading_bar(self):
        gif_address = './figures/loading.gif'

        with open(gif_address, 'rb') as f:
            img = f.read()

        self.w_loading = widgets.Image(value=img)
        
    #Set widgets and variables--------------------------------------------------------------------------------  
    def set_shift(self, change):
        if change['new']:
            self.up_shift.disabled = False
        else:
            self.up_shift.disabled = True
            
    def set_RefT(self, change):
        try:
            self.arr_RefT
        except AttributeError:
            self.RefT = change['new']
        else:
            self.RefT = self.arr_RefT.iloc[(self.arr_RefT-change['new']).abs().argsort()[:1]].values[0]
            self.ft_RefT.value = self.RefT

    #def set_nprony(self, change):
    #    self.nprony = change['new']

    def set_mastern(self, change):
        if change['new'] == 'raw':
            self.b_aT.disabled = False
        else:
            self.b_aT.disabled = True

    def set_ftype(self, change):
        if change['new'] == 'Eplexor':
            self.up_inp.accept='.xls, .xlsx'
        elif change ['new'] == 'user':
            self.up_inp.accept='.csv'

    def set_instr(self, change):
        if change['new'] == 'freq':
            self.rb_eplexor.disabled = False
        elif change ['new'] == 'time':
            self.rb_eplexor.value = 'user'
            self.rb_eplexor.disabled = True

    def set_dis(self, change):
        if change['new'] == 'default':
            self.rb_dis_win.disabled = True
            self.it_nprony.disabled = True
            self.it_nprony.value = 0
        elif change['new'] == 'manual':
            self.rb_dis_win.disabled = False
            self.it_nprony.disabled = False




           
        
    #Interactive functionality---------------------------------------------------------------------------------       
    def inter_load_master(self, b):
        with self.out_load:
            clear_output()
        try:
            self.df_master = None
            self.df_raw = None
            self.df_aT = None
            self.df_WLF = None
            self.df_GMaxw = None
            self.prony = None
            self.v_modulus.value = False
            self.v_aT.value = False
            self.v_WLF.value = False
            self.ft_RefT.disabled = False
            self.cb_aT.disabled = True
            self.cb_WLF.disabled = True
            self.b_shift.disabled = True
            self.collect_files()

            #Load modulus
            if self.rb_eplexor.value == 'Eplexor':
                if self.rb_type.value == 'master':
                    self.df_master, self.df_aT, self.df_WLF  = load_Eplexor_master(self.up_inp.data[0])
                    self.ft_RefT.value = self.df_master.RefT
                    self.ft_RefT.disabled = True

                   
                elif self.rb_type.value == 'raw':
                    self.df_raw, self.arr_RefT = load_Eplexor_raw(self.up_inp.data[0])
                    _change = {}
                    _change['new'] = self.ft_RefT.value
                    self.set_RefT(_change)
   
                
            elif self.rb_eplexor.value == 'user':
                if self.rb_type.value == 'master':
                    self.df_master = load_user_master(self.up_inp.data[0], self.rb_domain.value, self.RefT)

                elif self.rb_type.value == 'raw':
                    self.df_raw, self.arr_RefT = load_user_raw(self.up_inp.data[0], self.rb_domain.value)
                    _change = {}
                    _change['new'] = self.ft_RefT.value
                    self.set_RefT(_change)

            #Load shift factors
            if self.cb_shift.value:
                self.df_aT = load_user_shift(self.up_shift.data[0])

            #Add dataframes to zip package and adjust widgets
            if isinstance(self.df_master, pd.DataFrame):             
                self.files['df_master'] = self.df_master.to_csv(index = False)
                self.v_modulus.value = True
            if isinstance(self.df_raw, pd.DataFrame):             
                self.files['df_raw'] = self.df_raw.to_csv(index = False)
                self.v_modulus.value = True
            if isinstance(self.df_aT, pd.DataFrame):
                self.files['df_aT'] = self.df_aT.to_csv(index = False)
                self.v_aT.value = True
                self.b_shift.disabled = False
                if isinstance(self.df_raw, pd.DataFrame):   
                    self.cb_aT.disabled = False
            if isinstance(self.df_WLF, pd.DataFrame):
                self.files['df_WLF'] = self.df_WLF.to_csv(index = False)
                self.v_WLF.value = True
                self.cb_WLF.disabled = False

            with self.out_load:
                print('Upload successful!')
            
        #except IndexError:
        #    with self.out_load:
        #        print('Upload files first!')
        except KeyError:
            with self.out_load:
                print('Column names not as expected. Check the headers in your input files!')
        except ValueError:
            with self.out_load:
                print('Uploaded files not in required format!')


    def inter_aT(self,b):
        with self.out_aT:
            clear_output()
            if not isinstance(self.df_aT, pd.DataFrame) or self.cb_aT.value:
                self.df_aT = get_shift_pwr(self.df_raw, self.RefT)

            self.df_master = df_shift_master(self.df_raw, self.df_aT, self.RefT)

            self.files['df_master'] = self.df_master.to_csv(index = False)
            self.files['df_aT'] = self.df_aT.to_csv(index = False)
            self.b_shift.disabled = False

            try:
                self.fig_master_shift = plot_master_shift(self.df_raw, self.df_master)
                self.files['fig_master_shift'] = fig_bytes(self.fig_master_shift)
            except NameError:
                with self.out_aT:
                    print('Raw and/or master dataframes are missing!')

        
                
    def inter_shift(self, b):
        with self.out_shift:
            clear_output()
            if not isinstance(self.df_WLF, pd.DataFrame) or self.cb_WLF.value:
                self.df_WLF = fit_shift_WLF(self.df_master.RefT, self.df_aT)
            self.df_poly = fit_shift_poly(self.df_aT)

            self.fig_shift, self.df_shift = plot_shift_func(self.df_aT, self.df_WLF, self.df_poly)
            
        self.files['fig_shift'] = fig_bytes(self.fig_shift)
        self.files['df_shift_plot'] = self.df_shift.to_csv(index = False)
        self.files['df_shift_WLF'] = self.df_WLF.to_csv(index = False)
        self.files['df_shift_poly'] = self.df_poly.to_csv(index_label = 'Coeff.')
            
    def inter_smooth(self, win):
        self.df_master = smooth(self.df_master, win)
        self.fig_smooth = plot_smooth(self.df_master)
        
        self.files['df_master'] = self.df_master.to_csv(index = False)
        self.files['fig_smooth'] = fig_bytes(self.fig_smooth)

    def inter_smooth_fig(self, b):
        with self.out_smooth:
            clear_output()
            widgets.interact(self.inter_smooth, 
                     win=widgets.IntSlider(min=1, max=20, step=1, value=1, 
                        description = 'Window size:',
                        style = {'description_width' : 'initial'},
                        layout = widgets.Layout(height = 'auto', width = '300px'),
                        continuous_update=False))
            
    def inter_dis(self, b):
        self.df_dis = discretize(self.df_master, self.rb_dis_win.value, self.it_nprony.value)
        with self.out_dis:
            clear_output()
            self.fig_dis = plot_dis(self.df_master, self.df_dis)
        
        self.it_nprony.value = self.df_dis.nprony
        self.files['df_dis'] = self.df_dis.to_csv()
        self.files['fig_dis'] = fig_bytes(self.fig_dis)
            
    def inter_fit(self, b):
        if self.rb_domain.value == 'freq':
            self.prony = fit_prony_freq(self.df_dis)
        elif self.rb_domain.value == 'time':
            self.prony = fit_prony_time(self.df_dis, self.df_master)
            
        self.df_GMaxw = calc_GenMaxw(**self.prony)
        
        with self.out_fit:
            clear_output()
            self.fig_fit = plot_fit(self.df_master, self.df_GMaxw)
                
        self.files['fig_fit'] = fig_bytes(self.fig_fit)
        self.files['df_GMaxw'] = self.df_GMaxw.to_csv(index = False)
        self.files['df_prony'] = self.prony['df_terms'].to_csv(index_label = 'i')
        
    def inter_GMaxw(self, b):
        with self.out_GMaxw:
            clear_output()
            self.fig_GMaxw = plot_GMaxw(self.df_GMaxw)
        
        self.files['fig_GMaxw'] = fig_bytes(self.fig_GMaxw)

    def inter_opt_fig(self, N):
        self.df_GMaxw_opt, self.fig_opt = plot_nprony(self.df_master, self.dict_prony, N)
        self.files['df_prony_min'] = self.dict_prony[N]['df_terms'].to_csv(index_label = 'i')
        self.files['df_GMaxw_min'] = self.df_GMaxw_opt.to_csv(index = False)
        self.files['fig_fit_min'] = fig_bytes(self.fig_opt)
        

    def inter_opt(self, b):
        with self.out_opt:
            clear_output()
            display(self.w_loading)
            self.dict_prony, self.N_opt, self.N_opt_err = find_nprony(self.df_master, self.prony, window='min')
            clear_output()
            widgets.interact(self.inter_opt_fig, 
                    N=widgets.Dropdown(
                        options=self.dict_prony.keys(), 
                        value=self.N_opt, 
                        description = 'Number of Prony terms:',
                        style = {'description_width' : 'initial'},
                        layout = widgets.Layout(height = 'auto', width = '200px'),
                        continuous_update=False))

        with self.out_res:
            clear_output()
            self.fig_res = plot_residual(self.N_opt_err)
            self.files['fig_res'] = fig_bytes(self.fig_res)


                
       
    #Download functionality---------------------------------------------------------------------------------
    def trigger_download(self, data, filename, kind='text/json'):
        # see https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URIs for details
        if isinstance(data, str):
            content_b64 = b64encode(data.encode()).decode()
        elif isinstance(data, bytes):
            content_b64 = b64encode(data).decode()
        data_url = f'data:{kind};charset=utf-8;base64,{content_b64}'
        js_code = f"""
            var a = document.createElement('a');
            a.setAttribute('download', '{filename}');
            a.setAttribute('href', '{data_url}');
            a.click()
        """
        with self.out_html:
            clear_output()
            display(HTML(f'<script>{js_code}</script>'))
            
    def down_prony(self, b):
        self.trigger_download(self.files['df_prony'], 'df_prony.csv', kind='text/plain')
        
    def down_zip(self, b):
        zip_b64 = generate_zip(self.files)
        self.trigger_download(zip_b64, 'fit.zip', kind='text/plain')

    #Clear/refresh notebook---------------------------------------------------------------------------------
    def reload(self,b):
        with self.out_html:
            clear_output()
            display(HTML(
                '''
                    <script>
                        window.location.reload();
                    </script>            
                '''
            ))


