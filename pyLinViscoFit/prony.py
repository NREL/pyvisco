"""Collection of function to pre-process the master curve and perform the Prony 
series parameter identification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, nnls
from . import shift


#Prony series - Frequency domain
#-----------------------------------------------------------------------------

def E_freq_norm(omega, alpha_i, tau_i):
    A = (omega*tau_i[:,None])
    A2 = A**2
    
    E_stor = 1-np.sum(alpha_i) + np.dot(alpha_i, A2/(A2+1))
    E_loss = np.dot(alpha_i, A/(A2+1))
    E_norm = np.concatenate((E_stor, E_loss))
    
    return E_norm

def res_freq(alpha_i, tau_i, E_freq_meas, omega_meas):
    res = np.sum((E_freq_meas - E_freq_norm(omega_meas, alpha_i, tau_i))**2)
    return res

def opt_freq(x0, E_freq_meas, omega_meas):
    alpha_i = x0[0:int(x0.shape[0]/2)]
    tau_i = x0[int(x0.shape[0]/2):]
    return res_freq(alpha_i, tau_i, E_freq_meas, omega_meas)

def fit_freq(df_dis, df_master=None, opt=False):
    m = df_dis.modul
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
    E = np.concatenate((df_dis['{}_stor'.format(m)]/(E_0-E_inf), 
                        df_dis['{}_loss'.format(m)]/(E_0-E_inf), 
                        np.array([1])))

    #Solve equation system
    alpha_i, err = nnls(K_global, E)

    #use initial fit and try to optimize both alpha_i and tau_i
    if opt:
        #Get measurement data
        E_freq_meas = np.concatenate((df_master['{}_stor'.format(m)]/E_0, 
                                      df_master['{}_loss'.format(m)]/E_0))
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
        res = minimize(opt_freq, x0, args=(E_freq_meas, omega_meas), 
            bounds=bnd,  method='L-BFGS-B', options={'maxls':200})
        
        alpha_i = res.x[0:int(res.x.shape[0]/2)]
        df_dis['tau'] = res.x[int(res.x.shape[0]/2):]
        err = res.fun
        if res.success:
            print('Prony series fit N = {:02d}: Succesful!'.format(alpha_i.shape[0]))
        else:
            print('Prony series fit N = {:02d}: Failed to converge!'.format(alpha_i.shape[0]))

    #Ensure that Sum(alpha_i) < 1 (otherwise can lead to numerical difficulties in FEM)
    if alpha_i.sum() >= 1:
        df_dis['alpha'] = 0.999/alpha_i.sum()*alpha_i #Normalize alpha values to 0.999
    else:
        df_dis['alpha'] = alpha_i

    df_prony = df_dis[['tau', 'alpha']].copy()
    df_prony = df_prony.iloc[::-1].reset_index(drop=True)
    df_prony.index += 1 
    df_prony['{}_0'.format(m)] = E_0
    df_prony['{}_i'.format(m)] = E_0 * df_prony['alpha']
    df_prony.RefT = df_dis.RefT

    prony = {'E_0'.format(m):E_0, 'df_terms':df_prony, 'f_min':df_dis.f_min, 
        'f_max':df_dis.f_max, 'label':'equi.', 'err' : err, 'decades':df_dis.decades,
        'modul':m}
    
    return prony



#Prony series - Time domain
#-----------------------------------------------------------------------------

def E_relax_norm(time, alpha_i, tau_i):
    return 1-np.sum(alpha_i) + np.dot(alpha_i, np.exp(-time/tau_i[:,None]))

    #return (1-np.dot(alpha_i, 1-np.exp(-time/tau_i[:,None])))
    
    #y = np.zeros(time.shape[0])
    #for i, t in enumerate(time):
    #    y[i] = E_0 * (1 - np.sum(alpha_i*(1-np.exp(-t/tau_i))))
    #return y


def res_time(alpha_i, tau_i, E_meas_norm, time_meas):
    return np.sum((E_meas_norm - E_relax_norm(time_meas, alpha_i, tau_i))**2)
        

def opt_time(x0, E_meas_norm, time_meas):
    alpha_i = x0[0:int(x0.shape[0]/2)]
    tau_i = x0[int(x0.shape[0]/2):]
    return res_time(alpha_i, tau_i, E_meas_norm, time_meas)


def fit_time(df_dis, df_master, opt=False):
    m = df_dis.modul
    alpha_i = np.ones(df_dis['tau'].values.shape) #start all a_i = 1
    tau_i = df_dis['tau'].values

    E_meas_norm = df_master['{}_relax_filt'.format(m)].values / df_dis.E_0

    time_meas = df_master['t'].values
    #N = df_dis.nprony
    bnd_a = ((0,1),)*alpha_i.shape[0]


    res = minimize(res_time, alpha_i, args=(tau_i, E_meas_norm, time_meas), 
        method='L-BFGS-B', bounds=bnd_a)
    alpha_i = res.x

    #use initial fit and try to optimize both alpha_i and tau_i
    if opt:
        x0 = np.hstack((alpha_i, tau_i))
        tau_max = 1/(2*np.pi*df_dis.f_min)
        tau_min = 1/(2*np.pi*df_dis.f_max)
        bnd_t = ((tau_min, tau_max),)*alpha_i.shape[0]
        bnd = bnd_a + bnd_t

        res = minimize(opt_time, x0, args=(E_meas_norm, time_meas), 
            method='L-BFGS-B' , bounds=bnd) 

        if res.success:
            print('Prony series fit N = {:02d}: Succesful!'.format(alpha_i.shape[0]))
        else:
            print('Prony series fit N = {:02d}: Failed to converge!'.format(alpha_i.shape[0]))

        alpha_i = res.x[0:int(res.x.shape[0]/2)]
        df_dis['tau'] = res.x[int(res.x.shape[0]/2):]
     

    #Ensure that Sum(alpha_i) < 1 (otherwise can lead to numerical difficulties in FEM)
    if alpha_i.sum() >= 1:
        df_dis['alpha'] = 0.999/alpha_i.sum()*alpha_i #Normalize alpha values to 0.999
    else:
        df_dis['alpha'] = alpha_i

    df_prony = df_dis[['tau', 'alpha']].copy()
    df_prony = df_prony.iloc[::-1].reset_index(drop=True)
    df_prony.index += 1 
    df_prony['{}_0'.format(m)] = df_dis.E_0
    df_prony['{}_i'.format(m)] = df_dis.E_0 * df_prony['alpha']
    df_prony.RefT = df_dis.RefT

    prony = {'E_0':df_dis.E_0, 'df_terms':df_prony, 'f_min':df_dis.f_min, 
        'f_max':df_dis.f_max, 'label':'equi.', 'err' : res.fun, 'decades':df_dis.decades,
        'modul':m}
    
    return prony


#Fit general
#-----------------------------------------------------------------------------

def fit(df_dis, df_master=None, opt=False):
    if df_dis.domain == 'freq':
        prony = fit_freq(df_dis, df_master, opt)
    elif df_dis.domain == 'time':
        prony = fit_time(df_dis, df_master)

    df_GMaxw = calc_GMaxw(**prony)

    return prony, df_GMaxw

def plot_fit(df_master, df_GMaxw):
    m = df_master.modul
    if df_master.domain == 'freq':

        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=['{}_stor'.format(m), '{}_loss'.format(m)], 
            ax=ax1, logx=True, color=['C0', 'C1'], alpha=0.5, ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='f', y=['{}_stor'.format(m), '{}_loss'.format(m)], 
            ax=ax1, logx=True, ls='-', lw=2, color=['C0', 'C1'])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage and loss modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig

    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=['{}_relax'.format(m)], 
            ax=ax1, logx=True, color=['gray'], ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='t', y=['{}_relax'.format(m)], 
            label=['fit'], ax=ax1, logx=True, ls='-', lw=2, color=['r'])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig


#Generalized Maxwell model
#-----------------------------------------------------------------------------

def discretize(df_master, window='round', nprony=0):
    m = df_master.modul
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
        E_stor_dis = np.interp(freq_dis, df_master['f'], df_master['{}_stor_filt'.format(m)])
        E_loss_dis = np.interp(freq_dis, df_master['f'], df_master['{}_loss_filt'.format(m)])

        #Estimate instantenous (E_0) and equilibrium (E_inf) modulus
        E_0 = df_master['{}_stor_filt'.format(m)].iloc[-1]
        E_inf = df_master['{}_stor_filt'.format(m)].iloc[0]

        #Assembly data frame
        df_dis = pd.DataFrame([freq_dis, E_stor_dis, E_loss_dis, omega_dis, tau]).T
        df_dis.columns = ['f', '{}_stor'.format(m), '{}_loss'.format(m), 'omega', 'tau']

    elif df_master.domain == 'time':
    
        #Interpolate E_stor and E_loss at discretization poins
        E_relax_dis = np.interp(t_dis, df_master['t'], df_master['{}_relax_filt'.format(m)])

        #Estimate instantenous (E_0) and equilibrium (E_inf) modulus
        E_0 = df_master['{}_relax_filt'.format(m)].iloc[0]
        E_inf = df_master['{}_relax_filt'.format(m)].iloc[-1]

        #Assembly data frame
        df_dis = pd.DataFrame([tau, t_dis, E_relax_dis, omega_dis, freq_dis]).T
        df_dis.columns = ['tau', 't', '{}_relax'.format(m), 'omega', 'f']

    #Add df attributes    
    df_dis.index += 1 
    df_dis.nprony = nprony
    df_dis.E_0 = E_0
    df_dis.E_inf = E_inf
    df_dis.RefT = df_master.RefT
    df_dis.f_min = df_master['f'].min()
    df_dis.f_max = df_master['f'].max()
    df_dis.decades = decades
    df_dis.domain = df_master.domain
    df_dis.modul = df_master.modul

    return df_dis


def plot_dis(df_master, df_dis):
    m = df_master.modul
    if df_master.domain == 'freq':
        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=['{}_stor'.format(m), '{}_loss'.format(m)], 
            ax=ax1, logx=True, color=['C0', 'C1'], alpha=0.5)
        df_dis.plot(x='f', y=['{}_stor'.format(m), '{}_loss'.format(m)], 
            label=['tau_i', 'tau_i'], ax=ax1, logx=True, ls='', marker='o', color=['C0', 'C1'])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage and loss modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig

    elif df_master.domain == 'time':

        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=['{}_relax'.format(m)], ax=ax1, logx=True, color=['k'])
        df_dis.plot(x='t', y=['{}_relax'.format(m)], label = ['tau_i'], 
            ax=ax1, logx=True, ls='', marker='o', color=['red'])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...
        ax1.legend()

        fig.show()
        return fig


def calc_GMaxw(E_0, df_terms, f_min, f_max, decades, modul, **kwargs):
    m = modul
    alpha_i = df_terms['alpha'].values
    tau_i = df_terms['tau'].values

    #Define angular frequency range for plotting
    omega_min = 2*np.pi*f_min
    omega_max = 2*np.pi*f_max
    omega_len = 10*decades #number of datapoints along x-axis (approx. 10 per decade)

    #Define dataframe
    df_GMaxw = pd.DataFrame(np.zeros((omega_len, 8)), 
        columns=(['f', 'omega', 
            '{}_stor'.format(m), 
            '{}_loss'.format(m), 
            '{}_comp'.format(m), 
            'tan_del', 
            't', 
            '{}_relax'.format(m)]))

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
    df_GMaxw['{}_stor'.format(m)] = E_inf + np.dot(E_0*alpha_i, A2/(A2+1))
    df_GMaxw['{}_loss'.format(m)] = np.dot(E_0*alpha_i, A/(A2+1))
    df_GMaxw['{}_comp'.format(m)] = (df_GMaxw['{}_stor'.format(m)]**2 + df_GMaxw['{}_loss'.format(m)]**2)**0.5  
    df_GMaxw['tan_del'] = df_GMaxw['{}_loss'.format(m)]/df_GMaxw['{}_stor'.format(m)]

    #Calculate time domain
    df_GMaxw['{}_relax'.format(m)] =  E_0 * E_relax_norm(df_GMaxw['t'].values, alpha_i, tau_i)
     #for i, t in enumerate(df_GMaxw['t']):
    #    df_GMaxw['E_relax'][i] = E_0 * (1 - np.sum(alpha_i*(1-np.exp(-t/tau_i))))

    #Define attributes
    df_GMaxw.modul = m

    return df_GMaxw


def GMaxw_temp(shift_func, df_GMaxw, df_coeff, df_aT, freq = [1E-8, 1E-4, 1E0, 1E4]):
    m = df_GMaxw.modul
    df_temp = pd.DataFrame()

    T_min = int(df_aT['Temp'].min())
    T_max = int(df_aT['Temp'].max())

    for f in freq:
        for T in range(T_min, T_max+1):
            try:
                if shift_func == 'WLF':
                    coeff_WLF = df_coeff.values[0].tolist()
                    aT = 10**(-shift.WLF(T, *coeff_WLF))
                elif shift_func == 'D4':
                    coeff_D4 = df_coeff['P4 (C)'].tolist()
                    aT = 10**(-shift.poly4(T, *coeff_D4))
                elif shift_func == 'D3':
                    coeff_D3 = df_coeff['P3 (C)'].iloc[0:4].tolist()
                    aT = 10**(-shift.poly3(T, *coeff_D3))
                elif shift_func == 'D2':
                    coeff_D2 = df_coeff['P2 (C)'].iloc[0:3].tolist()
                    aT = 10**(-shift.poly2(T, *coeff_D2))
                elif shift_func == 'D1':
                    coeff_D1 = df_coeff['P1 (C)'].iloc[0:2].tolist()
                    aT = 10**(-shift.poly1(T, *coeff_D1))

                f_shift = aT * df_GMaxw['f']

            except OverflowError:
                continue

            if any(f_shift<=f) and not all(f_shift<=f):
                E_stor = np.interp(f, f_shift, df_GMaxw['{}_stor'.format(m)])
                E_loss = np.interp(f, f_shift, df_GMaxw['{}_loss'.format(m)])
                E_relax = np.interp(f, f_shift, df_GMaxw['{}_relax'.format(m)])
                tan_del = np.interp(f, f_shift, df_GMaxw['tan_del'])
                df = pd.DataFrame([[f, T, E_stor, E_loss, tan_del, E_relax]], 
                    columns=['f', 'T', '{}_stor'.format(m), '{}_loss'.format(m), 'tan_del', '{}_relax'.format(m)])
                df_temp = df_temp.append(df)
            else:
                continue
            
    df_GMaxw_temp = df_temp.reset_index(drop=True)
    df_GMaxw_temp.modul = m
    return df_GMaxw_temp


def plot_GMaxw(df_GMaxw):
    m = df_GMaxw.modul
    fig1, ax1 = plt.subplots() #figsize=(8,0.75*4)
    df_GMaxw.plot(x='f', y=['{}_stor'.format(m)], ax=ax1, logx=True, ls='-', lw=2, color=['C0'])
    df_GMaxw.plot(x='f', y=['{}_loss'.format(m)], ax=ax1, logx=True, ls=':', lw=2, color=['C1'])
    df_GMaxw.plot(x='f', y=['{}_relax'.format(m)], ax=ax1, logx=True, ls='--', lw=2, color=['C2'])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Relaxation, storage and \n loss modulus (MPa)') #TODO: Make sure it makes sense to include units here...
    fig1.show()
    
    #fig2, ax2 = plt.subplots()
    #df_GMaxw.plot(x='t', y=['E_relax'], ax=ax2, logx=True, ls='-', lw=2, color=['C0'])
    #ax2.set_xlabel('Time (s)')
    #ax2.set_ylabel('Relaxation modulus (MPa)') #TODO: Make sure it makes sense to include units here...
    #fig2.show()

    return fig1 #, fig2
    

def plot_GMaxw_temp(df_temp):
    m = df_temp.modul
    fig, ax1 = plt.subplots()
    for i, (f, df) in enumerate(df_temp.groupby('f')):
        df.plot(y='{}_stor'.format(m), x='T', ls='-', ax=ax1, label='f = {:.0e} Hz'.format(f), c='C{}'.format(i))
        df.plot(y='{}_loss'.format(m), x='T', ls=':', ax=ax1, label='', c='C{}'.format(i))
        df.plot(y='{}_relax'.format(m), x='T', ls='--', ax=ax1, c='C{}'.format(i), label='') #label='t = {:.0e} s'.format(1/f)
        
    ax1.set_xlabel('Temperature (\N{DEGREE SIGN}C)')
    ax1.set_ylabel('Relaxation, storage and \n loss modulus (MPa)')
    ax1.legend()

    fig.show()
    return fig



def plot_param(prony_list, labels=None):
    df_list = []
    for i, prony in enumerate(prony_list):
        df = prony['df_terms'][['tau', 'alpha']].copy()
        df = df.set_index('tau')
        if labels:
            df.columns = [labels[i]]
        else:
            df.columns = [prony['label']]
        
        df_list.append(df)

    df_bar = pd.concat(df_list, axis=1)

    fig, ax1 = plt.subplots(figsize=(8,0.75*4))
    df_bar.plot.bar(ax=ax1)

    xticklabels = [("{:.0e}".format(a)) for a in df_bar.index.tolist()]
    ax1.set_xticklabels(xticklabels)
    ax1.set_xlabel(r'$\tau_i$')
    ax1.set_ylabel(r'$\alpha_i$')
    ax1.grid(False)
    ax1.legend()

    fig.show()
    return fig





