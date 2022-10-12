"""
Collection of function to pre-process the master curve and perform the Prony 
series parameter identification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, nnls
from . import shift

"""
--------------------------------------------------------------------------------
Prony series - Domain independent functions
--------------------------------------------------------------------------------
"""
def discretize(df_master, window='round', nprony=0):
    """
    Discretizes relaxation times over time or frequency axis.

    Discrete relaxation times are required for Prony parameter curve fitting
    routine. This function spaces the relaxation times over the experimental characterization window.

    Parameters
    ----------
    df_master : pandas.DataFrame
         Contains the master curve data.
    window : {'round', 'exact', 'min'}
        Defines the location of the discretization of the relaxation times.
        - 'exact' : Use whole window of the experimental data and logarithmically 
        space the relaxation times inbetween.
        - 'round' : Round the minimum and maximum values of the experimental data
        to the nearest base 10 number and logarithmically space the 
        remaining relaxation times inbetween the rounded numbers
        - 'min'   : Position of relaxation times is optimized during minimization
        routine to reduce the number of Prony terms.
    nprony : numeric, default = 0
        Number of Prony terms to be used for the discretization. The number
        of Prony terms and the number of relaxation times is equal. If no number
        or 0 is specified, the default behavior of one Prony term per decade is
        used to automatically calculate the number of Prony terms.

    Returns
    -------
    df_dis : pandas.DataFrame
        Contains discrete point, equal to the relaxation times, of the 
        master curve data (df_master).

    References
    ----------
    Kraus, M. A., and M. Niederwald. "Generalized collocation method using 
    Stiffness matrices in the context of the Theory of Linear viscoelasticity 
    (GUSTL)." Technische Mechanik-European Journal of Engineering Mechanics 
    37.1 (2017): 82-106.
    """
    modul = df_master.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)
    stor_filt = '{}_stor_filt'.format(modul)
    loss_filt = '{}_loss_filt'.format(modul)
    relax_filt = '{}_relax_filt'.format(modul)

    #Get relaxation times
    a = 1 #[Tschoegl 1989]
    #omega = (1/(a*tau)) #[Kraus 2017, Eq. 25]
    _tau = 1/(a*df_master['omega'])

    #Window Time Domain
    if df_master.domain == 'freq':
        exp_inf = int(np.floor(np.log10(_tau.iloc[0])))  #highest time domain exponent
        exp_0 = int(np.ceil(np.log10(_tau.iloc[-1])))    #lowest time domain exponent
        val_inf = _tau.iloc[0]
        val_0 = _tau.iloc[-1]
    elif df_master.domain == 'time':
        exp_inf = int(np.floor(np.log10(_tau.iloc[-1]))) #highest time domain exponent
        exp_0 = int(np.ceil(np.log10(_tau.iloc[0])))     #lowest time domain exponent
        val_inf = _tau.iloc[-1]
        val_0 = _tau.iloc[0]
    decades = exp_inf - exp_0
    
    #Space evenly on a log-scale in time domain
    if nprony == 0:
        nprony = exp_inf - exp_0 + 1 #One prony term per decade 
    if window == 'round':
        tau = np.flip(np.geomspace(float(10**exp_0), float(10**exp_inf), nprony)) 
    elif window == 'exact':
        tau = np.flip(np.geomspace(val_0, val_inf, nprony)) 
    elif window == 'min':
        tau = np.flip(np.geomspace(val_0, val_inf, nprony+2))[1:-1]

    #Get dataframe with discretized values
    omega_dis = (1/(a*tau)) #[Kraus 2017, Eq. 25]
    freq_dis = omega_dis/(2*np.pi) #Convert to cycles per second [Hz] 
    t_dis = 1/freq_dis

    if df_master.domain == 'freq':
        #Interpolate E_stor and E_loss at discretization poins
        E_stor_dis = np.interp(freq_dis, df_master['f'], df_master[stor_filt])
        E_loss_dis = np.interp(freq_dis, df_master['f'], df_master[loss_filt])

        #Estimate instantenous (E_0) and equilibrium (E_inf) modulus
        E_0 = df_master[stor_filt].iloc[-1]
        E_inf = df_master[stor_filt].iloc[0]

        #Assembly data frame
        df_dis = pd.DataFrame([freq_dis, E_stor_dis, E_loss_dis, omega_dis, tau]).T
        df_dis.columns = ['f', stor, loss, 'omega', 'tau_i']

    elif df_master.domain == 'time':
        #Interpolate E_stor and E_loss at discretization poins
        E_relax_dis = np.interp(t_dis, df_master['t'], df_master[relax_filt])

        #Estimate instantenous (E_0) and equilibrium (E_inf) modulus
        E_0 = df_master[relax_filt].iloc[0]
        E_inf = df_master[relax_filt].iloc[-1]

        #Assembly data frame
        df_dis = pd.DataFrame([tau, t_dis, E_relax_dis, omega_dis, freq_dis]).T
        df_dis.columns = ['tau_i', 't', relax, 'omega', 'f']

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


def plot_dis(df_master, df_dis, units):
    """
    Plot relaxation times on top of master curve.

    Parameters
    ----------
    df_master : pandas.DataFrame
        Contains the master curve data.
    df_dis : pandas.DataFrame
        Contains the discrete relaxation times and corresponding data.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Plot showing the relaxation times on top of the master curve.
    """
    modul = df_master.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)

    if df_master.domain == 'freq':
        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=[stor, loss], 
            ax=ax1, logx=True, logy=True, color=['C0', 'C1'], alpha=0.5)
        df_dis.plot(x='f', y=[stor, loss], label=['tau_i', 'tau_i'], ax=ax1, 
            logx=True, logy=True, ls='', marker='o', color=['C0', 'C1'])
        ax1.set_xlabel('Frequency ({})'.format(units['f']))
        ax1.set_ylabel('Storage and loss modulus ({})'.format(units[stor]))
        ax1.legend()
        fig.show()
        return fig
    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=[relax], 
            ax=ax1, logx=True, logy=True, color=['k'])
        df_dis.plot(x='t', y=[relax], label = ['tau_i'], 
            ax=ax1, logx=True, logy=True, ls='', marker='o', color=['red'])
        ax1.set_xlabel('Time ({})'.format(units['t']))
        ax1.set_ylabel('Relaxation modulus ({})'.format(units[relax]))
        ax1.legend()
        fig.show()
        return fig


def ls_res(func):
    """
    Wrapper function that calculates the least squares residual.

    Parameters
    ----------
    func : function
         Time domain: prony.E_relax_norm
         Frequency domain: prony.E_freq_norm

    Returns
    -------
    residual : function
        Calculates least squares residual for specified domain.
    """
    def residual(alpha_i, tau_i, E_meas_norm, tf_meas):
        """
        Calculate least squares resdiual.

        Parameters
        ----------
        alpha_i : array-like
            Normalized relaxation moduli (unitless).
        tau_i : array-like
            relaxation times in s.
        E_meas_norm : array-like
            Normalized modulus from experimental measurement data.
        tf_meas : array-like
            Time domain: time data of measurements in s.
            Frequency domain: frequency data of measurements in Hz.

        Returns
        -------
        numeric
            Least squares residual of measurement data and curve fit data.
        """
        return np.sum((E_meas_norm - func(tf_meas, alpha_i, tau_i))**2)
    return residual


def split_x0(func):
    """
    Wrapper that splits array x0 of the minimization routine into two arrays.

    Splits the the first argument x0 into two arrays alpha_i and tau_i and 
    forwards both arrays to the called function. A single array x0 is necessary 
    to optimize both alpha_i and tau_i at the same time. However, typically, 
    only alpha_i is optimized and tau_i is kept constant. This wrapper allows 
    to use the same function in both scenarios.

    Parameters
    ----------
    func : function
        Function that calculates least squares residual.

    Returns
    -------
    split : function

    See also
    --------
    prony.ls_res : Function to be wrapped during minimization of Prony terms.
    """
    def split(*args):
        alpha_i = args[0][0:int(args[0].shape[0]/2)]
        tau_i = args[0][int(args[0].shape[0]/2):]
        return func(alpha_i, tau_i, args[1], args[2])
    return split


"""
--------------------------------------------------------------------------------
Prony series - Time domain
--------------------------------------------------------------------------------
"""
def E_relax_norm(time, alpha_i, tau_i):
    """
    Calculate normalized relaxation modulus values.

    Parameters
    ----------
    time : array-like
        Time in s.
    alpha_i : array-like
        Normalized relaxation moduli (unitless).
    tau_i : array-like
        relaxation times in s.

    Returns
    -------
    numpy.ndarray
        Relaxation modulus values.
    """
    #Loop implementation
    #-------------------
    #y = np.zeros(time.shape[0])
    #for i, t in enumerate(time):
    #    y[i] = E_0 * (1 - np.sum(alpha_i*(1-np.exp(-t/tau_i))))
    #return y
    #-----------------------------
    #Linear algebra implementation
    return 1-np.sum(alpha_i) + np.dot(alpha_i, np.exp(-time/tau_i[:,None]))


def fit_time(df_dis, df_master, opt=False):
    """
    Fit Prony series parameter in time domain.

    A least-squares minimization is performed using the L-BFGS-B method from 
    the scipy package. The implementation is similar to the optimization problem described by [1] for a homogenous distribution of discrete times. 

    Parameters
    ----------
    df_dis : pandas.DataFrame
        Contains the discrete relaxation times and corresponding data.
    df_master : pandas.DataFrame
        Contains the master curve data.
    opt : bool, default = False
        Flag indicates wether the Prony term minimization routine should be 
        executed or not.

    Returns
    -------
        prony : dict
            Contains the Prony series parameters of the fit.

    References
    ----------
    [1] Barrientos, E., Pelayo, F., Noriega, Á. et al. Optimal discrete-time 
    Prony series fitting method for viscoelastic materials. Mech Time-Depend 
    Mater 23, 193-206 (2019). https://doi.org/10.1007/s11043-018-9394-z
    """
    m = df_dis.modul

    #Initial guess: alpha_i = 1
    alpha_i = np.ones(df_dis['tau_i'].values.shape)
    tau_i = df_dis['tau_i'].values

    #Get measurement data and normalize modul
    E_meas_norm = df_master['{}_relax_filt'.format(m)].values / df_dis.E_0
    time_meas = df_master['t'].values

    #Define bounds
    bnd_a = ((0,1),)*alpha_i.shape[0]

    #Perform minimization to obtain alpha_i
    res = minimize(ls_res(E_relax_norm), alpha_i, 
        args=(tau_i, E_meas_norm, time_meas), method='L-BFGS-B', bounds=bnd_a)
    alpha_i = res.x

    #Use initial fit and try to optimize both alpha_i and tau_i
    if opt:
        #Stack alpha_i and tau_i into single array
        x0 = np.hstack((alpha_i, tau_i))

        #Define bounds
        tau_max = 1/(2*np.pi*df_dis.f_min)
        tau_min = 1/(2*np.pi*df_dis.f_max)
        bnd_t = ((tau_min, tau_max),)*alpha_i.shape[0]
        bnd = bnd_a + bnd_t

        #Find optimal Prony terms
        res = minimize(split_x0(ls_res(E_relax_norm)), x0, 
            args=(E_meas_norm, time_meas), method='L-BFGS-B' , bounds=bnd) 

        #Print success of optimization
        if res.success:
            msg = 'Prony series fit N = {:02d}: Convergence criterion reached!'
            print(msg.format(alpha_i.shape[0]))
        else:
            msg = 'Prony series fit N = {:02d}: Convergence criterion not reached!'
            print(msg.format(alpha_i.shape[0]))

        #Store Prony terms in dataframe
        alpha_i = res.x[0:int(res.x.shape[0]/2)]
        df_dis['tau_i'] = res.x[int(res.x.shape[0]/2):]
     
    #Ensure that Sum(alpha_i) <= 1
    if alpha_i.sum() > 1:
        df_dis['alpha_i'] = 1/alpha_i.sum()*alpha_i #normalize to 1
    else:
        df_dis['alpha_i'] = alpha_i

    #Store Prony terms in dataframe
    df_prony = df_dis[['tau_i', 'alpha_i']].copy()
    df_prony = df_prony.iloc[::-1].reset_index(drop=True)
    df_prony.index += 1 
    df_prony['{}_0'.format(m)] = df_dis.E_0
    df_prony['{}_i'.format(m)] = df_dis.E_0 * df_prony['alpha_i']
    df_prony.RefT = df_dis.RefT

    #Store Prony parameters in dictionary
    prony = {'E_0':df_dis.E_0, 'df_terms':df_prony, 'f_min':df_dis.f_min, 
        'f_max':df_dis.f_max, 'label':'equi.', 'err' : res.fun, 
        'decades':df_dis.decades, 'modul':m}
    return prony


"""
--------------------------------------------------------------------------------
Prony series - Frequency domain
--------------------------------------------------------------------------------
"""
def E_freq_norm(omega, alpha_i, tau_i):
    """
    Calculate normalized storage and loss modulus values.

    Parameters
    ----------
    omega : array-like
        Angular frequency in rad/s.
    alpha_i : array-like
        Normalized relaxation moduli (unitless).
    tau_i : array-like
        relaxation times in s.

    Returns
    -------
    numpy.ndarray
        Concatenated array of normalized storage and loss modulus values.
    """
    A = (omega*tau_i[:,None])
    A2 = A**2
    E_stor = 1-np.sum(alpha_i) + np.dot(alpha_i, A2/(A2+1))
    E_loss = np.dot(alpha_i, A/(A2+1))
    return np.concatenate((E_stor, E_loss))


def fit_freq(df_dis, df_master=None, opt=False):
    """
    Fit Prony series parameter in frequency domain.

    A generalized collocation method using stiffness matrices is used [1]. 
    This methods utilizes both the storage and loss modulus master curves to 
    estimate the Prony series parameters.

    Parameters
    ----------
    df_dis : pandas.DataFrame
        Contains the discrete relaxation times and corresponding data.
    df_master : pandas.DataFrame, default = None
        Contains the master curve data. Only required for Prone term 
        minimization routine (opt = True).
    opt : bool, default = False
        Flag indicates wether the Prony term minimization routine should be 
        executed or not.

    Returns
    -------
        prony : dict
            Contains the Prony series parameters of the fit.

    References
    ----------
    [1] Kraus, M. A., and M. Niederwald. "Generalized collocation method using 
    Stiffness matrices in the context of the Theory of Linear viscoelasticity 
    (GUSTL)." Technische Mechanik-European Journal of Engineering Mechanics 
    37.1 (2017): 82-106.
    """
    modul = df_dis.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    inst_mod = '{}_0'.format(modul)
    rel_mod = '{}_i'.format(modul)

    #Assembly 'K_global' matrix [Kraus 2017, Eq. 22]
    N = df_dis.nprony 
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
    E = np.concatenate((df_dis[stor]/(E_0-E_inf), 
                        df_dis[loss]/(E_0-E_inf), 
                        np.array([1])))

    #Solve equation system
    alpha_i, err = nnls(K_global, E)

    #Use initial fit and try to optimize both alpha_i and tau_i
    if opt:
        #Get measurement data
        E_freq_meas = np.concatenate((df_master[stor]/E_0, 
                                      df_master[loss]/E_0))
        omega_meas = df_master['omega'].values

        #Get Prony series
        tau_i = df_dis['tau_i']
        x0 = np.hstack((alpha_i, tau_i))

        #Define bounds
        tau_max = 1/(2*np.pi*df_dis.f_min)
        tau_min = 1/(2*np.pi*df_dis.f_max)
        bnd_t = ((tau_min, tau_max),)*alpha_i.shape[0]
        bnd_a = ((0,1),)*alpha_i.shape[0]
        bnd = bnd_a + bnd_t

        #Find optimal Prony terms
        res = minimize(split_x0(ls_res(E_freq_norm)), x0, 
            args=(E_freq_meas, omega_meas), bounds=bnd,  method='L-BFGS-B', 
            options={'maxls' : 200})
        
        #Store Prony terms in dataframe
        alpha_i = res.x[0:int(res.x.shape[0]/2)]
        df_dis['tau_i'] = res.x[int(res.x.shape[0]/2):]
        err = res.fun

        #Print success of optimization
        if res.success:
            _msg = 'Prony series N = {:02d}: Convergence criterion reached!'
            print(_msg.format(alpha_i.shape[0]))
        else:
            _msg = 'Prony series N = {:02d}: Convergence criterion not reached!'
            print(_msg.format(alpha_i.shape[0]))

    #Ensure that Sum(alpha_i) <= 1
    if alpha_i.sum() >= 1:
        df_dis['alpha_i'] = 1/alpha_i.sum()*alpha_i #normalize to 1
    else:
        df_dis['alpha_i'] = alpha_i

    #Store Prony terms in dataframe
    df_prony = df_dis[['tau_i', 'alpha_i']].copy()
    df_prony = df_prony.iloc[::-1].reset_index(drop=True)
    df_prony.index += 1 
    df_prony[inst_mod] = E_0
    df_prony[rel_mod] = E_0 * df_prony['alpha_i']
    df_prony.RefT = df_dis.RefT

    #Store Prony parameters in dictionary
    prony = {'E_0':E_0, 'df_terms':df_prony, 'f_min':df_dis.f_min, 
        'f_max':df_dis.f_max, 'label':'equi.', 'err' : err, 
        'decades':df_dis.decades, 'modul':modul}
    return prony




"""
--------------------------------------------------------------------------------
Generalized Maxwell model
--------------------------------------------------------------------------------
"""
def calc_GMaxw(E_0, df_terms, f_min, f_max, decades, modul, **kwargs):
    """
    Calculate the Generalized Maxwell model data from the Prony series parameter.

    Parameters
    ----------
    E_0 : numeric
        Instantaneous storage modulus. Same variable name is used for either 
        tensile (E_0) or shear (G_0) loading.
    df_terms : pandas.DataFrame
        Contains the Prony series parameters tau_i and alpha_i.
    f_min : numeric
        Lower bound frequency for calculation of physical quanitities.
    f_max : numeric
        Upper bound frequency for calculation of physical quanitities.
    decades : integer
        Number of decades spanning the frequency window. Is used to calculate
        the necessary number of data points spanning the frequency range for
        an appropriate resolution.
    modul : {'E', 'G'}
        Indicates wether tensile ('E') or shear ('G') modulus data are provided.

    Returns
    -------
    df_GMaxw : pandas.DataFrame
        Contains the calculated Generalized Maxwell model data for the fitted
        Prony series parameters with the specified boundaries and parameters.   
    """
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    comp = '{}_comp'.format(modul)
    relax = '{}_relax'.format(modul)

    alpha_i = df_terms['alpha_i'].values
    tau_i = df_terms['tau_i'].values

    #Define angular frequency range for plotting
    omega_min = 2*np.pi*f_min
    omega_max = 2*np.pi*f_max
    omega_len = 10*decades #number of datapoints along x-axis (10 per decade)

    #Define dataframe
    df_GMaxw = pd.DataFrame(np.zeros((omega_len, 8)), 
        columns=(['f', 'omega', stor, loss, comp, 'tan_del', 't', relax]))

    #Fill frequency and time axis
    df_GMaxw['omega'] = np.geomspace(omega_min, omega_max, omega_len)
    df_GMaxw['f'] = df_GMaxw['omega']/(2*np.pi)
    df_GMaxw['t'] = 1/df_GMaxw['f'] 
    E_inf = E_0*(1-np.sum(alpha_i))
    A = (df_GMaxw['omega'].values*tau_i[:,None])
    A2 = (df_GMaxw['omega'].values*tau_i[:,None])**2
    df_GMaxw[stor] = E_inf + np.dot(E_0*alpha_i, A2/(A2+1))
    df_GMaxw[loss] = np.dot(E_0*alpha_i, A/(A2+1))
    df_GMaxw[comp] = (df_GMaxw[stor]**2 + df_GMaxw[loss]**2)**0.5  
    df_GMaxw['tan_del'] = df_GMaxw[loss]/df_GMaxw[stor]

    #Calculate time domain
    df_GMaxw[relax] =  E_0 * E_relax_norm(df_GMaxw['t'].values, alpha_i, tau_i)

    #Define attributes
    df_GMaxw.modul = modul
    return df_GMaxw


def GMaxw_temp(shift_func, df_GMaxw, df_coeff, df_aT, freq = [1E-8, 1E-4, 1E0, 1E4]):
    """
    Calculate Gen. Maxwell model for different loading frequencies and temperatures.

    This function showcases the temperature and rate-dependence of the visco-
    elastic material. The specified shift function is used to calculate
    the material response at different temperatures and different loading
    rates.

    Parameters
    ----------
    shift_func : {'WLF', 'D4', 'D3', 'D2', 'D1'}
        Specifies the shift function to be used for calculations.
    df_GMaxw : pandas.DataFrame
        Contains the Generalized Maxwell model data for the reference 
        temperature at different loading rates.
    df_coeff : pandas.DataFrame
        Contains the coefficients and parameters for the specified shift 
        function. 
    df_aT : pandas.DataFrame
        Contains the shift factors. The shift factors are used to identify
        the Temperature range for the calculation.
    freq : array-like, default = [1E-8, 1E-4, 1E0, 1E4]
        Loading frequencies for which the calculations are performed. 

    Returns
    -------
    df_GMaxw_temp
        Contains the Generalized Maxwell model data for a wide range of 
        temperatures at the specified frequencies.

    See also
    --------
    shift.fit_WLF : Returns WLF shift functions.
    shift.fit_poly : Returns polynomial shift functions of degree 1 to 4.
    """
    modul = df_GMaxw.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)

    df_GMaxw_temp = pd.DataFrame()
    T_min = int(df_aT['T'].min())
    T_max = int(df_aT['T'].max())

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
                E_stor = np.interp(f, f_shift, df_GMaxw[stor])
                E_loss = np.interp(f, f_shift, df_GMaxw[loss])
                E_relax = np.interp(f, f_shift, df_GMaxw[relax])
                tan_del = np.interp(f, f_shift, df_GMaxw['tan_del'])
                df = pd.DataFrame([[f, T, E_stor, E_loss, tan_del, E_relax]], 
                    columns=['f', 'T', stor, loss, 'tan_del', relax])
                df_GMaxw_temp = pd.concat([df_GMaxw_temp, df])
            else:
                continue
            
    df_GMaxw_temp = df_GMaxw_temp.reset_index(drop=True)
    df_GMaxw_temp.modul = modul
    return df_GMaxw_temp


def plot_GMaxw(df_GMaxw, units):
    """
    Plot Generalized Maxwell model data for the reference temperature.

    Parameters
    ----------
    df_GMaxw : pandas.DataFrame
        Contains the Generalized Maxwell model data for the reference 
        temperature at different loading rates.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Plot of calculated storage, loss, and relaxation modulus.
    """
    modul = df_GMaxw.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)

    fig1, ax1 = plt.subplots() 
    df_GMaxw.plot(x='f', y=[stor],  
            ax=ax1, logx=True, logy=True, ls='-', lw=2, color=['C0'])
    df_GMaxw.plot(x='f', y=[loss],  
            ax=ax1, logx=True, logy=True, ls=':', lw=2, color=['C1'])
    df_GMaxw.plot(x='f', y=[relax], 
            ax=ax1, logx=True, logy=True, ls='--', lw=2, color=['C2'])
    ax1.set_xlabel('Frequency ({})'.format(units['f']))
    ax1.set_ylabel('Relaxation, storage and \n loss modulus ({})'.format(units[stor]))
    ax1.set_ylim(min(df_GMaxw[stor].min(), df_GMaxw[loss].min()), )
    fig1.show()
    return fig1 
    

def plot_GMaxw_temp(df_GMaxw_temp, units):
    """
    Plot Generalized Maxwell model data for varies temperature and frequencies.

    Parameters
    ----------
    df_GMaxw_temp : pandas.DataFrame
        Contains the Generalized Maxwell model data for various
        temperatures and different loading rates.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Plot of showing the temperature and rate dependence of the  storage, 
        loss, and relaxation modulus.
    """
    modul = df_GMaxw_temp.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)

    fig, ax1 = plt.subplots()
    for i, (f, df) in enumerate(df_GMaxw_temp.groupby('f')):
        df.plot(y=stor, x='T', ls='-', ax=ax1, logy=True, label='f = {:.0e} Hz'.format(f), 
            c='C{}'.format(i))
        df.plot(y=loss, x='T', ls=':', ax=ax1, logy=True, label='', c='C{}'.format(i))
        df.plot(y=relax, x='T', ls='--', ax=ax1, logy=True, c='C{}'.format(i), label='') 
    ax1.set_xlabel('Temperature ({})'.format(units['T']))
    ax1.set_ylabel('Relaxation, storage and \n loss modulus ({})'.format(units[stor]))
    ax1.set_ylim(min(df_GMaxw[stor].min(), df_GMaxw[loss].min()), )
    ax1.legend()
    fig.show()
    return fig


def plot_param(prony_list, labels=None):
    """
    Plot illustrating the Prony series parameters of one or more fits.

    Parameters
    ----------
    prony_list : list
        List of `prony` dictionaries containing the Prony series parameters.
    labels : list of str
        List of strings to be used as legend label names.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Plot showing the relaxation moduli over the relaxation times.

    See also
    --------
    prony.fit : Returns the prony dictionary to be used in prony_list.
    """
    df_list = []
    for i, prony in enumerate(prony_list):
        df = prony['df_terms'][['tau_i', 'alpha_i']].copy()
        df = df.set_index('tau_i')
        if labels:
            df.columns = [labels[i]]
        else:
            df.columns = [prony['label']]
        df_list.append(df)
    df_bar = pd.concat(df_list, axis=1)
    df_bar.sort_index(inplace = True)

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


"""
--------------------------------------------------------------------------------
Prony series & Generalized Maxwell model - Generalized functions
--------------------------------------------------------------------------------
"""
def fit(df_dis, df_master=None, opt=False):
    """
    Generalized function to call the domain dependent curve fitting routine.

    Parameters
    ----------
    df_dis : pandas.DataFrame
        Contains the discrete relaxation times and corresponding data.
    df_master : pandas.DataFrame, default = None
        Contains the master curve data. Not required for the initial fit in 
        the frequency domain (opt = False).
    opt : bool, default = False
        Flag indicates wether the Prony term minimization routine should be 
        executed or not.

    Returns
    -------
    prony : dict
        Contains the Prony series parameters of the fit.
    df_GMaxw : pandas.DataFrame
        Contains the calculated Generalized Maxwell model data for the fitted
        Prony series parameters.
    """
    if df_dis.domain == 'freq':
        prony = fit_freq(df_dis, df_master, opt)
    elif df_dis.domain == 'time':
        prony = fit_time(df_dis, df_master)
    df_GMaxw = calc_GMaxw(**prony)
    return prony, df_GMaxw


def plot_fit(df_master, df_GMaxw, units):
    """
    Plot the master curve and corresponding Prony fit (Gen. Maxwell model).

    Parameters
    ----------
    df_master : pandas.DataFrame
        Contains the master curve data.
    df_GMaxw : pandas.DataFrame
        Contains the calculated Generalized Maxwell model data for the fitted
        Prony series parameters.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Domain dependent plot of master curve and Prony fit.
    """
    modul = df_master.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)

    if df_master.domain == 'freq':
        fig, ax1 = plt.subplots()
        df_master.plot(x='f', y=[stor, loss], 
            ax=ax1, logx=True, logy=True, color=['C0', 'C1'], alpha=0.5, ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='f', y=[stor, loss], 
            ax=ax1, logx=True, logy=True, ls='-', lw=2, color=['C0', 'C1'])
        ax1.set_xlabel('Frequency ({})'.format(units['f']))
        ax1.set_ylabel('Storage and loss modulus ({})'.format(units[stor])) 
        ax1.legend()
        fig.show()
        return fig
    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots()
        df_master.plot(x='t', y=[relax], ax=ax1, logx=True, logy=True, 
            color=['gray'], ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='t', y=[relax], label=['fit'], ax=ax1, 
            logx=True, logy=True, ls='-', lw=2, color=['r'])
        ax1.set_xlabel('Time ({})'.format(units['t']))
        ax1.set_ylabel('Relaxation modulus ({})'.format(units[relax])) 
        ax1.legend()
        fig.show()
        return fig
