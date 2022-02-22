"""
Collection of functions to load measurement data and prepare pandas 
dataframes for further processing.
"""

import numpy as np
import pandas as pd
import io


def conventions(modul):
    """
    Create dictionary with unit conventions for expected physical quantities.

    Parameters
    ----------
    modul : {'E', 'G'}
        Indicates wether tensile ('E') or shear ('G') modulus data are provided.

    Returns
    -------
    conv : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    Notes
    -----
    Tensile moduli are denoted as 'E' and shear moduli  are denoted as 'G'. 
    Only the tensile moduli are summarized in the table below. For shear modulus 
    data, replace 'E' with 'G', e.g. 'E_relax' -> 'G_relax'.

    | Physical quantity        | Variable   | Unit                  |
    | :---------------------   | :--------- | :-------------------: |
    | Relaxation modulus:      | `E_relax`  | `[Pa, kPa, MPa, GPa]` |
    | Storage modulus:         | `E_stor`   | `[Pa, kPa, MPa, GPa]` |
    | Loss modulus:            | `E_loss`   | `[Pa, kPa, MPa, GPa]` |
    | Complex modulus:         | `E_comp`   | `[Pa, kPa, MPa, GPa]` |
    | Loss factor:             | `tan_del`  | `-`                   |
    | Instantaneous modulus:   | `E_0`      | `[Pa, kPa, MPa, GPa]` |
    | Equilibrium modulus:     | `E_inf`    | `[Pa, kPa, MPa, GPa]` |
    | Angular frequency:       | `omega`    | `rad/s`               |
    | Frequency:               | `f`        | `Hz`                  |
    | Time:                    | `t`        | `s`                   |
    | Temperature:             | `T`        | `°C`                  |
    | Relaxation times:        | `tau_i`    | `s`                   |
    | Relaxation moduli:       | `E_i`      | `[Pa, kPa, MPa, GPa]` |
    | Norm. relaxation moduli: | `alpha_i`  | `-`                   |
    | Shift factor:            | `log_aT`   | `-`                   |
    """
    m = modul
    conv = {'f' : ['Hz'],
            't' : ['s'],
            'omega' : ['rad/s'],
            'T' : ['°C', 'C'],
            'tau_i' : ['s'],
            'alpha_i': ['-', ''],
            'tan_del': ['-', ''],
            'log_aT': ['-', ''],
            '{}_relax'.format(m): ['Pa', 'kPa', 'MPa', 'GPa'],
            '{}_stor'.format(m):  ['Pa', 'kPa', 'MPa', 'GPa'],
            '{}_loss'.format(m):  ['Pa', 'kPa', 'MPa', 'GPa'],
            '{}_comp'.format(m):  ['Pa', 'kPa', 'MPa', 'GPa'],
            '{}_0'.format(m):     ['Pa', 'kPa', 'MPa', 'GPa'],
            '{}_inf'.format(m):   ['Pa', 'kPa', 'MPa', 'GPa'],
            '{}_i'.format(m):   ['Pa', 'kPa', 'MPa', 'GPa'],
            '{}_relax_filt'.format(m): ['Pa', 'kPa', 'MPa', 'GPa'],
            '{}_stor_filt'.format(m):  ['Pa', 'kPa', 'MPa', 'GPa'],
            '{}_loss_filt'.format(m):  ['Pa', 'kPa', 'MPa', 'GPa'],
            'Set' : ['-', ''],
            'RefT' : ['°C', 'C'],
            'C1' : ['-'],
            'C2' : ['°C', 'C']}
    return conv


def file(path):
    """
    Read data from file.
    
    Parameters
    ----------
    path : str
        Filepath to the file that is being read.

    Returns
    -------
    data : bytes
        File content.

    Notes
    -----
    This function is included to simplify the file upload within the interactive
    module where graphical dashboards are hosted on a webserver. 
    """
    with open(path, 'rb') as file:  
        data = file.read() 
    return data


def prep_csv(data):
    """
    Load csv files into pandas dataframe and prepare data.
    
    The function removes NANs from the input data, and identifies the names of the
    pysical quantities and units from the file header. The file header must
    consit of two rows. The first row provides the name of the physical quantity
    and the second row provides the corresponding physical unit. 

    Parameters
    ----------
    data : bytes
        CSV file content to be loaded into a pandas.DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Contains the csv file data.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    See Also:
    ---------
    load.file : Returns bytes object to be used as input parameter `data`.
    """
    df = pd.read_csv(io.BytesIO(data), header=[0,1])
    df.dropna(inplace=True)
    df.rename(columns=lambda s: s.strip(), inplace=True)
    units = dict(zip(df.columns.get_level_values(0).values, 
                     df.columns.get_level_values(1).values))
    df.columns = df.columns.droplevel(1)
    return df, units


def prep_excel(data):
    """
    Load Excel files into pandas dataframe and prepare data.
    
    The function removes NANs from the input data, and identifies the names of 
    the pysical quantities and units from the file header. The file header must
    consit of two rows. The first row provides the name of the physical quantity
    and the second row provides the corresponding physical unit. 

    Parameters
    ----------
    data : bytes
        Excel file content to be loaded into a pandas.DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Contains the Excel file data.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    See Also:
    ---------
    load.file : Returns bytes object to be used as input parameter `data`.
    """
    df = pd.read_excel(io.BytesIO(data), 'Exported Data', header=[0,1], na_values='---')
    df.dropna(inplace=True)
    units = dict(zip(df.columns.get_level_values(0).values, 
                     df.columns.get_level_values(1).values))
    df.columns = df.columns.droplevel(1)
    return df, units
    
    
def Eplexor_raw(data, modul):
    """
    Load raw measurement data from an EPLEXOR Excel file. 
    
    This function loads raw Dynamic Mechanical Thermal Analysis (DMTA) measurement
    files created by a Netzsch Gabo DMA EPLEXOR. Use the `Excel Export!` 
    feature of the Eplexor software with the default template to create the 
    Excel file. The column headers are renamed to follow the conventions used in 
    this modul. The names of the physical quantities and units are checked
    against the conventions used in this module. the individual measurements at 
    different temperatures are grouped into data sets based on the frequency 
    range of the individual measurement sets.

    Parameters
    ----------
    data : bytes
        Excel file content loaded into pandas.DataFrame.

    modul : {'E', 'G'}
        Indicates wether tensile ('E') or shear ('G') modulus data are provided.

    Returns
    -------
    df_raw : pandas.DataFrame
        Contains the processed raw measurement data.
    arr_RefT : pandas.Series
        Contains the temperature levels of the measurement sets.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    Raises
    ------
    KeyError
        If the file header variable names or units do not follow the 
        conventions used in this module.

    See also
    --------
    load.conventions : Summarizes conventions for variable names and units.
    load.file : Returns bytes object to be used as parameter `data`.
    load.check_units : Raises KeyError if units don't comply with conventions.
    """
    df, units = prep_excel(data)
    
    df.rename(columns={"f":"f_set", 
        "{}'".format(modul)  :'{}_stor'.format(modul), 
        "{}''".format(modul) :'{}_loss'.format(modul), 
        "|{}*|".format(modul):'{}_comp'.format(modul), 
        "tan delta":'tan_del'}, inplace=True, errors='raise')
    df_raw = df[['f_set', 
        '{}_stor'.format(modul), 
        '{}_loss'.format(modul), 
        '{}_comp'.format(modul), 
        'tan_del', 'T']].copy()
    #df_raw['omega'] = 2*np.pi*df_raw['f']
    #df_raw['t'] = 1/df_raw['omega']

    df_raw = get_sets(df_raw, num=0)
    arr_RefT = df_raw.groupby('Set')['T'].mean().round()
    df_raw = df_raw.join(arr_RefT, on='Set', rsuffix='_round')
    df_raw.domain = 'freq'
    df_raw.modul = modul

    check_units(units, modul)
    units = get_units(units, modul, df_raw.domain, True)
    return df_raw, arr_RefT, units


def user_raw(data, domain, modul):
    """
    Load raw measurement data from csv file. 
    
    This function loads raw data of viscoelastic material characterizations
    conducted at one or more temperature levels. Either tensile or shear modulus
    data from characterizations performed in either the frequency or time domain 
    are accepted. The file header must consit of two rows. The first row provides 
    the name of the physical quantity and the second row provides the 
    corresponding physical unit. The column headers are checked against the 
    conventions used in this modul. Details on the variable names, units, and 
    file header names are provided in the Notes section below.

    Parameters
    ----------
    data : bytes
        Excel file content.

    domain : {'freq', 'time'}
        Defines wether frequency domain ('freq') or time domain ('time') input 
        data are provided.

    modul : {'E', 'G'}
        Indicates wether tensile ('E') or shear ('G') modulus data are provided.

    Returns
    -------
    df_raw : pandas.DataFrame
        Contains the preprocessed raw measurement data.
    arr_RefT : pandas.Series
        Contains the temperature levels of the measurement sets.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    Raises
    ------
    KeyError
        If the file header variable names or units do not follow the 
        conventions used in this module.

    See also
    --------
    load.conventions : Summarizes conventions for variable names and units.
    load.file : Returns bytes object to be used as parameter `data`.
    load.check_units : Raises KeyError if units don't comply with conventions.

    Notes
    -----
    Dependent on the performed material characterization, either tensile or 
    shear modulus values in either the time or frequency domain must be provided.

    Tensile moduli are denoted as 'E' and shear moduli  are denoted as 'G'.
    Frequency domain data are provided in Hertz: [f] = Hz
    Time domain data are provided in seconds: [t] = s
    Temperature levels are provided in Celsius: [T] = C
    Measurement set identifiers are provided by the column `Set`. In Set, all 
    measurement points at the same temperature level are marked with the same 
    number, e.g. 0, for the first measurement set. The first measurement Set (0) 
    represents the coldest temperature followed by the second set (1) at the 
    next higher temperature level and so forth. 
    
    Example input files can be found here: 
    https://github.com/martin-springer/LinViscoFit/tree/main/examples

    Various examples for file headers:
    | ---------- | --------------------------- | --------------------------- |
    | Domain     | Tensile Modulus             | Shear Modulus               | 
    | :--------- | :-------------------------- | :-------------------------- |
    | Frequency  | `f, E_stor, E_loss, T, Set` | `f, G_stor, G_loss, T, Set` |
    |            | `Hz, MPa, MPa, C, -`        | `Hz, GPa, GPa, C, -`        |
    | ---------- | --------------------------- | --------------------------- |
    | Time :     | `t, E_relax, T, Set`        | `t, G_relax, T, Set`        |
    |            | `s, MPa, C, -`              | `s, GPa, C, -`              |
    | ---------- | --------------------------- | --------------------------- |
    """
    df_raw, units = prep_csv(data)

    if domain == 'freq':
        df_raw.rename(columns={"f":"f_set", 
            "{}_stor".format(modul):'{}_stor'.format(modul), 
            "{}_loss".format(modul):'{}_loss'.format(modul), 
            "T":"T", 'Set':'Set'}, 
            inplace=True, errors='raise')
    elif domain == 'time':
        df_raw.rename(columns={"t":"t_set", 
            "{}_relax".format(modul):'{}_relax'.format(modul), 
            "T":"T", 'Set':'Set'}, inplace=True, errors='raise')
        df_raw['f_set'] = 1/df_raw['t_set']

    check_units(units, modul)
    units = get_units(units, modul, domain)

    arr_RefT = df_raw.groupby('Set')['T'].mean().round()
    df_raw = df_raw.join(arr_RefT, on='Set', rsuffix='_round')
    df_raw.domain = domain
    df_raw.modul = modul
    return df_raw, arr_RefT, units


def Eplexor_master(data, modul):
    """
    Load master curve data from an EPLEXOR excel file. 
    
    This function loads master curve data from an Excel file created with the
    EPLEXOR software. Use the `Excel Export!` feature of the Eplexor software 
    with the default template to create the Excel file. The column headers are 
    renamed to follow the conventions used in this modul. The names of the 
    physical quantities and units are checked against the conventions used in 
    this module.The modulus data, shift factors, and WLF shift function 
    parameters are extracted from the Excel file.

    Parameters
    ----------
    data : bytes
        Excel file content.
    modul : {'E', 'G'}
        Indicates wether tensile ('E') or shear ('G') modulus data are provided.

    Raises
    ------
    KeyError
        If the file header variable names or units do not follow the 
        conventions used in this module.

    Returns
    -------
    df_master : pandas.DataFrame
        Contains the master curve data.
    df_aT : pandas.DataFrame
        Contains the shift factors.
    df_WLF : pandas.DataFrame
        Contains the WLF shift function parameters.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    See also
    --------
    load.conventions : Summarizes conventions for variable names and units.
    load.file : Returns bytes object to be used as parameter `data`.
    load.check_units : Raises KeyError if units don't comply with conventions
    """
    df = pd.read_excel(io.BytesIO(data), 'Shiftlist',header=[0,1,2])

    df_master_raw, units = prep_excel(data)

    #Prep Shift factors into df
    RefT = float(df.columns.values[1][0][:-3])
    C1 = float(df.columns.values[1][1][5:])
    C2 = float(df.columns.values[2][1][5:-2])
    WLF = [RefT, C1, C2]

    df.columns = ['T', 'log_aT', 'DEL']
    df.drop(['DEL'], axis = 1, inplace = True)
    df_aT = df.round({'T': 0})

    #Put fitted WLF shift function in df
    df_WLF = pd.DataFrame(data = WLF).T #
    df_WLF.columns = ['RefT', 'C1', 'C2']

    #Prep Master curve data into df
    df_master_raw.rename(columns={
        "{}'".format(modul):'{}_stor'.format(modul), 
        "{}''".format(modul):'{}_loss'.format(modul), 
        "|{}*|".format(modul):'{}_comp'.format(modul), 
        "tan delta":'tan_del'}, inplace=True, errors='raise')

    df_master = df_master_raw[['f', 
        '{}_stor'.format(modul), 
        '{}_loss'.format(modul), 
        '{}_comp'.format(modul), 
        'tan_del']].copy()
    df_master['omega'] = 2*np.pi*df_master['f']
    df_master['t'] = 1/df_master['f']  
    df_master.RefT = RefT
    df_master.domain = 'freq'
    df_master.modul = modul

    check_units(units, modul)
    units = get_units(units, modul, df_master.domain, True)
    return df_master, df_aT, df_WLF, units


def user_master(data, domain, RefT, modul):
    """
    Load master curve data from csv file. 
    
    Either tensile or shear modulus data from materials characterizations 
    performed in either the frequency or time domain are accepted. 
    The file header must consit of two rows. The first row provides 
    the name of the physical quantity and the second row provides the 
    corresponding physical unit. The column headers are checked against the 
    conventions used in this modul. Details on the variable names, units, and 
    file header names are provided in the Notes section below.

    Parameters
    ----------
    data : bytes
        Excel file content.
    domain : {'freq', 'time'}
        Defines wether frequency domain ('freq') or time domain ('time') input 
        data are provided.
    RefT : int or float
        Reference tempeature of the master curve in Celsius.
    modul : {'E', 'G'}
        Indicates wether tensile ('E') or shear ('G') modulus data are provided.

    Returns
    -------
    df_master : pandas.DataFrame
        Contains the master curve data.
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    Raises
    ------
    KeyError
        If the file header variable names or units do not follow the 
        conventions used in this module.

    See also
    --------
    load.conventions : Summarizes conventions for variable names and units.
    load.file : Returns bytes object to be used as parameter `data`.
    load.check_units : Raises KeyError if units don't comply with conventions.

    Notes
    -----
    Dependent on the performed materials characterization, either tensile or 
    shear modulus values in either the time or frequency domain must be provided.

    Tensile moduli are denoted as 'E' and shear moduli  are denoted as 'G'.
    Frequency domain data are provided in Hertz: [f] = Hz
    Time domain data are provided in seconds: [t] = s
    
    Example input files can be found here: 
    https://github.com/martin-springer/LinViscoFit/tree/main/examples

    Various examples for file headers:
    | ---------- | ------------------- | ------------------- |
    | Domain     | Tensile Modulus     | Shear Modulus       | 
    | :--------- | :------------------ | :------------------ |
    | Frequency  | `f, E_stor, E_loss` | `f, G_stor, G_loss` |
    |            | `Hz, MPa, MPa`      | `Hz, GPa, GPa`      |
    | ---------- | --------------------| --------------------|
    | Time :     | `t, E_relax`        | `t, G_relax`        |
    |            | `s, MPa`            | `s, GPa`            |
    | ---------- | --------------------| --------------------|
    """
    df_master, units = prep_csv(data)

    if domain == 'freq':
        df_master.rename(columns = {'f':'f', 
            '{}_stor'.format(modul):'{}_stor'.format(modul), 
            '{}_loss'.format(modul):'{}_loss'.format(modul)}, 
            inplace=True, errors='raise')
        df_master['omega'] = 2*np.pi*df_master['f']
        df_master['t'] = 1/df_master['f'] 
    elif domain == 'time':
        df_master.rename(columns = {'t':'t', 
            '{}_relax'.format(modul):'{}_relax'.format(modul)}, 
            inplace=True, errors='raise')
        df_master['f'] = 1/df_master['t']
        df_master['omega'] = 2*np.pi*df_master['f']

    check_units(units, modul)
    units = get_units(units, modul, domain)

    df_master.RefT = RefT
    df_master.domain = domain
    df_master.modul = modul
    return df_master, units


def user_shift(data_shift):
    """
    Load user provided shift factors from csv file.

    Two columns need to be provided in the input file. One for the shift 
    factors `log_aT` and one for the corresponding temperatures `T`.
    The file header must consit of two rows. The first row provides 
    the name of the physical quantity and the second row provides the 
    corresponding physical unit. The column headers are checked against the 
    conventions used in this modul. Details on the variable names, units, and 
    file header names are provided in the Notes section below.

    Parameters
    ----------
    data_shift : bytes
        CSV file content.

    Returns
    -------
    df_aT : pandas.DataFrame
        Contains the shift factors and corresponding temperatures.

    Raises
    ------
    KeyError
        If the file header variable names or units do not follow the 
        conventions used in this module.

    See also
    --------
    load.conventions : Summarizes conventions for variable names and units.
    load.file : Returns bytes object to be used as parameter `data`.
    load.check_units : Raises KeyError if units don't comply with conventions.

    Notes
    -----
    Example file header:  1st row -> T, log_aT
                          2nd row -> C, -

    Example input files can be found here: 
    https://github.com/martin-springer/LinViscoFit/tree/main/examples
    """
    df_aT, units = prep_csv(data_shift)
    df_aT.rename(columns = {'T':'T', 'log_aT':'log_aT'}, inplace=True, 
        errors='raise')

    check_units(units)
    return df_aT

def get_sets(df_raw, num=0):
    """
    Group raw data into measurement sets conducted at the same temperature.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        Raw measurement data. 
    num : int, optional 
        Number of measurement points per temperature level.
        Default is 0, which means that the number of measurement points 
        within a set is evaluated based on the provided frequency range 
        of the measurement points. The first occurance of the 
        maximum frequency is used to identify the number of measurement points
        per temperature level.

    Returns
    -------
    df_raw : pandas.DataFrame
        Contains additional column `Set` compared to input data frame.

    Notes
    -----
    This function is intended to be used in combination with input files provided
    by the Eplexor software. Hence, it is limited to measurements in the 
    frequency domain.
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

def get_units(units, modul, domain, Eplexor=False):
    """
    Get the physical units based on the measurement data loading condition
    (tensile or shear) and domain (time or frequency).

    Parameters
    ----------
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.
    modul : {'E', 'G'}
        Indicates wether tensile ('E') or shear ('G') modulus data are provided.
    domain : {'freq', 'time'}
        Defines wether frequency domain ('freq') or time domain ('time') input 
        data are provided.
    Eplexor : bool, default = false
        If input file is Excel file from Eplexor software -> True
        If input file is CSV file from user instrument -> False

    Returns
    -------
    conv : dict of {str : str}
        Conventions for physical quantities as key and corresponding units as 
        itmes for the provided measurement loading conditions and domain.
    """
    conv = conventions(modul)

    if domain == 'freq':
        if Eplexor:
            pascal = units["{}'".format(modul)]
        else:
            pascal = units['{}_stor'.format(modul)]
    elif domain == 'time':
        pascal = units['{}_relax'.format(modul)]

    for key in conv.keys():
        if key[0] == modul:
            conv[key] = pascal
        else:
            conv[key] = conv[key][0]
    return conv


def check_units(units, modul='E'):
    """
    Check that the input file units conform with the conventions used in this 
    modul.

    The input units are compared against the conventions used for the measurement
    loading conditions and domain. For frequency domain data, the storage and 
    loss modulus values must have the same unit.

    Parameters
    ----------
    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.
    modul : {'E', 'G'}
        Indicates wether tensile ('E') or shear ('G') modulus data are provided.

    Raises
    ------
    KeyError
        If the file header variable names or units do not follow the 
        conventions used in this module.
    """
    m = modul
    conv = conventions(m)

    for key in units.keys():
        if key in conv.keys():
            if units[key] not in conv[key]:
                msg = "Wrong unit provided for {}! {} provided, which should be {}."
                raise KeyError(msg.format(key, units[key], conv[key]))           
        if key == '{}_stor'.format(m):
            if units['{}_stor'.format(m)] != units['{}_loss'.format(m)]:
                msg = "Storage modulus ('{0}_stor') and loss modulus('{0}_loss') must have same unit!"
                raise KeyError(msg.format(m))