"""Collection of functions to load measurement data and prepare pandas 
dataframes for further processing.
"""


import numpy as np
import pandas as pd
import io


def get_sets(df_raw, num=0):
    """Group raw DMTA data into measurements conducted at same temperature.

        num: Number of measurements at one temperature level
    
    If num is not provided the number of measurements is evaluated based on 
    the frequency range and the index of the first occurance of the maximum
    frequency is used to group the data frame.
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

def file(path):
    """Read file from path."""
    with open(path, 'rb') as file:  
        data = file.read() 
    return data

def Eplexor_raw(data, modul):
    """Load raw measurement data from the Eplexor software. The columns are 
    renamed and the individual measurements at different temperatures are
    grouped based on the frequency data.
    """
    df = pd.read_excel(io.BytesIO(data), 'Exported Data', header=[0,1], na_values='---')
    df.dropna(inplace=True)
    df.columns = df.columns.droplevel(1)

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

    return df_raw, arr_RefT


def user_raw(data, domain, modul):
    """Load raw measurement data from user instrument. The columns are 
    renamed and the individual measurements at different temperatures are
    grouped based on user data.
    """
    df_raw = pd.read_csv(io.BytesIO(data))
    df_raw.dropna(inplace=True)
    df_raw.columns = df_raw.columns.str.replace(' ', '')

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

    arr_RefT = df_raw.groupby('Set')['T'].mean().round()
    df_raw = df_raw.join(arr_RefT, on='Set', rsuffix='_round')
    df_raw.domain = domain
    df_raw.modul = modul
    
    return df_raw, arr_RefT


def Eplexor_master(data, modul):
    """Load master curve data from the Eplexor software. The columns are 
    renamed and the shift factors as well as the WLF shift function parameters 
    are extracted from the excel file.
    """
    df = pd.read_excel(io.BytesIO(data), 'Shiftlist',header=[0,1,2])
    df_master_raw = pd.read_excel(io.BytesIO(data), 'Exported Data', 
        header=[0,1], na_values='---')
    df_master_raw.dropna(inplace=True)

    #Prep Shift factors into df
    RefT = float(df.columns.values[1][0][:-3])
    C1 = float(df.columns.values[1][1][5:])
    C2 = float(df.columns.values[2][1][5:-2])
    WLF = [RefT, C1, C2]

    df.columns = ['Temp', 'log_aT', 'DEL']
    df.drop(['DEL'], axis = 1, inplace = True)
    df_aT = df.round({'Temp': 0})

    #Put fitted WLF shift function in df
    df_WLF = pd.DataFrame(data = WLF).T #
    df_WLF.columns = ['RefT', 'C1', 'C2']

    #Prep Master curve data into df
    df_master_raw.columns = df_master_raw.columns.droplevel(1)
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

    return df_master, df_aT, df_WLF


def user_master(data, domain, RefT, modul):
    """Load master curve data from user instrument. The columns are 
    renamed and additional time and frequency quantities calculated.
    """
    df_master = pd.read_csv(io.BytesIO(data))
    df_master.dropna(inplace=True)
    df_master.columns = df_master.columns.str.replace(' ', '')

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

    df_master.RefT = RefT
    df_master.domain = domain
    df_master.modul = modul

    return df_master


def user_shift(data_shift):
    """Load user provided shift factors."""
    df_aT = pd.read_csv(io.BytesIO(data_shift))
    df_aT.rename(columns = {'Temp':'Temp', 'log_aT':'log_aT'}, inplace=True, 
        errors='raise')

    return df_aT



