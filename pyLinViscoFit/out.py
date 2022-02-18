"""
Collection of functions to write pandas dataframes to files or buffer.
"""

import pandas as pd

def add_units(df, units, index_label=None):
    """
    Add second row units header to pandas dataframe containing.

    A multiindex pandas dataframe is created. The first header row contains
    the physical quantities and the second row the corresponding physical units.

    
    Parameters
    ----------
    df : pandas.DataFrame
        Units will be added to this dataframe.

    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    index_label : str
        If index_label is specified, a name for the row index will be included
        in the multiindex column header.

    Returns
    -------
    df_out : pandas.DataFrame
        Dataframe with a multiindex colum header containing the physical units.


    See Also:
    ---------
    out.to_csv : Parent function of current function.
    """
    variables = df.columns.get_level_values(0)
    if index_label == None:
        idx = pd.MultiIndex.from_tuples(zip(variables, [units[q] for q in variables]))
    else:
        idx = pd.MultiIndex.from_tuples(zip(variables, [units[q] for q in variables]),
            names = [index_label, '-'])

    df_out = df.copy()
    df_out.columns = idx
    return df_out


def to_csv(df, units, index_label=None, filepath=None):
    """
    Write dataframe to csv file or returns bytes object with csv file content.

    This function adds units to the input dataframe and creates a csv file object.
    If filepath is specified the csv file is written to the specified location.
    If filepath is not specified a bytes object containing the csv file data
    is returned.

    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to be written to csv file.

    units : dict of {str : str}
        Contains the names of the physical quantities as key and 
        the corresponding names of the units as item.

    index_label : str
        If index_label is specified, a name for the row index will be included
        in the multiindex column header.

    filepath : str, default = None
        Filepath to storage location of the new csv file.

    Returns
    -------
    pandas.DataFrame
        If filepath is specified the dataframe will be written to a csv file.
        If filepath is not specified, the function returns a bytes object 
        containing the csv file data.
    """    
    df = add_units(df, units, index_label)
    if index_label == None:
        index = False
    else:
        index = True
    if filepath == None:
        return df.to_csv(index = index)
    else:
        df.to_csv(filepath, index = index)