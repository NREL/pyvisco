"""Collection of functions to write dataframes to files or buffer.
"""

import pandas as pd

def add_units(df, units, index_label=None):
    variables = df.columns.get_level_values(0)
    if index_label == None:
        idx = pd.MultiIndex.from_tuples(zip(variables, [units[q] for q in variables]))
    else:
        idx = pd.MultiIndex.from_tuples(zip(variables, [units[q] for q in variables]),
            names = [index_label, '-'])

    df_out = df.copy()
    df_out.columns = idx
    return df_out

def csv(df, units, index_label=None, filepath=None):
    df = add_units(df, units, index_label)
    if index_label == None:
        index = False
    else:
        index = True
    if filepath == None:
        return df.to_csv(index = index)
    else:
        df.to_csv(filepath, index = index)