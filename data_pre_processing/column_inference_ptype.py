from ptype.Ptype import Ptype
import pandas as pd

def detect_datatypes_ptype(df):
    ''' Infers and returns the data types of the columns in a data set using Ptype().
    This results in a more detailed inferred data type compared to pandas data type detection.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains the dataframe of a given dataset.
    '''
    ptype = Ptype()
    ptype.run_inference(_data_frame=df)
    return pd.Series(ptype.predicted_types)
