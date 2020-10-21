import pandas as pd
import numpy as np

def make_missing_np_nan(df, replace_with=np.nan):
    '''Replaces the missing values (encoded in multiple ways) with np.nan

    Parameters:
    -----------
    df: pd.DataFrame
    replace_with: the value to replace them with, default = np.nan
    '''
    df.fillna(value=np.nan, inplace=True)
    df.replace(to_replace=['?', ' ?', '--', '---', '-', '-999', 'n/a', 'NA', 'N/A', 'NAN', 'na', '', 'nan', 'NaN'], value=replace_with, inplace=True)
    return df

def get_nr_unique(df, col):
    '''Returns the number of unique values in a column of a dataframe

    Parameters:
    -----------
    df: pd.DataFrame
    col: str, column name
    '''

    unique_vals = np.unique([str(int_element) for int_element in df[col].tolist()])
    return int(len(unique_vals))

def col_astype(df, col, type):
    '''Returns the dataframe with a specified column changed to a specified data type

    Parameters:
    -----------
    df: pd.DataFrame
    col: str, column to be changed
    type: str, target data type of the column
    '''
    return df.astype({col: type})

def detect_datatypes(df):
    ''' Returns the data types of the columns in a data set using Pandas.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains the dataframe of a given dataset
    '''
    df_new = df.copy()
    dtypes = df.dtypes

    for idx, col in enumerate(df.columns):
        if str(dtypes[idx]) == 'object' or str(dtypes[idx]) == 'category':
            try:
                to_dt = pd.to_datetime(df_new[col], infer_datetime_format=True)
                df_new[col] = to_dt
            except:
                try:
                    to_num = pd.to_numeric(df_new[col])
                    df_new[col] = to_num
                except ValueError:
                    df_new = col_astype(df_new, col, 'category')
                except TypeError:
                    df_new = col_astype(df_new, col, 'category')
        else:
            continue

    return df_new.dtypes

def cast_datatypes(df):
    '''Change datatypes of a dataframe to inferred datatypes
    
    Parameters:
    -----------
    df: pd.DataFrame
    '''
    dtypes = detect_datatypes(df)

    for idx, col in enumerate(df.columns):
      df = col_astype(df, col, dtypes[idx])
        
    return df

def get_nr_missing(df):
    '''Returns the number of missing values in a dataframe

    Parameters:
    -----------
    df: pd.DataFrame
    '''
    df = make_missing_np_nan(df) # set all missing values to np.nan
    return df.isnull().sum().sum()

def get_nr_duplicates(df):
    '''Returns the number of duplicated rows in a dataframe

    Parameters:
    -----------
    df: pd.DataFrame
    '''
    return df.duplicated(subset=None, keep=False).sum().sum()
