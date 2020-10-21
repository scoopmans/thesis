from utils.column_inference import detect_datatypes, make_missing_np_nan, get_nr_missing, get_nr_duplicates
from utils.column_inference_ptype import detect_datatypes_ptype
from utils.plots import distribution_plot, missing_values_plot, correlations_heatmap

import pandas as pd
import numpy as np
import json
import plotly

def dataset_statistics(df):
    '''Returns the following statistics about a dataset:
       - number of variables          -> nr_variables
       - number of observations       -> nr_observations
       - number of missing values     -> nr_missing
       - percentage of missing values -> pct_missing
       - number of duplicate rows     -> nr_duplicates
       - percentage of duplicate rows -> pct_duplicates

    Parameters:
    -----------
    df: pd.DataFrame
    '''
    nr_missing = get_nr_missing(df)
    nr_duplicates = get_nr_duplicates(df)

    dataset_descriptives = {
        'Number of variables': int(df.shape[1]),
        'Number of observations': int(df.shape[0]),
        'Number of missing cells': int(nr_missing),
        'Missing cells (%)': str('{:.1f}%'.format(nr_missing/(df.shape[0]*df.shape[1]) * 100)),
        'Number of duplicate rows': int(nr_duplicates),
        'Duplicates rows (%)': str('{:.1f}%'.format(nr_duplicates/df.shape[0] * 100))
    }

    return dataset_descriptives

def format_dataset_statistics(df):
  '''Formatting of the dataset statistics output so it can be used in the frontend.

  Parameters:
  -----------
  df: pd.DataFrame
  '''
  result = []
  for (key,value) in dataset_statistics(df).items():
    result.append(
      {
        'statistic': key,
        'value': value
      }
    )
  return result

def inference(df, fast=True):
    ''' Returns a dictionary with information about each column in a dataframe using PANDAS, information includes:
        - detected data type (using pandas)           -> data_type_pandas
        - nr. of missing values                       -> nr_missing
        - % of data consist of missing values         -> pct_missing
        - nr. of unique values                        -> nr_unique

    Parameters:
    -----------
    df: pd.DataFrame
    fast: bool, indicates whether to run fast (less accurate) or slow (more accurate) data type detection inference
    '''

    df = make_missing_np_nan(df) #set all missing value encodings to np.nan

    if fast:
        data_types = detect_datatypes(df)
    else:
        data_types = detect_datatypes_ptype(df)

    columns_missing = df.columns[df.isna().any()].tolist()

    result_inference = {}
    for idx,col in enumerate(df.columns):
        unique_vals, unique_vals_counts = np.unique([str(int_element) for int_element in df[col].tolist()], return_counts=True)
        nan_idx = np.where(unique_vals == 'nan')
        nr_missing = df[col].isnull().sum()

        if col in columns_missing:
            result_inference[col] = {'data_type': str(data_types[col]),
                                     'nr_missing': int(nr_missing),
                                     'pct_missing': str('{:.1f}%'.format(nr_missing / len(df) * 100)),
                                     'nr_unique': int(len(unique_vals)),
                                     'pct_unique': str('{:.1f}%'.format(len(unique_vals) / len(df) * 100)),
                                     'distribution_plot': distribution_plot(col, df[col], str(data_types[col]), unique_vals, unique_vals_counts)
                                     }
        else:
            result_inference[col] = {'data_type': str(data_types[col]),
                                     'nr_missing': 0,
                                     'pct_missing': '0.0%',
                                     'nr_unique': int(len(unique_vals)),
                                     'pct_unique': str('{:.1f}%'.format(len(unique_vals) / len(df) * 100)),
                                     'distribution_plot': distribution_plot(col, df[col], str(data_types[col]), unique_vals, unique_vals_counts)
                                     }
    return result_inference

def format_inference(df, fast=True, inferred=None):
  '''Formatting of the inference output so it can be used in the frontend.

  Parameters:
  -----------
  df: pd.DataFrame
  inferred: dict from inference
  '''
  if not inferred:
    print('Performing inference...')
    inferred = inference(df, fast=fast)
  else:
    print('Inference given...')
  
  result = []

  for key,value in inferred.items():
    col = {
      'name': key,
      'badges': [value['data_type']],
      'table': [],
    }
    
    if inferred[key]['nr_missing'] > 0:
      col['badges'].append('MISSING')
    if inferred[key]['nr_unique'] == 1:
      col['badges'].append('1 UNIQUE')

    for key2,value2 in value.items():
      if key2 != 'distribution_plot':
        col['table'].append(
          {
            'feature': key2,
            'value': value2
          } 
        )
      else:
        col['plot'] = value2

    result.append(col)

  return result

def type_cols(df, inferred=None):
  '''Based on the inference, it will return three lists of columns names, 
  one for the categorical columns, one for the datetime columns and one for the numerical columns.

  Parameters:
  -----------
  df: pd.DataFrame
  '''
  if not inferred:
    inferred = inference(df)
  
  cat_cols = []
  num_cols = []
  date_cols = []
  for col in inferred.keys():
      if inferred[col]['data_type'] in ['category', 'string', 'gender', 'boolean']:
          cat_cols.append(col)
      elif inferred[col]['data_type'] in ['date-non-std', 'datetime64[ns]']:
          date_cols.append(col)
      else:
          num_cols.append(col)
  return cat_cols, date_cols, num_cols

def report_type_cols(df, cat_cols=None, date_cols=None, num_cols=None):
    '''For the three types of columns it will return a dictionary indicating how many columns are that specific type.

    Parameters:
    -----------
    df: pd.DataFrame
    '''
    if not cat_cols or num_cols or date_cols:
      cat_cols, date_cols, num_cols = type_cols(df)
    
    result= [{
        'variableType': 'NUM',
        'value': len(num_cols)
    },
    {
        'variableType': 'CAT',
        'value': len(cat_cols)
    },
    {
        'variableType': 'DATE',
        'value': len(date_cols)
    }]
    
    return result

def fabrice_type_cols(df):
  '''Specific code for recommender system, that returns a list with True/False if a column is categorical.

  Parameters:
  -----------
  df: pd.DataFrame
  '''
  inferred = inference(df)
  result = []

  for col in inferred.keys():
    if inferred[col]['data_type'] in ['category', 'string', 'gender', 'boolean']:
      result.append(True)
    else:
      result.append(False)
  
  return result

