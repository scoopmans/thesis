from utils.column_inference import make_missing_np_nan, get_nr_missing
from utils.imputation_heuristics import nr_rows_missing, imputation_heuristic_column
from utils.report import inference, type_cols

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from datawig import simple_imputer

# THE FOLLOWING FUNCTIONS ARE FOR THE IMPUTATION INFORMATION TABLE!
class ImputationTable():
  '''Class constructed to get the imputation information table 
  '''
  def __init__(self, data, inferred=None):
    self.df = data

    if not inferred:
      self.inferred = inference(self.df)
    else:
      self.inferred = inferred
  
  def col_data_imputation(self):
    cols_missing = []
    num_cols_missing = []
    cat_cols_missing = []
    
    for col in self.inferred.keys():
        if self.inferred[col]['nr_missing'] > 0:
            cols_missing.append(col)
            if self.inferred[col]['data_type'] in ['category', 'string', 'gender', 'boolean']:
                cat_cols_missing.append(col)
            else:
                num_cols_missing.append(col)
        else:
            continue

    col_result = {
        'measure': 'Number of columns with missing values',
        'number': '{} ({}%)'.format(len(cols_missing), round(len(cols_missing)/self.df.shape[1] * 100, 2)),
        'details': [
            {
                'text': 'Categorical columns with missing cells',
                'value': '{}, {}'.format(len(cat_cols_missing), cat_cols_missing)
            },
            {
                'text': 'Numerical columns with missing cells',
                'value': '{}, {}'.format(len(num_cols_missing), num_cols_missing)
            }
        ]
    }
    
    return col_result

  def cell_data_imputation(self):
    nr_missing = get_nr_missing(self.df)
    result = {
        'measure': 'Total number of cells missing',
        'number': '{} ({}%)'.format(nr_missing, round(nr_missing/(self.df.shape[0]*self.df.shape[1]) * 100, 2)),
        'details': [
            {
                'text': 'Number of missing cells',
                'value': str(nr_missing)
            },
            {
                'text': 'Total number of cells',
                'value': str(self.df.shape[0]*self.df.shape[1])
            },
            {
                'text': 'Percentage of cells missing',
                'value': '{}%'.format(round(nr_missing/(self.df.shape[0]*self.df.shape[1]) * 100, 2))
            }
        ]
    }
    return result

  def observation_data_imputation(self):
    nr_rows = nr_rows_missing(self.df)
    result = {
        'measure': 'Number of observations with missing cells',
        'number': '{} ({}%)'.format(nr_rows, round(nr_rows/self.df.shape[0] * 100, 2)),
        'details': [
            {
                'text': 'Number of observations with missing values',
                'value': str(nr_rows)
            },
            {
                'text': 'Total number of observations',
                'value': str(self.df.shape[0])
            },
            {
                'text': 'Percentage of observations with missing values',
                'value': '{}%'.format(round(nr_rows/self.df.shape[0] * 100, 2))
            }
        ]
    }
    return result

  def imputation_table(self):
    result = []
    result.append(self.observation_data_imputation())
    result.append(self.col_data_imputation())
    result.append(self.cell_data_imputation())
    return result

def delete_cols(df, cols):
    '''Deletes a list of columns from a dataframe

    Parameters:
    -----------
    df: pd.DataFrame
    col: list, containing the columns to be deleted

    Returns:
    --------
    pd.DataFrame: where the specified columns are deleted
    '''
    return df.drop(columns = cols, axis=1)

def placeholder_imputation(df, col, placeholder):
    '''Returns the data frame with the missing values of the specified columns replaced by a placeholder

    Parameters:
    -----------
    df: pd.DataFrame
    col: str, corresponding to a categorical column in the data frame
    placehold: str, placeholder value to replace the missing values with

    Returns:
    --------
    pd.DataFrame: where the missing values for the specified columns are replaced by a placeholder
    '''

    df = make_missing_np_nan(df)
    df_new = df[[col]]
    df_new.fillna(value=placeholder, inplace=True)
    df[col] = df_new[col]

    return df
  
def mean_imputation(df):
    '''Imputes the missing values in a data frame using mean imputation

    Parameters:
    -----------
    df: pd.DataFrame

    Returns:
    --------
    df_result: pd.DataFrame where the missing values are imputed using the mean
    '''
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

    df = make_missing_np_nan(df)
    cat_cols, date_cols, num_cols = type_cols(df)
    df_new = df[num_cols]

    columns = df.columns
    df_imputed = imp_mean.fit_transform(df_new)
    df_imputed = pd.DataFrame(df_imputed, columns = columns)

    not_imputed_cols = cat_cols + date_cols
    df_result = pd.concat([df_imputed, df[not_imputed_cols]], axis=1)

    return df_result

def median_imputation(df):
    '''Imputes the missing values in a data frame using median imputation

    Parameters:
    -----------
    df: pd.DataFrame

    Returns:
    --------
    df_result: pd.DataFrame where the missing values are imputed using the median 
    '''

    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

    df = make_missing_np_nan(df)
    cat_cols, date_cols, num_cols = type_cols(df)
    df_new = df[num_cols]

    columns = df_new.columns
    df_imputed = imp_median.fit_transform(df_new)
    df_imputed = pd.DataFrame(df_imputed, columns = columns)

    not_imputed_cols = cat_cols + date_cols
    df_result = pd.concat([df_imputed, df[not_imputed_cols]], axis=1)

    return df_result

def KNN_imputation(df, k=5):
    '''Imputes the missing values in a dataframe using K-Nearest Neighbor

    Parameters:
    -----------
    df: pd.DataFrame
    k: int, number of neighboring samples to use for imputation

    Returns:
    --------
    df_result: pd.DataFrame where the missing values are imputed using KNN
    '''
    df_new = df.copy()

    df_new = make_missing_np_nan(df_new)

    missing, unique = imputation_heuristic_column(df, 1)
    df_new = delete_cols(df_new, missing)
    df_new = delete_cols(df_new, unique)

    cat_cols, date_cols, num_cols = type_cols(df_new)
    df_new = df_new[num_cols]

    columns = df_new.columns

    imputer = KNNImputer(n_neighbors = k)
    imputed = imputer.fit_transform(df_new)
    df_imputed = pd.DataFrame(imputed, columns=columns)

    not_imputed_cols = cat_cols + date_cols
    df_result = pd.concat([df_imputed, df[not_imputed_cols]], axis=1)
    
    return df_result

def regression_imputation(df):
    '''Returns the dataframe where missing values are imputed using IterativeImputer and BayesianRidge()
    This is a regression imputation method.

    Parameters:
    -----------
    df: pd.DataFrame

    Returns:
    --------
    df_result: pd.DataFrame where the missing values are imputed using multiple imputation
    '''
    df_new = df.copy()

    df_new = make_missing_np_nan(df_new)

    missing, unique = imputation_heuristic_column(df, 0.99)
    
    df_new = delete_cols(df_new, missing)
    df_new = delete_cols(df_new, unique)

    cat_cols, date_cols, num_cols = type_cols(df_new)
    df_new = df_new[num_cols]

    columns = df_new.columns

    imputer = IterativeImputer(random_state=0)
    imputed = imputer.fit_transform(df_new)
    df_imputed = pd.DataFrame(imputed, columns=columns)

    not_imputed_cols = cat_cols + date_cols
    df_result = pd.concat([df_imputed, df[not_imputed_cols]], axis=1)
   
    return df_result

def MICE_imputation(df, categorical=False, nr_iter=3):
    '''Returns the dataframe where missing values are imputed using MICE

    Parameters:
    -----------
    df: pd.DataFrame
    categorical: boolean, if set to True, the returned dataframe will contain the original category values
                 (as opposed to their integer index)
    nr_iter: int, the number of imputations to be generated

    Returns:
    --------
    df_result: pd.DataFrame where the missing values are imputed using MICE
    '''
    df_new = df.copy()

    df_new = make_missing_np_nan(df_new)

    missing, unique = imputation_heuristic_column(df, 0.99)
    df_new = delete_cols(df_new, missing)
    df_new = delete_cols(df_new, unique)

    cat_cols, date_cols, num_cols = type_cols(df_new)
    df_new = df_new[num_cols]

    columns = df_new.columns

    result = [0] * nr_iter
    for i in range(nr_iter):
        imputer = IterativeImputer(sample_posterior=True)
        imputed = imputer.fit_transform(df_new)
        df_imputed = pd.DataFrame(imputed, columns=columns)
        result[i] = df_imputed

    return result

def RF_imputation(df, fast=True):
    '''Returns the dataframe where missing values are imputed using Random Forest Imputation (sklearn)
    ExtraTreesRegressor is used for increased speed.

    Parameters:
    -----------
    df: pd.DataFrame
    fast: boolean, if set to True, ExtraTreesRegressor is used in preference of RandomForestRegressor

    Returns:
    --------
    df_result: pd.DataFrame where the missing values are imputed using Random Forest (MissForest)
    '''

    df_new = df.copy()

    df_new = make_missing_np_nan(df_new)

    missing, unique = imputation_heuristic_column(df, 0.99)
    df_new = delete_cols(df_new, missing)
    df_new = delete_cols(df_new, unique)

    #categorical and datetime columns cannot be imputed, so are removed from the imputation dataframe
    cat_cols, date_cols, num_cols = type_cols(df_new)
    df_new = df_new[num_cols]

    columns = df_new.columns
    
    if fast:
        imputer = IterativeImputer(random_state=0, estimator=ExtraTreesRegressor(n_estimators=10, random_state=0))
    else:
        imputer = IterativeImputer(random_state=0, estimator=RandomForestRegressor(n_estimators=10, random_state=0))

    imputed = imputer.fit_transform(df_new)
    df_imputed = pd.DataFrame(imputed, columns=columns)

    #categorical and datetime columns are added back
    not_imputed_cols = cat_cols + date_cols
    df_result = pd.concat([df_imputed, df[not_imputed_cols]], axis=1)

    return df_result

def DL_imputation(df, categorical=True):
    '''Returns the dataframe where missing values are imputed using DataWig
    
    Parameters:
    -----------
    df: pd.DataFrame
    categorical: boolean, if set to True, the returned dataframe will contain the original category values
                 (as opposed to their integer index)

    Returns:
    --------
    df_result: pd.DataFrame where the missing values are imputed using DataWig
    '''

    df_new = df.copy()

    df_new = make_missing_np_nan(df_new)

    missing, unique = imputation_heuristic_column(df, 0.99)
    df_new = delete_cols(df_new, missing)
    df_new = delete_cols(df_new, unique)

    cat_cols, date_cols, num_cols = type_cols(df_new)
    df_new = df_new[num_cols]

    columns = df_new.columns

    num_cols = [col for col in df_new.columns if is_numeric_dtype(df_new[col])]
    string_cols = list(set(df_new.columns) - set(num_cols))
    imputer = simple_imputer.SimpleImputer(input_columns=['1'], output_column='2')
    imputed = imputer.complete(df_new)
    df_imputed = pd.DataFrame(imputed, columns=columns)

    not_imputed_cols = cat_cols + date_cols
    df_result = pd.concat([df_imputed, df[not_imputed_cols]], axis=1)

    return df_result

def create_missing(df, perc_missing=0.1):
    ''' Returns 1) a complete dataframe (without missing values) and 2) the same dataframe with X% missing values

    Parameters:
    -----------
    df: pd.DataFrame
    perc_missing: float,
        indicates the percentage of values to synthetically make missing

    Returns:
    --------
    df_nona: the provided dataframe but with observations with missing values dropped
    df_synna: df_nona where perc_missing% of the cells are synthetically made missing
    '''

    df_nona = df.dropna()
    #df_nona = df_nona.reset_index()
    nan_mat = np.random.random(df_nona.shape)<perc_missing
    df_synna = df_nona.mask(nan_mat)

    return df_nona, df_synna

def imputation_error(df, imputation='KNN', perc_missing=0.15):
    ''' Returns a dataframe containing the squared errors between the original values and the imputed values

    Parameters:
    -----------
    df: pd.DataFrame
    imputation: str,
        the selected imputation technique
    perc_missing: float,
        the percentage of values to synthetically make missing
    
    Returns:
    --------
    NRMSE: pd.DataFrame containing the normalized root mean squared errors between the original values and the synthetically imputed values
    '''    
    if df.shape[0] > 5000:
      df = df.sample(5000).reset_index()

    scaler = StandardScaler()
    scaler.fit(df.values)
    df_scaled = pd.DataFrame(scaler.transform(df.values), columns=df.columns)
        
    original, with_nan = create_missing(df_scaled, perc_missing)
    print("=====> DF shape original: ", original.shape)
    print("=====> DF shape with_nans: ", with_nan.shape)
    
    if imputation == 'KNN':
      print('=====> KNN imputation started...')
      imputed = KNN_imputation(with_nan, k=5)
#         imputed = KNN_imputation(pd.DataFrame(scaler.transform(with_nan), columns=with_nan.columns), k=5)
#         imputed = pd.DataFrame(scaler.inverse_transform(imputed.values), columns=imputed.columns)
    elif imputation == 'mean':
      print('=====> Mean imputation started...')
      imputed = mean_imputation(with_nan)
    elif imputation == 'median':
      print('=====> Median imputation started...')
      imputed = median_imputation(with_nan)
    elif imputation == "DL":
      imputed = DL_imputation(with_nan)
    elif imputation == "RF":
      print('=====> RF imputation started...')
      imputed = RF_imputation(with_nan)
    elif imputation == "regression":
      print('=====> Regression imputation started...')
      imputed = regression_imputation(with_nan)
    
    original_features = scaler.inverse_transform(original.values)
    original = pd.DataFrame(original_features, columns=df.columns)
    
    imputed_features = scaler.inverse_transform(imputed.values)
    imputed = pd.DataFrame(imputed_features, columns=df.columns)
    
    errors = (original-imputed)
    NRMSE = np.sqrt(np.nanmean(errors.values**2)/original.values.var())
        
    return NRMSE

def get_imputation_scores(df):
    '''For the specified dataframe, each of the imputation methods is tried on the observations without missing values and synthethically imputed
    to establish the best performing imputation method.

    Parameters:
    -----------
    df: pd.DataFrame

    Returns:
    result: dict containing for each of the imputation techniques (mean, median, multiple, KNN and RF) the NRMSE
    '''
    cat_cols, date_cols, num_cols = type_cols(df)
    df_new = df[num_cols]

    if df.shape[0] * df.shape[1] < 10000:
      imputations = ['mean', 'median', 'regression', 'KNN', 'RF']
    else:
      imputations = ['mean', 'median', 'regression', 'RF']
    result = {}
    for imputation in imputations:
        score = imputation_error(df_new, imputation=imputation)
        result[imputation] = score
    return result
