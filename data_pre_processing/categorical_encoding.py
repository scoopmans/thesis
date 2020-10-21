import category_encoders as ce
import pandas as pd
import numpy as np

from utils.column_inference import get_nr_unique, make_missing_np_nan, col_astype
from utils.report import inference

def get_categorical_columns(df, inferred=None):
    '''Returns a list of all categorical columns in a dataframe
    
    Parameters:
    -----------
    df: pd.DataFrame
    inferred: dict, containing for each column information on data type, # of missing values etc.
        (can be generated using the inference() function)
    
    '''
    if not inferred: 
        inferred = inference(df)
        
    cat_cols = []
    for col in inferred.keys():
        if inferred[col]['data_type'] in ['category', 'string', 'gender', 'boolean', 'all identical']:
            cat_cols.append(col)

    return cat_cols

# The following 3 functions are for the front-end:
def cat_cols_vue(df, inferred=None):
  '''Returns a list of categorical columns appropriate for front-end

  Parameters:
  -----------
  df: pd.DataFrame
  '''
  cat_cols = get_categorical_columns(df, inferred=inferred)
  result = []
  for col in cat_cols:
    result.append({
      "value": col,
      "text": col
    })
  
  return result
  
def categories_ord_cols(df, ord_cols):
    '''Returns the categories of the ordinal columns in such a way they can be displayed to the front-end

    Parameters:
    -----------
    df: pd.DataFrame
    ord_cols: list of str, containing the names of the ordinal columns
    '''
    result = []
    df_new = make_missing_np_nan(df)
    for col in ord_cols:
        col_result = {
          "name": col,
          "values": []
        }

        i = 1
        categories = list(set(df_new[col]))
        if pd.isnull(categories).any():
            idx = np.where(pd.isnull(categories) == True)[0][0]
            del categories[idx]
        for category in categories:
            col_result['values'].append({
                "name": category,
                "id": i,
                "fixed": "false"
            })
            i += 1

        result.append(col_result)

    return result

def get_orderings_from_vue(ordering):
  '''Based on an ordering from the front-end, the mapping for Ordinal Encoding are generated
  Input is ordered in the following way:
  [
    {
      "name": category (= lowest),
      "fixed": "false"
    },
    {
      "name": category (= middle),
      "fixed": "false
    },
    {
      "name": category (= highest),
      "fixed": false
    }
  ]

  Parameters:
  -----------
  ordering: list of dicts, see above for example
  '''
  result = {}
  i=1
  for category in ordering:
      result[category['name']] = i
      i +=1 

  return result

def ordinal_encoding(df, cols, orderings, handle_nan=True):
    df_new = df.copy()
    for col in cols:
        df_new = col_astype(df_new, col, 'object')

    if handle_nan:
        encoder = ce.OrdinalEncoder(cols=cols, mapping=orderings, handle_unknown='value', handle_missing='value')
    else:
        encoder = ce.OrdinalEncoder(cols=cols, mapping=orderings, handle_unknown='return_nan', handle_missing='return_nan')
    df_new = encoder.fit_transform(df_new)
    return df_new

def one_hot_encoding(df, cols, handle_nan=True):

    if handle_nan:
        encoder = ce.OneHotEncoder(cols=cols, handle_unknown='value', handle_missing='value', use_cat_names=True)
    else:
        encoder = ce.OneHotEncoder(cols=cols, handle_unknown='return_nan', handle_missing='return_nan', use_cat_names=True)
    df_new = encoder.fit_transform(df)
    return df_new

def binary_encoding(df, cols, handle_nan=True):
    if handle_nan:
        encoder = ce.BinaryEncoder(cols=cols, handle_unknown='indicator', handle_missing='indicator')
    else:
        encoder = ce.BinaryEncoder(cols=cols, handle_unknown='return_nan', handle_missing='return_nan')
    df_new = encoder.fit_transform(df)
    return df_new

def hashing_encoding(df, cols, handle_nan=True):
    if handle_nan:
        encoder = ce.HashingEncoder(cols=cols)
    else:
        encoder = ce.HashingEncoder(cols=cols)
    df_new = encoder.fit_transform(df)
    return df_new

def target_encoding(df, cols, handle_nan=True, target=False):
    if handle_nan:
        encoder = ce.TargetEncoder(cols=cols, handle_unknown='value', handle_missing='value')
    else:
        encoder = ce.TargetEncoder(cols=cols, handle_unknown='return_nan', handle_missing='return_nan')

    if target:
        df_new = encoder.fit_transform(df, y=df[[target]])
        return df_new
    else:
        df_new = encoder.fit_transform(df, y=df[df.columns[-1]])
        return df_new


def leave_one_out_encoding(df, cols, handle_nan=True, target=False):
    if handle_nan:
        encoder = ce.LeaveOneOutEncoder(cols=cols, handle_unknown='value', handle_missing='value')
    else:
        encoder = ce.LeaveOneOutEncoder(cols=cols, handle_unknown='return_nan', handle_missing='return_nan')

    if target:
        df_new = encoder.fit_transform(df, y=df[[target]])
        return df_new
    else:
        df_new = encoder.fit_transform(df, y=df[df.columns[-1]])
        return df_new

def above_below_cardinality(df, nom_cols):
    '''For all the nominal columns (i.e. categorical columns that are not ordinal)
    it will check whether they are above or below 15 categories.

    Parameters:
    -----------
    df: pd.DataFrame
    nom_cols: list of str, containing the names of the nominal columns
    '''
    below = []
    above = []
    for col in nom_cols:
        if get_nr_unique(df, col) < 15:
            below.append(col)
        else:
            above.append(col)

    return above, below

def below_cardinality_handling(cols, info_loss=False, overfitting=True, decision_tree=False):
    '''Returns the intended encoding handling for columns which have less than 15 categories.
    
    Parameters:
    -----------
    cols: list of str, containing all categorical columns that contain LESS than 15 categories
    info_loss: boolean (default = False), if set to True, some info loss during encoding is 
        acceptable for lower dimensionality
    overfitting: boolean (default = True), if set to False, the encoder will NOT handle overfitting inherently
    decision_tree: boolean (default = False), if set to True, the user will eventually make use of a decision
        tree algorithm
    '''
    if decision_tree:
        if info_loss:
            return "bin_enc"
        else:
            if overfitting:
                return "leave_one_out"
            else:
                return "target_enc"
    else:
        return "one_hot"

def above_cardinality_handling(cols, memory_issue=True, info_loss=False, overfitting=True, decision_tree=False):
    '''Returns the intended encoding handling for columns which have more than 15 categories.
    
    Parameters:
    -----------
    cols: list of str, containing all categorical columns that contain MORE than 15 categories
    memory_issue: boolean (default = True), if set to False, sparse encoding will NOT lead to memory issues, and 
        the encoder will prefer sparse encoding to avoid information loss
    info_loss: boolean (default = False), if set to True, some info loss during encoding is 
        acceptable for lower dimensionality
    overfitting: boolean (default = True), if set to False, the encoder will NOT handle overfitting inherently
    decision_tree: boolean (default = False), if set to True, the user will eventually make use of a decision
        tree algorithm
    '''
    if memory_issue:
        if info_loss:
            return "bin_enc"
        else:
            if overfitting:
                return "leave_one_out"
            else:
                return "target_enc"

    else:
        if not decision_tree:
            return "one_hot"
        else:
            if info_loss:
                return "bin_enc"
            else:
                if overfitting:
                    return "leave_one_out"
                else:
                    return "target_enc"

def default_encoding(df, nom_cols):
    '''Returns the default encoding, which is one-hot encoding for columns with less than 15 categories and
    target encoding for columns with more than 15 categories.
    
    An example of the result looks like this:
    {
    'target_enc': ['column1', 'column3', 'column4'],
    'one_hot_enc': ['column2', 'column5']
    }
    
    Parameters:
    -----------
    df: pd.DataFrame
    nom_cols: nom_cols: list of str, containing the nominal column names in the data set
    '''
    above, below = above_below_cardinality(df, nom_cols)
    
    result = {}
    
    result[below_cardinality_handling(below)] = below
    result[above_cardinality_handling(above)] = above 
    
    return result
