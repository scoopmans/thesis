import pandas as pd
import numpy as np
import scipy.stats as st
import math as ma

from utils.report import inference
from utils.column_inference import make_missing_np_nan

def nr_rows_missing(df):
    '''Returns the number of rows with missing values

    Parameters:
    -----------
    df: pd.DataFrame

    Returns:
    --------
    int: number of rows with missing values
    '''
    df_all_nan = make_missing_np_nan(df)
    return len(df_all_nan[df_all_nan.isnull().any(axis=1)])

def imputation_heuristic_rows(df, threshold = 0.05):
    '''Checks whether the number of rows with missing values is below a certain threshold

    Parameters:
    -----------
    df: pd.DataFrame
    threshold: float, portion of rows that is allowed to contain missing values

    Returns:
    --------
    bool: indicating if the percentage of rows with missing values is below the threshold
    '''
    return (nr_rows_missing(df)/df.shape[0]) < threshold

def imputation_heuristic_column(df, threshold = 0.8):
    '''Suggest columns to delete based on the % of missing values in that column

    Parameters:
    -----------
    df: pd.DataFrame
    threshold: float, portion of values in a column that are allowed to be missing

    Returns:
    --------
    tuple: of the form (list1, list2)
      list1: list of columns in the dataframe where the number of observations missing is above the threshold
      list2: list of columns in the dataframe which have only a single unique observation 
    '''
    columns_suggested_missing = []
    columns_suggested_unique = []
    infer = inference(df)
    for col in infer.keys():
        if float(infer[col]['pct_missing'][:-1])/100 > threshold:
            columns_suggested_missing.append(col)
        elif int(infer[col]['nr_unique']) == 1:
            columns_suggested_unique.append(col)
        else:
            continue

    return columns_suggested_missing, columns_suggested_unique

def checks_input_mcar_tests(data):
    ''' Checks whether the input parameter of class McarTests is correct

    Parameters
    ----------
    data: The input of McarTests specified as 'data'
    '''

    if not isinstance(data, pd.DataFrame):
        print("Error: Data should be a Pandas DataFrame")
        return False

    if not any(data.dtypes.values == np.float):
        if not any(data.dtypes.values == np.int):
            print("Error: Dataset cannot contain other value types than floats and/or integers")
            return False

    if not data.isnull().values.any():
        print("Error: No NaN's in given data")
        return False

    return True
  
class McarTests():

    def __init__(self, data):
        self.data = make_missing_np_nan(data)

    def mcar_test(self):
        '''
        Implementation of Little's MCAR test, returning the p-value of a chi-square statistical test. Testing
        whether the null hypothesis 'the missingness mechanism of the incomplete dataset is MCAR' can be rejected.

        Parameters:
        -----------
        data: pd.DataFrame, an incomplete dataset with samples as index and variables as columns
        '''

        if not checks_input_mcar_tests(self.data):
            raise Exception("Input not correct")

        dataset = self.data.copy()
        vars = dataset.dtypes.index.values
        n_var = dataset.shape[1]

        # mean and covariance estimates
        # ideally, this is done with a maximum likelihood estimator
        gmean = dataset.mean()
        gcov = dataset.cov()

        # set up missing data patterns
        r = 1 * dataset.isnull()
        mdp = np.dot(r, list(map(lambda x: ma.pow(2, x), range(n_var))))
        sorted_mdp = sorted(np.unique(mdp))
        n_pat = len(sorted_mdp)
        correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
        dataset['mdp'] = pd.Series(correct_mdp, index=dataset.index)

        # calculate statistic and df
        pj = 0
        d2 = 0
        for i in range(n_pat):
            dataset_temp = dataset.loc[dataset['mdp'] == i, vars]
            select_vars = ~dataset_temp.isnull().any()
            pj += np.sum(select_vars)
            select_vars = vars[select_vars]
            means = dataset_temp[select_vars].mean() - gmean[select_vars]
            select_cov = gcov.loc[select_vars, select_vars]
            mj = len(dataset_temp)
            parta = np.dot(means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[1])))
            d2 += mj * (np.dot(parta, means))

        df = pj - n_var

        # perform test and save output
        p_value = 1 - st.chi2.cdf(d2, df)

        return p_value

    def mcar_t_tests(self):
        """ MCAR tests for each pair of variables
        Parameters
        ----------
        data: Pandas DataFrame
            An incomplete dataset with samples as index and variables as columns
        Returns
        -------
        mcar_matrix: Pandas DataFrame
            A square Pandas DataFrame containing True/False for each pair of variables
            True: Missingness in index variable is MCAR for column variable
            False: Missingness in index variable is not MCAR for column variable
        """

        if not checks_input_mcar_tests(self.data):
            raise Exception("Input not correct")

        dataset = self.data.copy()
        vars = dataset.dtypes.index.values
        mcar_matrix = pd.DataFrame(data=np.zeros(shape=(dataset.shape[1], dataset.shape[1])),
                                   columns=vars, index=vars)

        for var in vars:
            for tvar in vars:
                part_one = dataset.loc[dataset[var].isnull(), tvar].dropna()
                part_two = dataset.loc[~dataset[var].isnull(), tvar].dropna()
                mcar_matrix.loc[var, tvar] = st.ttest_ind(part_one, part_two, equal_var=False).pvalue

        mcar_matrix = mcar_matrix[mcar_matrix.notnull()] > 0.05

        return mcar_matrix

    def MCAR_MAR(self):
        '''Checks if the p-value from Little's MCAR test is below 0.05, which is interpreted as being that the missing data is not MCAR. 

        Parameters:
        -----------
        df: pd.DataFrame

        Returns:
        --------
        bool: Boolean indicating if the null hypothesis is rejected.
        '''

        if self.mcar_test() < 0.05:
            return True
        else:
            return False

def get_heuristics(df):
    '''Summarizes the heuristics and suggestions in a way that the frontend can handle it.

    Parameters:
    -----------
    df: pd.DataFrame

    Returns:
    --------
    dict: with the suggestions for the frontend
    '''
    result = []
  
    if imputation_heuristic_rows(df):
      result.append({
        'type': 'ROW_SAFE',
        'text': 'Less than 5% of the observations have missing cells, deletion will not affect the results drastically.',
      })
    else:
      result.append({
        'type': 'ROW_VIOLATION',
        'text': 'More than 5% of the observations have missing cells, deletion is discouraged.',
      })

    if df.shape[0] < 5000:
      try:
        little = McarTests(df)

        if little.MCAR_MAR():
          result.append({
            'type': 'LITTLE_MAR',
            'text': 'Your missing data is probably MAR, which indicates that deletion of missing observations will affect the results. Please use a form of imputation for optimal results.'
          })
          
        else:
          result.append({
            'type': 'LITTLE_MCAR',
            'text': 'Your missing data is probably MCAR, meaning that any form of imputation will not affect results significantly.'
          })

        return result
      except:
        return result
    else:
      return result
