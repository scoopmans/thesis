from utils.report import type_cols
from utils.column_inference import make_missing_np_nan

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from pyod.models.knn import KNN
from pyod.models.vae import VAE


def get_outliers_info(df, outlier_method):
  '''For the frontend produces the outlier scores, the three sigmas, and the corresponding plot of the outliers

  Parameters:
  ----------
  df: pd.DataFrame
  outlier_method: str, corresponding to one of the following outlier detection methods:
    ['LOF', 'IF', 'SVM', 'KNN', 'VAE']
  '''
  df_new = make_missing_np_nan(df)
  df_new = df_new.dropna()
  cat_cols, date_cols, num_cols = type_cols(df_new)
  df_new = df_new[num_cols]
  
  if outlier_method == "LOF":
    outliers = detect_outliers_LOF(df_new)
  elif outlier_method == "IF":
    outliers = detect_outliers_IF(df_new)
  elif outlier_method == "SVM":
    outliers = detect_outliers_SVM(df_new)
  elif outlier_method == "KNN":
    outliers = detect_outliers_KNN(df_new)
  elif outlier_method == "VAE":
    outliers = detect_outliers_VAE(df_new)

  return outliers, three_sigma(outliers), detection_scores(outliers), outlier_scores_plot(outliers)

def detect_outliers_LOF(df):
    ''' Returns the outlier scores using Local Outlier Factor
    Inliers tend to have a LOF score close to 1, while outliers tend to have a larger LOF score.

    Parameters:
    -----------
    df: pd.DataFrame,
    '''

    clf = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    clf.fit_predict(df)
    scores = clf.negative_outlier_factor_
    return scores

def detect_outliers_IF(df, n_estimators=100):
    ''' Returns the outlier scores using IsolationForest

    Parameters:
    -----------
    df: pd.DataFrame,
    '''
    clf = IsolationForest(n_estimators=n_estimators, contamination=0.1, random_state=123)
    clf.fit_predict(df)
    scores = clf.score_samples(df)
    # dec_func = clf.decision_function(df_imputed)
    return scores

def detect_outliers_SVM(df):
    ''' Returns the outlier scores using SVM (beware: prone to overfitting)

    Parameters:
    -----------
    df: pd.DataFrame,
    '''
    clf = OneClassSVM()
    clf.fit_predict(df)
    scores = clf.score_samples(df)
    # dec_func = clf.decision_function(df_imputed)
    return scores

def detect_outliers_KNN(df):
    ''' Returns the outlier scores using K-Nearest Neighbor

    Parameters:
    -----------
    df: pd.DataFrame,
    '''
    clf = KNN(contamination=0.2)
    clf.fit(df)
    outlier_score = clf.decision_scores_
    # df_result = pd.DataFrame(outlier_pred, columns=['outlier_pred'])
    return outlier_score * -1

def detect_outliers_VAE(df):
    ''' Returns the outlier scores using Variational AutoEncoders

    Parameters:
    -----------
    df: pd.DataFrame,
    '''
    if df.shape[1] < 128:
        encoder = [df.shape[1], df.shape[1]/2, df.shape[1]/4]
        decoder = encoder[::-1]
    else:
        encoder = [128, 64, 32]
        decoder = encoder[::-1]

    clf = VAE(contamination=0.1, encoder_neurons=encoder, decoder_neurons=decoder)

    df = df.astype(np.float32)

    clf.fit(df)
    outlier_score = clf.decision_scores_
    # df_result = pd.DataFrame(outlier_pred, columns=['outlier_pred'])
    return outlier_score * -1

def get_threshold(outliers):
    '''
    The threshold which classifies a data point as an outlier or inlier is established as follows:
        mean - 3 * standard deviation
    of the outlier scores obtained. Values below this threshold are classified as outliers.

    Parameters:
    -----------
    outliers: list/array, containing the outlier scores
    '''

    return np.mean(outliers) - 3 * np.std(outliers)

def three_sigma(outliers):
    '''Returns the three sigmas which can be used as rough guides for outlier detection cut-off

    Parameters:
    -----------
    outliers: list/array, containing the outlier scores
    '''
    mean = np.mean(outliers)
    std = np.std(outliers)

    return mean - std, mean - 2 * std, mean - 3 * std

def outlier_scores_plot(outliers):
    '''Returns a histogram of the outlier scores, which includes the line of the thresholds
    from which observations can be classified as outliers

    Parameters:
    -----------
    df: pd.DataFrame
    '''

    threshold_1, threshold_2, threshold_3 = three_sigma(outliers)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=outliers,
        nbinsx=20,
        marker_color='rgb(175,46,61)'
    ))

    fig.update_layout(
        xaxis={
            "title": "Outlier scores"
        },
        yaxis={
            "title": "Frequency",
            'gridcolor': 'rgb(211,211,211)'
        },
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        shapes=[
            {
                "type": 'line',
                "yref": 'paper',
                'y0': 0,
                'y1': 1,
                'xref': 'x',
                'x0': threshold_3,
                'x1': threshold_3,
                'name': '99.7%'
            },
            {
                "type": 'line',
                "yref": 'paper',
                'y0': 0,
                'y1': 1,
                'xref': 'x',
                'x0': threshold_2,
                'x1': threshold_2,
                'name': '95%',
                'line': {
                    'dash': 'dot'
                }
            },
            {
                "type": 'line',
                "yref": 'paper',
                'y0': 0,
                'y1': 1,
                'xref': 'x',
                'x0': threshold_1,
                'x1': threshold_1,
                'name': '68%',
                'line': {
                    'dash': 'dash'
                }
            },
        ],
        annotations=[
            {
                'x': threshold_1,
                'y': 1,
                'yref': 'paper',
                'xref': 'x',
                'text': round(threshold_1, 3)
            },
            {
                'x': threshold_2,
                'y': 1,
                'yref': 'paper',
                'xref': 'x',
                'text': round(threshold_2, 3)
            },
            {
                'x': threshold_3,
                'y': 1,
                'yref': 'paper',
                'xref': 'x',
                'text': round(threshold_3, 3)
            },
        ]
    )

    return fig

def get_outliers(outliers, threshold):
    '''Returns a list where a 1 indicates an outliers and a 0 indicates an inlier based on the outlier scores
    and a given threshold

    Parameters:
    -----------
    outliers: list/array, containing the outlier scores
    threshold: float, specifying after which point an observation is considered an outlier
    '''
    result = []
    for score in outliers:
        if score < threshold:
            result.append(1)
        else:
            result.append(0)
    return result

def get_nr_outliers(outliers, threshold):
    '''Returns the number of outliers (& percentage of all observations) based on a given threshold

    Parameters:
    -----------
    outliers: list/array, containing the outlier scores
    threshold: float, specifying after which point an observation is considered an outlier
    '''
    pred = get_outliers(outliers, threshold)
    return sum(pred), round(sum(pred) / len(pred), 4) * 100

def get_outlier_idx(outliers, threshold):
    '''Returns a list with the indices of the detected outliers based on a given threshold

    Parameters:
    -----------
    outliers: list/array, containing the outlier scores
    threshold: float, specifying after which point an observation is considered an outlier
    '''
    pred = get_outliers(outliers, threshold)
    result = []
    for idx, score in enumerate(pred):
        if score == 1:
            result.append(idx)
        else:
            continue
    return result

def detection_scores(outliers):
    '''Generates a list of dictionaries, each containing information on the outlier scores:
       - for each threshold it will have the number of outliers and the corresponding indices

    Parameters:
    -----------
    outliers: list/array, containing the outlier scores
    '''
    thresholds = three_sigma(outliers)
    result = []

    for threshold in thresholds:
        nr_outliers, pct_outliers = get_nr_outliers(outliers, threshold)
        result.append({
            'threshold': round(threshold, 3),
            'nr_outliers': "{} ({}%)".format(nr_outliers, round(pct_outliers, 2)),
            'indices': get_outlier_idx(outliers, threshold)
        })

    return result
