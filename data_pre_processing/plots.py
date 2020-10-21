from utils.column_inference import make_missing_np_nan
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly

def distribution_plot(col, col_values, data_type, unique_vals, unique_vals_counts):
    '''Returns a plot of the distribution of a column given its data type, unique values and counts corresponding
    to these unique values.

    Parameters:
    -----------
    col: str, name of a column in the DataFrame
    data_type: str, pandas datatype
    unique_vals: list, contains all the unique values in the column
    unique_vals_counts: list, contains the number of observations per unique value (index should match unique_vals)
    '''

    fig = go.Figure()
    if data_type == 'category':
        index = np.argsort(unique_vals_counts)

        sorted_unique_vals = []
        sorted_unique_vals_counts = []
        for val in index:
            sorted_unique_vals.append(unique_vals[val])
            sorted_unique_vals_counts.append(unique_vals_counts[val])

        fig.add_trace(go.Bar(
            x = sorted_unique_vals_counts,
            y = sorted_unique_vals,
            name = col,
            text = sorted_unique_vals_counts,
            textposition = 'auto',
            marker_color = 'rgb(175,46,61)',
            orientation = 'h'
        ))

        fig.update_layout(
        xaxis_title = 'number of observations',
        yaxis_title = col,
        )

    elif int(len(unique_vals)) < 20:
        index = np.argsort(unique_vals_counts)

        sorted_unique_vals = []
        sorted_unique_vals_counts = []
        for val in index:
            sorted_unique_vals.append(unique_vals[val])
            sorted_unique_vals_counts.append(unique_vals_counts[val])

        fig.add_trace(go.Bar(
            x = sorted_unique_vals_counts,
            y = sorted_unique_vals,
            name = col,
            text = sorted_unique_vals_counts,
            textposition = 'auto',
            marker_color = 'rgb(175,46,61)',
            orientation = 'h'
        ))

        fig.update_layout(
        xaxis_title = 'number of observations',
        yaxis_title = col,
        )

    # uncomment the else-statement if boxplot is preferred over histogram for numerical data
    # else:
    #     fig.add_trace(go.Box(
    #         x = unique_vals,
    #         name = col,
    #         marker_color = 'rgb(175,46,61)'
    #     ))
    #
    #     fig.update_layout(
    #         xaxis_title ='value',
    #     )

    # comment the else-statement if boxplot is preferred over histogram for numerical data
    else:
        fig.add_trace(go.Histogram(
            x = col_values,
            nbinsx=10,
            text = unique_vals_counts,
            marker_color = 'rgb(175,46,61)'
        )),

        fig.update_layout(
            xaxis_title = col,
            yaxis_title ='Frequency',
            bargap=0.01
        )

    fig.update_layout(margin=dict(
        l=10,
        r=10,
        b=10,
        t=10),
    plot_bgcolor='rgba(0,0,0,0)',
    yaxis ={'gridcolor': 'rgb(211,211,211)'})

    return fig

def missing_values_plot(df, inferred):
    '''Returns a bar chart of the number of observations per column

    Parameters:
    -----------
    df: pd.DataFrame
    inferred: dict, result of inference (see report.py) (dict containing information for each column in the dataframe)
    '''

    nr_observations = [df.shape[0]] * df.shape[1]
    nr_missing = [0] * df.shape[1]

    for idx, col in enumerate(list(df.columns)):
        nr_observations[idx] -= inferred[col]['nr_missing']
        nr_missing[idx] += inferred[col]['nr_missing']

    fig = go.Figure()

    fig.add_trace(go.Bar(
            x = df.columns,
            y = nr_observations,
            text = nr_missing,
            textposition = 'auto',
            marker_color = 'rgb(175,46,61)',
        ))

    fig.update_layout(
        xaxis_title = 'features',
        yaxis_title ='number of observations',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
        )

    fig.update_xaxes(
        ticks = 'outside',
        tickson = 'boundaries',
        ticklen = 20,
        tickfont = {'size': 8}
    )

    return fig.to_json()

def correlations_heatmap(df):
    '''Returns a heatmap of the correlations between the numerical columns of a dataframe

    Parameters:
    -----------
    df: pd.DataFrame
    '''
    correlations = df.corr()
    columns = correlations.columns
    correlations = correlations.to_numpy().round(decimals=2)

    fig = ff.create_annotated_heatmap(
        z = correlations,
        x = list(columns),
        y = list(columns),
        colorscale = 'Geyser'
    )
    fig.update_xaxes(
        ticks = 'outside',
        tickson = 'boundaries',
        ticklen = 20,
        tickfont = {'size': 8}
    )

    fig.update_yaxes(
        ticks = 'outside',
        tickson = 'boundaries',
        ticklen = 20,
        tickfont = {'size': 8}
    )
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 8
    
    fig.update_layout(
      plot_bgcolor='rgba(0,0,0,0)',
      margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig.to_json()

def scatter_plot(df, col1, col2):
    '''Returns a scatter plot between two columns

    Parameters:
    -----------
    df: pd.DataFrame
    col1: str, corresponding to the first column
    col2: str, corresponding to the second column
    '''
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[col1],
        y=df[col2],
        mode="markers"
    ))

    fig.update_layout(
        xaxis = {
            "title": col1},
        yaxis = {"title": col2}
    )
    return fig

def line_plot(df, target=False, color=False, plot_range=None):
    '''Returns a lineplot of a prespecified column

    Parameters:
    -----------
    df: pd.DataFrame
    target: str, name of the column, default is the last column of the dataframe (which is usually the target variable)
    color: str, name of a column which will illustrate the color of the lines (in combination with the target)
    plot_range: list with 2 values, indicating the range of the x-axis
    '''
    if not target:
        target = df.columns[-1] #sets it to the "target" variable on default

    df = make_missing_np_nan(df, replace_with='nan')

    if color:
        df_small = df[[target, color]]
        fig = go.Figure()

        for val in set(df_small[color]):
            df_new = df_small[df_small[color] == val]
            unique_vals, unique_vals_counts = np.unique([int_element for int_element in df_new[target].tolist()], return_counts=True)
            fig.add_trace(go.Scatter(x=unique_vals, y=unique_vals_counts,
                                 mode='lines',
                                 name=val))

        fig.update_layout(xaxis = {'range': plot_range,
                                   'title': target},
                          yaxis = {'title': 'count'})
        return fig

    else:
        unique_vals, unique_vals_counts = np.unique([int_element for int_element in df[target].tolist()], return_counts=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=unique_vals, y=unique_vals_counts,
                        mode='lines',
                        name=target))
        fig.update_layout(xaxis = {'range': plot_range,
                                   'title': target},
                          yaxis = {'title': 'count'})

        return fig

def area_plot(df, col=False, color=False, plot_range=None):
    ''' WIP

    '''
    if not col:
        col = df.columns[-1]

    X_values = sorted(list(set(df[col])))

    fig = go.Figure()

    df_results = pd.DataFrame(columns = ['unique_values', 'counts', 'value'])

    for val in set(df[col]):
        df_new = df[df[col] == val]
        unique_vals, unique_vals_counts = np.unique([int_element for int_element in df_new[col].tolist()], return_counts=True)

        df_col = df_new[color].value_counts(normalize=True).rename_axis('unique_values').reset_index(name='counts')
        df_col = df_col.sort_values(by=['unique_values'])
        df_col['value'] = val

        df_results = df_results.append(df_col).sort_values(by=['unique_values', 'value'])

    for val in set(df_results['unique_values']):
        df_new = df_results[df_results['unique_values'] == val]

        fig.add_trace(go.Scatter(
            x=df_new['value'],
            y=df_new['counts'],
            mode = 'lines',
            stackgroup = 'one',
            name=val,
        ))

    fig.update_layout(
        xaxis = {'range': plot_range,
                'title': col},
        yaxis = {'title': 'proportion'}
    )
    return fig
