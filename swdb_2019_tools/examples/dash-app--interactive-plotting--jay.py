import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go


# Import allensdk modules for loading and interacting with the data
from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc


#####################

# This will find the appropriate path to the data based on your platform.
# You may need to edit the strings in this cell based on your configuration.

import platform
platstring = platform.platform()

if 'Darwin' in platstring:
    # macOS
    data_root = "/Volumes/Brain2019/"
elif 'Windows'  in platstring:
    # Windows (replace with the drive letter of the hard drive)
    data_root = "E:/"
elif ('amzn1' in platstring):
    # then on AWS
    data_root = "/data/"
else:
    # then linux (default here is for Ubuntu - insert your username; your distribution may differ)
    data_root = "/media/$USERNAME/Brain2019"

cache_path = os.path.join(data_root, 'dynamic-brain-workshop/visual_behavior/2019')
#####################



def dff_change_cell(tr_loc):

    '''
    Input: session.trial_response_df (DataFrame) "tr_loc"
    Output: df/f traces for each neuron on go trials, organized by cell
            I.e., #neurons x #go trials x #df data points numpy array
    '''

    cell = tr_loc['cell_specimen_id'].unique()[0]    
    cell_df = tr_loc.groupby('cell_specimen_id').get_group(cell)
    traces_fr_cell = cell_df[ cell_df['go']==True ]['dff_trace'].values

    num_cells = tr_loc['cell_specimen_id'].nunique()
    num_traces = traces_fr_cell.shape[0]
    num_trace_samples = traces_fr_cell[0].shape[0]
    traces = np.empty( (num_cells,num_traces,num_trace_samples) )

    for i in range(0, num_cells):
        cell = tr_loc['cell_specimen_id'].unique()[i]
        cell_df = tr_loc.groupby('cell_specimen_id').get_group(cell)
        traces_fr_cell = cell_df[ cell_df['go']==True ]['dff_trace'].values

        for j, t in enumerate(traces_fr_cell):
            traces[i][j] = t

    return traces


##################################################################
##################################################################

# Set up initial values
cache = bpc.BehaviorProjectCache(cache_path)

# get the table of all active experiment sessions for this dataset
experiments = cache.experiment_table
active_experiments = experiments[experiments.passive_session==False]
experiment_id = active_experiments.ophys_experiment_id.values[5]
# get a session from the cache
session = cache.get_session(experiment_id)



# Set up global variables (i.e., variables to be shared by Dash callback
# funcitons).  They need to be lists.

# get the trial response dataframe and assign it to 'tr'
tr = [session.trial_response_df]
num_cells = [ 1 ]
cells_dash = [ np.arange(0,1) ]
choices_dash = active_experiments.ophys_experiment_id.values



# Dash setup:

# How it looks
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# For dynamically-populated menus
app.config['suppress_callback_exceptions'] = True



#####################################################################
#####################################################################

# Dash Layout
app.layout = html.Div([
    html.Div([
        'Select Experiment ID',
        dcc.Dropdown(
            id='exp-id',
            options=[{'label': i, 'value': i} for i in choices_dash],
            value=choices_dash[0]
        ),
    ],
    style={'width': '30%', 'display': 'inline-block'}),
    dcc.Graph(id='calcium-traces'),
    html.Div(id='plot-all-button', 
            children = [html.Button('Plot all cells', id='button--plot-all')
            ], style=None),  
    html.Div(id='plot-all', style={'display': None}),  
    html.Div('Select Cell Number'), 
    html.Div(id='cells', style=None),
    html.Div(id='signal--cells', style={'display': 'none'}),
    html.Div(id='signal--exp-id', style={'display': 'none'}),
])

#####################################################################

# Dash callbacks
@app.callback(
    dash.dependencies.Output('plot-all', 'children'),
    [dash.dependencies.Input('button--plot-all', 'n_clicks')
    ])
def plot_all(n_clicks):#, signal__exp_id):
    '''
    Button to plot all cells from experiment
    '''
    cells_dash[0] = np.arange( num_cells[0] )

@app.callback(
        Output('signal--exp-id', 'children'),
        [Input('exp-id', 'value')
         ])
def update_session(choice):
    '''
    Dropdown menu to choose experiment ID
    '''
    # get a session from the cache
    session = cache.get_session(choice)
    # get the trial response dataframe and assign it to 'tr'
    tr[0] = session.trial_response_df
    get_traces = dff_change_cell( tr[0] )
    num_cells[0] = get_traces.shape[1]
    cells_dash[0] = np.arange(0,1)


@app.callback(
        Output( 'signal--cells', 'children' ),
        [Input( 'cell-selection', 'value' ),
         ])
def update_cells( cell_selection ):
    '''
    Assigns array of cells to plot
    '''
    cells_dash[0] = cell_selection

@app.callback(
        Output('cells', 'children'),
        [Input('signal--exp-id', 'children'),
         Input('plot-all', 'children')]
        )
def update_cells_displayed( signal__exp_id, plot_all ):
    '''
    Dynamically-populated menu to select which cells to plot.
    Menu repopulates when a new experiment is chosen.
    '''
    return html.Div([
           dcc.Dropdown(
                id='cell-selection',
                options=[{'label': str(i), 'value': i} for i in np.arange( num_cells[0] )], multi=True, value=cells_dash[0],
            )])
            
@app.callback(
    Output('calcium-traces', 'figure'),
    [Input('signal--cells', 'children'),
     Input('signal--exp-id', 'children')]
    )
def update_trace_graph( signal__cells, signal__exp_id ):
    '''
    Graphs cell traces around go trials.  
    '''
    traces = []
    get_traces = dff_change_cell( tr[0] )
    traces_avg_cell = np.mean( get_traces, axis=0 )
    time_seconds = tr[0].iloc[0].dff_trace_timestamps  - tr[0].iloc[0].change_time
    for i in cells_dash[0]:
        traces.append( go.Scatter(
            x=time_seconds,
            y=traces_avg_cell[i],
            name='Cell #{}'.format(str(i)),
            mode='lines', 
               line={'width': 3}
        ) )        
        
    figure = {
    'data': traces,
    'layout': go.Layout(
        xaxis={
            'title': 'time (sec)',
            'type': 'linear'
        },
        yaxis={
            'title': 'df/f',
            'type': 'linear'
        },
        margin={'l': 80, 'b': 40, 't': 20, 'r': 0},
        hovermode='closest'
    )
    }
    return figure




if __name__ == '__main__':
    app.run_server(debug=True, port=8050)