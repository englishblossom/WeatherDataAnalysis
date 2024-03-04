import os

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import norm, kstest, shapiro, normaltest

# cleaning data
df = pd.read_csv('weatherAUS.csv')
print((df.isna().sum()) / len(df) * 100)
sunshine_median = df['Sunshine'].median()
evaporation_median = df['Evaporation'].median()
df['Sunshine'] = df['Sunshine'].fillna(sunshine_median)
df['Evaporation'] = df['Evaporation'].fillna(evaporation_median)
df.drop(columns=['Cloud9am', 'Cloud3pm'], inplace=True)
df.dropna(inplace=True)
print("after dropping rows that have null values:\n", (df.isnull().sum()) / len(df) * 100)
df['Date'] = pd.to_datetime(df['Date'])
df_og = df.copy()
cleaned_df = df.copy()
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Create the Dash app
my_app = dash.Dash('my project')
server = my_app.server
unique_cities = df['Location'].unique()
dropdown_cities = [{'label': city, 'value': city} for city in unique_cities]
numeric_df = df.drop(columns=['Date', 'Month', 'Year']).select_dtypes(include='number')

numeric_columns = numeric_df.columns


def calculate_test_results(column, test_name):
    if test_name == 'K-S Test':
        stats, p_value = kstest(column, 'norm')
    elif test_name == 'Shapiro Test':
        stats, p_value = shapiro(column)
    elif test_name == 'da_k_squared Test':
        stats, p_value = normaltest(column)

    interpretation = "Normal" if p_value > 0.05 else "Not Normal"

    return {
        'Statistics': stats,
        'p-value': p_value,
        'Interpretation': interpretation
    }


def generate_jointplot_images(df, features, image_paths):
    for feature, image_path in zip(features, image_paths):
        jp = sns.jointplot(data=df, x=feature[0], y=feature[1], kind='kde', hue='RainTomorrow')
        # Save the plot as an image\
        jp.savefig(image_path)
        plt.close()


def generate_jointplot_images2(df, features, image_paths):
    for feature, image_path in zip(features, image_paths):
        jp = sns.jointplot(data=df, x=feature[0], y=feature[1], hue='RainTomorrow')
        # Save the plot as an image
        jp.savefig(image_path)
        plt.close()


def find_min_values(dataframe):
    min_values = dataframe.min()
    return min_values


def calculate_constants_to_add(min_values):
    constants_to_add = {}
    for column, min_value in min_values.items():
        if min_value <= 0:
            constants_to_add[column] = abs(min_value) + 1
    return constants_to_add


stats_data = []

for column in numeric_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    stats_data.append({'Column': column, 'Q1': round(q1, 2), 'Q3': round(q3, 2), 'IQR': round(iqr, 2)})

statsdf = pd.DataFrame(stats_data)


def remove_outliers(dataframe, columns):
    clean_data = dataframe.copy()
    for column in columns:
        q1 = np.percentile(clean_data[column], 25)
        q3 = np.percentile(clean_data[column], 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        clean_data = clean_data[(clean_data[column] >= lower_bound) & (clean_data[column] <= upper_bound)]

    return clean_data


# Remove outliers from all columns
cleaned_data = remove_outliers(df, numeric_columns)

min_values = find_min_values(cleaned_data[numeric_columns])

constants_to_add = calculate_constants_to_add(min_values)

# transform data
transformed_data = pd.DataFrame()
temp_data = cleaned_data.copy()
for column in numeric_columns:
    if column in constants_to_add:
        temp_data[column] += constants_to_add[column]
    if (temp_data[column] <= 0).any():
        raise ValueError(f"All values in {column} must be positive for Box-Cox transformation.")
    _, optimal_lambda = stats.boxcox(temp_data[column])
    transformed_data_column = stats.boxcox(temp_data[column], lmbda=optimal_lambda)
    transformed_data[column] = transformed_data_column

stats_data = []
for column in numeric_columns:
    stats_data.append({
        'Column': column,
        'Mean': round(df[column].mean(), 2),
        'Median': round(df[column].median(), 2),
        'Mode': round(df[column].mode()[0], 2),
        'Standard Deviation': round(df[column].std(), 2),
        'Minimum': df[column].min(),
        'Maximum': df[column].max()
    })

all_stats_df = pd.DataFrame(stats_data)

# Define your features and corresponding image paths
features = [('Humidity3pm', 'Pressure3pm'), ('Temp3pm', 'WindSpeed3pm'), ('MinTemp', 'Sunshine')]
image_paths = ['assets/jointplot4.png', 'assets/jointplot5.png', 'assets/jointplot6.png']
generate_jointplot_images2(df, features, image_paths)

# Define your features and corresponding image paths
features = [('Humidity3pm', 'Temp3pm'), ('Pressure3pm', 'WindSpeed3pm'), ('MaxTemp', 'Evaporation')]
image_paths = ['assets/jointplot1.png', 'assets/jointplot2.png', 'assets/jointplot3.png']
generate_jointplot_images(df, features, image_paths)


my_app.layout = html.Div([
    html.H1('Rainfall in Australia Dataset Analysis', style={'color': 'blue', 'textAlign': 'center'}),
    html.Br(),
    html.Br(),
    # Plot 1 - Monthly Rainfall
    html.H3('Select Location for viewing monthly rainfall', style={'color': 'green'}),
    dcc.Dropdown(
        id='city-drop-plot1',
        options=dropdown_cities,
        value=unique_cities[0]
    ),
    dcc.Graph(id='plot1'),
    html.Br(),
    html.Br(),
    # Plot 2 - Average Max and Min Temperature
    html.H3('Select Location for viewing monthly average max and min temperature', style={'color': 'green'}),
    dcc.Dropdown(
        id='city-drop-plot2',
        options=dropdown_cities,
        value=unique_cities[0]
    ),
    dcc.Graph(id='plot2'),
    html.Br(),
    html.Br(),
    # Plot 3 - Correlation Heatmap
    html.H3('Select features for heatmap showing correlation', style={'color': 'green'}),
    dcc.Checklist(
        id='column-checklist',
        options=[{'label': column, 'value': column} for column in numeric_columns],
        value=numeric_columns,
        labelStyle={'display': 'block'}
    ),
    html.Div([
        dcc.Graph(id='plot3')
    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    html.Div([
        html.H3('Select features', style={'color': 'green'}),
        html.Br(),
        dcc.Checklist(
            id='checklist-subplot-titles',
            options=[
                {'label': 'Wind Speed at 9AM', 'value': 'WindSpeed9am'},
                {'label': 'Wind Speed at 3PM', 'value': 'WindSpeed3pm'},
                {'label': 'Humidity at 9AM', 'value': 'Humidity9am'},
                {'label': 'Humidity at 3PM', 'value': 'Humidity3pm'},
                {'label': 'Pressure at 9AM', 'value': 'Pressure9am'},
                {'label': 'Pressure at 3PM', 'value': 'Pressure3pm'},
                {'label': 'Temperature at 9AM', 'value': 'Temp9am'},
                {'label': 'Temperature at 3PM', 'value': 'Temp3pm'},
            ],
            value=['WindSpeed9am', 'WindSpeed3pm'],
            labelStyle={'display': 'block'}
        ),
        html.Br(),
        html.Br(),
        html.H3('Select bin size', style={'color': 'green'}),
        html.Br(),
        dcc.Slider(
            id='bin-size-slider',
            min=1,
            max=50,
            step=1,
            value=10,
            marks={i: str(i) for i in range(1, 51)},
            tooltip={'placement': 'bottom', 'always_visible': True}
        )
    ]),
    html.Div([
        dcc.Graph(id='plot4')
    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    html.Div([
        html.H3('Select feature for count plot', style={'color': 'green'}),
        html.Br(),
        dcc.RadioItems(
            id='column-selector-plot5',
            options=[
                {'label': 'Wind Direction at 9am', 'value': 'WindDir9am'},
                {'label': 'Wind Direction at 3pm', 'value': 'WindDir3pm'},
                {'label': 'Wind Gust Direction', 'value': 'WindGustDir'},
            ],
            value='WindDir9am',  # Default selected value
            labelStyle={'display': 'block'}  # Display radio items vertically
        )]),
    html.Div([
        dcc.Graph(id='plot5')
    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    html.H3('Select features for pairplot', style={'color': 'green'}),
    dcc.Checklist(
        id='column-checklist-plot6',
        options=[{'label': column, 'value': column} for column in numeric_columns],
        value=numeric_columns,
        labelStyle={'display': 'block'}
    ),
    html.Div([
        dcc.Graph(id='plot6')
    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    html.H3('Select year range', style={'color': 'green'}),
    html.Div([
        dcc.RangeSlider(
            id='year-range-slider',
            marks={int(year): str(year) for year in df['Year'].unique()},
            min=df['Year'].min(),
            max=df['Year'].max(),
            value=[df['Year'].min(), df['Year'].max()],
        )]),
    html.Div([
        dcc.Graph(id='plot7')
    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    dcc.Tabs(id='kde-tab', value='kde-tab', children=[
        dcc.Tab(label='Histogram with KDE + Rugplot', value='kde_hist-tab'),
        dcc.Tab(label='KDE + Rugplot', value='no-hist-tab')
    ]),
    html.H3('Select the feature for histogram+kde', style={'color': 'green'}),
    html.Div([
        dcc.Dropdown(
            id='numerical-feature-dropdown',
            options=[{'label': col, 'value': col} for col in numeric_columns],
            value=numeric_columns[0],
            style={'width': '50%'}
        ), ]),
    html.Div([
        dcc.Graph(id='plot8')
    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    html.H3('Select the pie-charts', style={'color': 'green'}),
    dcc.Dropdown(
        id='pie-chart-dropdown',
        options=[
            {'label': 'Rainfall Distribution', 'value': 'rainfall'},
            {'label': 'Total rainfall by Location', 'value': 'location'},
            {'label': 'Wind Gust Direction Distribution', 'value': 'wind_gust_dir'},
            {'label': 'Total rainfall for Wind Direction at 9 AM', 'value': 'wind_dir_9am'},
            {'label': 'Total rainfall for Wind Direction at 3 PM', 'value': 'wind_dir_3pm'},
            {'label': 'Rain Today', 'value': 'rain_today'},
            {'label': 'Rain Tomorrow', 'value': 'rain_tomorrow'}
        ],
        value=['rainfall'],
        multi=True
    ),
    html.Div([
        dcc.Graph(id='plot9')
    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    html.H3("Select the column for QQ plot", style={'color': 'green'}),
    dcc.RadioItems(
        id='column-selector-qqplot',
        options=[{'label': 'Temperature at 9 am', 'value': 'Temp9am'},
                 {'label': 'Pressure at 9 am', 'value': 'Pressure9am'},
                 {'label': 'Humidity at 9 am', 'value': 'Humidity9am'},
                 {'label': 'Wind gust speed', 'value': 'WindGustSpeed'}
                 ],
        value='Temp9am',
        labelStyle={'display': 'block'},
    ),
    html.Div([
        dcc.Loading(
            id="loading-qq-plots",
            children=dcc.Graph(id='qq-plot'),
            type="default"
        )

    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    dcc.Tabs(id='violin-tabs', value='tab-1', children=[
        dcc.Tab(label='Wind Gust Speed by Location', value='tab-1'),
        dcc.Tab(label='Wind Gust Speed by Month', value='tab-2'),
        dcc.Tab(label='Humidity at 9am by Month', value='tab-3'),
        dcc.Tab(label='Humidity at 3pm  by Month', value='tab-4'),

    ]),
    html.Div([
        dcc.Loading(
            id="loading-violin-plots",
            children=dcc.Graph(id='violin-plot'),
            type="default"
        )

    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    html.Center(html.H2("Select the column for Box plot", style={'color': 'green'})),
    html.Div([
        dcc.Tabs(id='boxen-tabs', value='btab-1', children=[
            dcc.Tab(label='Pressure', value='btab-1'),
            dcc.Tab(label='Temperature', value='btab-2'),
            dcc.Tab(label='Humidity', value='btab-3'),
            dcc.Tab(label='Windspeed', value='btab-4'),
            dcc.Tab(label='Rainfall', value='btab-5'),

        ])
    ]),
    html.Div([
        dcc.Loading(
            id="loading-boxen-plots",
            children=dcc.Graph(id='boxen-plot'),
            type="default"
        )
    ], style={
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center'
    }),
    html.Br(),
    html.Br(),
    html.Center(html.H3("Select features for Regression plot", style={'color': 'green'})),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='xaxis-dropdown-reg',
                options=[{'label': i, 'value': i} for i in numeric_columns],
                value=numeric_columns[0]
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-dropdown-reg',
                options=[{'label': i, 'value': i} for i in numeric_columns],
                value=numeric_columns[1]
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ]),
    html.Div([
        dcc.Loading(
            id="loading-reg-plots",
            children=dcc.Graph(id='reg-plot'),
            type="default"
        )

    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    html.Center(html.H2("Select the column for Strip plot", style={'color': 'green'})),
    html.Div([
        dcc.Tabs(id='strip-tabs', value='WindGustDir', children=[
            dcc.Tab(label='WindGustDir', value='WindGustDir'),
            dcc.Tab(label='WindDir3pm', value='WindDir3pm'),
            dcc.Tab(label='WindDir9am', value='WindDir9am'),
            dcc.Tab(label='Location', value='Location'),
            dcc.Tab(label='RainToday', value='RainToday'),

        ])
    ]),
    html.Div([
        dcc.Loading(
            id="loading-strip-plots",
            children=dcc.Graph(id='strip-plot'),
            type="default"
        )
    ], style={
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center'
    }),
    html.Br(),
    html.Br(),
    html.Center(html.H2("Select the column for Area plot", style={'color': 'green'})),
    html.Div([
        dcc.Tabs(id='area-tabs', value='MaxMinTemp', children=[
            dcc.Tab(label='Max and Min Temperatures', value='MaxMinTemp'),
            dcc.Tab(label='Rainfall', value='Rainfall'),
            dcc.Tab(label='WindGustSpeed', value='WindGustSpeed'),
            dcc.Tab(label='Sunshine', value='Sunshine'),
            dcc.Tab(label='Evaporation', value='Evaporation'),

        ])
    ]),
    html.Div([
        dcc.Loading(
            id="loading-area-plots",
            children=dcc.Graph(id='area-plot'),
            type="default"
        )
    ], style={
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center'
    }),
    html.Br(),
    html.Br(),
    html.Center(html.H3("Select features for Hexbin plot", style={'color': 'green'})),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='hex-in1',
                options=[{'label': i, 'value': i} for i in numeric_columns],
                value=numeric_columns[0]
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='hex-in2',
                options=[{'label': i, 'value': i} for i in numeric_columns],
                value=numeric_columns[1]
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ]),
    html.Div([
        dcc.Loading(
            id="loading-hex-plots",
            children=dcc.Graph(id='hex-plot'),
            type="default"
        )

    ],
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        }
    ),
    html.Br(),
    html.Br(),
    html.Center(html.H2("Joint plots", style={'color': 'green'})),
    html.Div([
        dcc.Tabs(id='joint-tabs', value='tab-1', children=[
            dcc.Tab(label='Humidity vs Temperature', value='tab-1'),
            dcc.Tab(label='pressure vs wind speed', value='tab-2'),
            dcc.Tab(label='evaporation vs MaxTemp', value='tab-3'),
            dcc.Tab(label='Humidity vs pressure', value='tab-4'),
            dcc.Tab(label='temperature vs wind speed', value='tab-5'),
            dcc.Tab(label='sunshine vs MinTemp', value='tab-6'),

        ])
    ]),
    html.Div([
        dcc.Loading(
            id="loading-joint-plots",
            children=html.Img(id='joint-plot'),
            type="default"
        )
    ], style={
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center'
    }),
    html.Br(),
    html.Br(),
    html.Center(html.H2("Normality tests", style={'color': 'green'})),
    html.Div([
        dcc.Tabs([
            dcc.Tab(label='K-S Test', children=[
                dash_table.DataTable(
                    id='ks-test-table',
                    columns=[
                        {'name': 'Numeric Column', 'id': 'Column'},
                        {'name': 'Statistics', 'id': 'Statistics'},
                        {'name': 'p-value', 'id': 'p-value'},
                        {'name': 'Interpretation', 'id': 'Interpretation'}
                    ],
                    data=[
                        {'Column': col, **calculate_test_results(df[col], 'K-S Test')}
                        for col in numeric_columns
                    ],
                )
            ]),
            dcc.Tab(label='Shapiro Test', children=[
                dash_table.DataTable(
                    id='shapiro-test-table',
                    columns=[
                        {'name': 'Numeric Column', 'id': 'Column'},
                        {'name': 'Statistics', 'id': 'Statistics'},
                        {'name': 'p-value', 'id': 'p-value'},
                        {'name': 'Interpretation', 'id': 'Interpretation'}
                    ],
                    data=[
                        {'Column': col, **calculate_test_results(df[col], 'Shapiro Test')}
                        for col in numeric_columns
                    ],
                )
            ]),
            dcc.Tab(label='da_k_squared Test', children=[
                dash_table.DataTable(
                    id='da-k-squared-test-table',
                    columns=[
                        {'name': 'Numeric Column', 'id': 'Column'},
                        {'name': 'Statistics', 'id': 'Statistics'},
                        {'name': 'p-value', 'id': 'p-value'},
                        {'name': 'Interpretation', 'id': 'Interpretation'}
                    ],
                    data=[
                        {'Column': col, **calculate_test_results(df[col], 'da_k_squared Test')}
                        for col in numeric_columns
                    ],
                )
            ]),
        ]),
    ]),
    html.Br(),
    html.Br(),
    html.Center(html.H2("Normality tests after transformation", style={'color': 'green'})),
    html.Div([
        dcc.Tabs([
            dcc.Tab(label='K-S Test', children=[
                dash_table.DataTable(
                    id='ks-test-table-trans',
                    columns=[
                        {'name': 'Numeric Column', 'id': 'Column'},
                        {'name': 'Statistics', 'id': 'Statistics'},
                        {'name': 'p-value', 'id': 'p-value'},
                        {'name': 'Interpretation', 'id': 'Interpretation'}
                    ],
                    data=[
                        {'Column': col, **calculate_test_results(transformed_data[col], 'K-S Test')}
                        for col in numeric_columns
                    ],
                )
            ]),
            dcc.Tab(label='Shapiro Test', children=[
                dash_table.DataTable(
                    id='shapiro-test-table-trans',
                    columns=[
                        {'name': 'Numeric Column', 'id': 'Column'},
                        {'name': 'Statistics', 'id': 'Statistics'},
                        {'name': 'p-value', 'id': 'p-value'},
                        {'name': 'Interpretation', 'id': 'Interpretation'}
                    ],
                    data=[
                        {'Column': col, **calculate_test_results(transformed_data[col], 'Shapiro Test')}
                        for col in numeric_columns
                    ],
                )
            ]),
            dcc.Tab(label='da_k_squared Test', children=[
                dash_table.DataTable(
                    id='da-k-squared-test-table-trans',
                    columns=[
                        {'name': 'Numeric Column', 'id': 'Column'},
                        {'name': 'Statistics', 'id': 'Statistics'},
                        {'name': 'p-value', 'id': 'p-value'},
                        {'name': 'Interpretation', 'id': 'Interpretation'}
                    ],
                    data=[
                        {'Column': col, **calculate_test_results(transformed_data[col], 'da_k_squared Test')}
                        for col in numeric_columns
                    ],
                )
            ]),
        ]),
    ]),
    html.Br(),
    html.Br(),
    html.Center(html.H2("Table for data statistics", style={'color': 'green'})),
    html.Div([
        dash_table.DataTable(
            id='stats-table',
            columns=[{"name": i, "id": i} for i in all_stats_df.columns],
            data=all_stats_df.to_dict('records'),
            style_table={'overflowX': 'auto'},
        )
    ]),
    html.Button("Download data stats CSV", id="btn_csv_stats"),
    dcc.Download(id="download-datastats-csv"),
    html.Br(),
    html.Br(),
    html.Center(html.H2("Table for outlier detection", style={'color': 'green'})),
    html.Div([
        dash_table.DataTable(
            id='iqr-table',
            columns=[{"name": i, "id": i} for i in statsdf.columns],
            data=statsdf.to_dict('records'),
            style_table={'overflowX': 'auto'},
        )
    ]),
    html.Button("Download iqr analysis CSV", id="btn_csv_iqr"),
    dcc.Download(id="download-iqr-csv")

])


@my_app.callback(
    Output("download-datastats-csv", "data"),
    Input("btn_csv_stats", "n_clicks"),
    prevent_initial_call=True)
def download_stats_csv(n_clicks):
    return dcc.send_data_frame(all_stats_df.to_csv, filename="data_table_stats.csv")


@my_app.callback(
    Output("download-iqr-csv", "data"),
    Input("btn_csv_iqr", "n_clicks"),
    prevent_initial_call=True)
def download_iqr_csv(n_clicks):
    return dcc.send_data_frame(statsdf.to_csv, filename="data_table_iqr.csv")


@my_app.callback(
    Output('plot1', 'figure'),
    Input('city-drop-plot1', 'value')
)
def update_plot1(selected_city):
    df_filtered = df[df['Location'] == selected_city]
    monthly_rainfall = df_filtered.groupby(['Year', 'Month'])['Rainfall'].sum().reset_index()
    fig = px.bar(monthly_rainfall, x='Month', y='Rainfall', color='Year', color_continuous_scale='Viridis')
    font_title = {'family': 'serif', 'size': 26, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 20, 'color': 'darkred'}

    fig.update_layout(
        title=f'Monthly Rainfall in city {selected_city}',
        title_font=font_title,
        title_x=0.5,
        xaxis_title='Month',
        xaxis_title_font=font_labels,
        yaxis_title='Rainfall in mm',
        yaxis_title_font=font_labels,
        xaxis=dict(
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )
    )

    return fig


@my_app.callback(
    Output('plot2', 'figure'),
    Input('city-drop-plot2', 'value')
)
def update_plot2(selected_city):
    maxTemp = df.groupby(['Location', 'Year', 'Month'])['MaxTemp'].mean().reset_index()
    minTemp = df.groupby(['Location', 'Year', 'Month'])['MinTemp'].mean().reset_index()
    maxTemp = maxTemp[maxTemp['Location'] == selected_city]
    minTemp = minTemp[minTemp['Location'] == selected_city]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Average MaxTemp Monthly", " Average MinTemp Monthly"))
    maxTemp_colors = px.colors.sequential.YlOrRd
    minTemp_colors = px.colors.sequential.Viridis
    for i, year in enumerate(maxTemp['Year'].unique()):
        year_data = maxTemp[maxTemp['Year'] == year]
        color_index = i % len(maxTemp_colors)
        fig.add_trace(
            go.Scatter(x=year_data['Month'], y=year_data['MaxTemp'], mode='lines+markers', name=f'Year(maxTemp) {year}',
                       line=dict(color=maxTemp_colors[color_index])), row=1, col=1)

    for i, year in enumerate(minTemp['Year'].unique()):
        year_data = minTemp[minTemp['Year'] == year]
        color_index = i % len(minTemp_colors)
        fig.add_trace(
            go.Scatter(x=year_data['Month'], y=year_data['MinTemp'], mode='lines+markers', name=f'Year(minTemp) {year}',
                       line=dict(color=minTemp_colors[color_index])), row=2, col=1)

    font_labels = {'family': 'serif', 'size': 16, 'color': 'darkred'}
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    fig.update_xaxes(
        tickvals=list(range(1, 13)),
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], row=2, col=1
        , title_text='Month', title_font=font_labels)

    fig.update_yaxes(title_text='Avg MaxTemp (°C)', row=1, col=1, title_font=font_labels)
    fig.update_yaxes(title_text='Avg MinTemp (°C)', row=2, col=1, title_font=font_labels)

    fig.update_layout(
        title=f"Avg monthly min and max temperatures for {selected_city}",
        title_font=font_title,
        title_x=0.5,
        showlegend=True,
        legend=dict(
            traceorder='normal',
            y=0.95,
            x=1.05,
        ),
        legend_tracegroupgap=0
    )

    return fig


@my_app.callback(
    Output('plot3', 'figure'),
    Input('column-checklist', 'value')
)
def update_plot3(columns):
    correlation_matrix = df[columns].corr()
    fig = px.imshow(correlation_matrix, color_continuous_scale='YlOrRd')
    fig.update_traces(hovertemplate='Correlation: %{z:.2f}<extra></extra>')

    font_title = {'family': 'serif', 'size': 26, 'color': 'blue'}

    fig.update_layout(
        title='Correlation Heatmap for numeric features',
        height=800,
        width=800,
        title_font=font_title,
        title_x=0.5,
    )
    return fig


@my_app.callback(
    Output('plot4', 'figure'),
    Input('checklist-subplot-titles', 'value'),
    Input('bin-size-slider', 'value')
)
def update_plot4(selected_titles, bin_size):
    num_selected = len(selected_titles)
    num_rows = num_selected // 2 + num_selected % 2
    fig = make_subplots(rows=num_rows, cols=2, subplot_titles=selected_titles)

    colors = ['green', 'green', 'orange', 'orange', 'red', 'red', 'blue', 'blue']
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 16, 'color': 'darkred'}
    for i, title in enumerate(selected_titles):
        column_data = df[title]
        trace = go.Histogram(x=column_data, marker=dict(color=colors[i]), nbinsx=bin_size)
        row_num = i // 2 + 1
        col_num = i % 2 + 1
        fig.add_trace(trace, row=row_num, col=col_num)

        fig.update_xaxes(title_text=title, row=row_num, col=col_num, title_font=font_labels, automargin=True)
        fig.update_yaxes(title_text="Frequency", row=row_num, col=col_num, title_font=font_labels, automargin=True)

    fig.update_layout(
        title='Distribution plots',
        title_font=font_title,
        title_x=0.5,
        showlegend=False,
        height=1000,
        width=800
    )

    return fig


@my_app.callback(
    Output('plot5', 'figure'),
    Input('column-selector-plot5', 'value')
)
def update_plot5(selected_column):
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 16, 'color': 'darkred'}
    fig = px.histogram(df, x=selected_column, color_discrete_sequence=['darkred'])
    fig.update_yaxes(title_text='Count', title_font=font_labels)
    fig.update_xaxes(title_text=selected_column, title_font=font_labels)
    fig.update_layout(
        title=f'Count Plot for {selected_column}',
        title_font=font_title,
        title_x=0.5
    )

    return fig


@my_app.callback(
    Output('plot6', 'figure'),
    Input('column-checklist-plot6', 'value')
)
def update_plot6(selected_columns):
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    fig = px.scatter_matrix(
        df,
        dimensions=selected_columns,
        title='Pairplot',
        width=1500,
        height=1500,
        color='RainTomorrow'

    )
    fig.update_traces(
        marker=dict(size=2)
    )

    fig.update_layout(
        title_font=font_title,
        title_x=0.5
    )

    return fig


@my_app.callback(
    Output('plot7', 'figure'),
    Input('year-range-slider', 'value')
)
def update_plot7(selected_years):
    selected_data = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]

    mean_temp_data = selected_data.groupby('Location').agg({'MinTemp': 'mean', 'MaxTemp': 'mean'}).reset_index()
    color_map = {'MinTemp': '#aed6dc', 'MaxTemp': '#f4898b'}
    fig = px.bar(mean_temp_data, x='Location', y=['MinTemp', 'MaxTemp'],
                 title="Mean Min and Max Temperature Comparison by Location",
                 labels={'value': 'Temperature (°C)', 'variable': 'Metric'},
                 width=1500, height=600, color_discrete_map=color_map)
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 20, 'color': 'darkred'}
    fig.update_layout(legend=dict(title='', font=dict(size=18)),
                      title_font=font_title,
                      title_x=0.5,
                      )
    fig.update_layout(barmode='group')
    fig.update_yaxes(title_font=font_labels)
    fig.update_xaxes(tickangle=45, title_font=font_labels)

    return fig


@my_app.callback(
    Output('plot8', 'figure'),
    Input('numerical-feature-dropdown', 'value'),
    Input('kde-tab', 'value')
)
def update_plot8(selected_feature, selected_tab):
    if selected_tab == 'kde_hist-tab':
        fig = ff.create_distplot(
            [df[selected_feature]],
            [selected_feature],
            show_hist=True,
            colors=['#f4898b'],
            curve_type='kde',
            show_rug=True,
            bin_size=1,
        )
        fig.update_layout(
            title=f'Histogram with KDE for {selected_feature}',
        )
        for trace in fig['data']:
            trace['opacity'] = 0.7
    else:
        fig = ff.create_distplot(
            [df[selected_feature]],
            [selected_feature],
            show_hist=False,
            colors=['#000080'],
            curve_type='kde',
            show_rug=True,
            bin_size=1,
        )
        fig.update_layout(
            title=f'KDE for {selected_feature}'
        )
        kde_curve = fig['data'][0]
        kde_filled_area = go.Scatter(
            x=kde_curve['x'],
            y=kde_curve['y'],
            fill='tozeroy',
            fillcolor='rgba(0, 0, 128, 0.6)',
            mode='none',
            line=dict(
                width=4,
                color='rgba(0, 0, 128, 1)'
            ),
        )

        fig.add_trace(kde_filled_area)

    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 20, 'color': 'darkred'}

    fig.update_layout(
        title_font=font_title,
        title_x=0.5,
        showlegend=False,
        yaxis_title='Density',
        yaxis_title_font=font_labels,
        xaxis_title=selected_feature,
        xaxis_title_font=font_labels,
    )

    return fig


@my_app.callback(
    Output('plot9', 'figure'),
    [Input('pie-chart-dropdown', 'value')]
)
def update_plot9(selected_values):
    rows = (len(selected_values) + 1) // 2

    fig = make_subplots(rows=rows, cols=2, specs=[[{'type': 'pie'}] * 2 for _ in range(rows)])

    for i, value in enumerate(selected_values):
        row = (i // 2) + 1
        col = (i % 2) + 1

        if value == 'rainfall':
            bins = [0, 0.1, 20, 50, 380]
            labels = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
            df['RainfallCategory'] = pd.cut(df['Rainfall'], bins=bins, labels=labels, include_lowest=True)
            rainfall_counts = df['RainfallCategory'].value_counts()
            fig.add_trace(
                go.Pie(labels=rainfall_counts.index, values=rainfall_counts.values, name='Rainfall',
                       title='Rainfall Distribution'),
                row=row, col=col)

        elif value == 'location':
            total_rainfall_by_location = df.groupby('Location')['Rainfall'].sum().reset_index()
            fig.add_trace(go.Pie(labels=total_rainfall_by_location['Location'],
                                 values=total_rainfall_by_location['Rainfall'],
                                 name='Total Rainfall',
                                 title='Total Rainfall by Location'), row=row, col=col)

        elif value == 'wind_gust_dir':
            wind_gust_dir_counts = df['WindGustDir'].value_counts()
            fig.add_trace(
                go.Pie(labels=wind_gust_dir_counts.index, values=wind_gust_dir_counts.values, name='WindGustDir',
                       title='WindGustDir'),
                row=row, col=col)

        elif value == 'wind_dir_9am':
            total_rainfall_by_wind_dir_9am = df.groupby('WindDir9am')['Rainfall'].sum().reset_index()
            fig.add_trace(go.Pie(labels=total_rainfall_by_wind_dir_9am['WindDir9am'],
                                 values=total_rainfall_by_wind_dir_9am['Rainfall'],
                                 name='Total Rainfall 9am',
                                 title='Total Rainfall at 9 AM by Wind Direction'),
                          row=row, col=col)

        elif value == 'wind_dir_3pm':
            total_rainfall_by_wind_dir_3pm = df.groupby('WindDir3pm')['Rainfall'].sum().reset_index()
            fig.add_trace(go.Pie(labels=total_rainfall_by_wind_dir_3pm['WindDir3pm'],
                                 values=total_rainfall_by_wind_dir_3pm['Rainfall'],
                                 name='Total Rainfall 3pm',
                                 title='Total Rainfall at 3 PM by Wind Direction'),
                          row=row, col=col)

        elif value == 'rain_today':
            rain_today_counts = df['RainToday'].value_counts()
            fig.add_trace(go.Pie(labels=rain_today_counts.index, values=rain_today_counts.values, name='RainToday',
                                 title='RainToday'),
                          row=row, col=col)

        elif value == 'rain_tomorrow':
            rain_tomorrow_counts = df['RainTomorrow'].value_counts()
            fig.add_trace(
                go.Pie(labels=rain_tomorrow_counts.index, values=rain_tomorrow_counts.values, name='RainTomorrow',
                       title='RainTomorrow'),
                row=row, col=col)

    fig.update_layout(height=400 * rows, width=1000)
    return fig


@my_app.callback(
    Output('qq-plot', 'figure'),
    Input('column-selector-qqplot', 'value')
)
def update_qq_plot(selected_column):
    selected_data = df[selected_column]
    sorted_data = np.sort(selected_data)

    percentiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    expected_quantiles = norm.ppf(percentiles)

    fig = px.scatter(x=expected_quantiles, y=sorted_data, title=f'Q-Q Plot for {selected_column}')

    fig.add_shape(
        type='line',
        x0=min(expected_quantiles),
        y0=min(sorted_data),
        x1=max(expected_quantiles),
        y1=max(sorted_data),
        line=dict(color='red', dash='dash')
    )

    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 20, 'color': 'darkred'}

    fig.update_layout(
        title_font=font_title,
        title_x=0.5,
        showlegend=False,
        yaxis_title=f'Sorted {selected_column}',
        yaxis_title_font=font_labels,
        xaxis_title='Normal theoretical quantile',
        xaxis_title_font=font_labels,
    )

    return fig


@my_app.callback(
    Output('violin-plot', 'figure'),
    Input('violin-tabs', 'value')
)
def update_violin(tab):
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 20, 'color': 'darkred'}
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if tab == 'tab-1':
        fig = px.violin(df, x='Location', y='WindGustSpeed', box=True,
                        title='Wind Gust Speed by Location', width=1500,
                        height=500)
        fig.update_layout(
            title_font=font_title,
            title_x=0.5,
            yaxis_title=f'WindGustSpeed',
            yaxis_title_font=font_labels,
            xaxis_title='Location',
            xaxis_title_font=font_labels,
        )
        return fig
    elif tab == 'tab-2':
        fig = px.violin(df, x='Month', y='WindGustSpeed', box=True, title='Wind Gust Speed by Month', width=1500,
                        height=500)
        fig.update_layout(
            title_font=font_title,
            title_x=0.5,
            yaxis_title_font=font_labels,
            xaxis_title_font=font_labels,
        )
        fig.update_xaxes(
            tickvals=list(range(1, 13)),
            ticktext=month_order
            , title_text='Month', title_font=font_labels)
        return fig
    elif tab == 'tab-3':
        fig = px.violin(df, x='Month', y='Humidity9am', color='RainToday', box=True,
                        title='Humidity at 9 am by Month (Separated by RainToday)', width=1500, height=500)
        fig.update_layout(
            title_font=font_title,
            title_x=0.5,
            yaxis_title_font=font_labels,
            xaxis_title_font=font_labels,
        )
        fig.update_xaxes(
            tickvals=list(range(1, 13)),
            ticktext=month_order,
            title_text='Month', title_font=font_labels
        )
        return fig
    elif tab == 'tab-4':
        fig = px.violin(df, x='Month', y='Humidity3pm', color='RainToday', box=True,
                        title='Humidity at 3pm  by Month (Separated by RainToday)', width=1500, height=500)
        fig.update_layout(
            title_font=font_title,
            title_x=0.5,
            yaxis_title_font=font_labels,
            xaxis_title_font=font_labels,
        )
        fig.update_xaxes(
            tickvals=list(range(1, 13)),
            ticktext=month_order,
            title_text='Month', title_font=font_labels
        )
        return fig


@my_app.callback(
    Output('boxen-plot', 'figure'),
    [Input('boxen-tabs', 'value')]
)
def update_boxen_plots(tab):
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 20, 'color': 'darkred'}
    long_df = df.melt(id_vars='Year', value_vars=['Pressure9am', 'Pressure3pm'], var_name='measurement',
                      value_name='value')
    val = 'Pressure'
    if tab == 'btab-1':
        long_df = df.melt(id_vars='Year', value_vars=['Pressure9am', 'Pressure3pm'], var_name='measurement',
                          value_name='value')
        val = 'Pressure'
    elif tab == 'btab-3':
        long_df = df.melt(id_vars='Year', value_vars=['Humidity9am', 'Humidity3pm'], var_name='measurement',
                          value_name='value')
        val = 'Humidity'
    elif tab == 'btab-2':
        long_df = df.melt(id_vars='Year', value_vars=['Temp9am', 'Temp3pm'], var_name='measurement',
                          value_name='value')
        val = 'Temperature'
    elif tab == 'btab-4':
        long_df = df.melt(id_vars='Year', value_vars=['WindSpeed9am', 'WindSpeed3pm'], var_name='measurement',
                          value_name='value')
        val = 'Wind Speed'
    elif tab == 'btab-5':
        df_temp = df.copy()
        df_temp['Year'] = df_temp['Year'].astype(str)
        fig = px.box(
            df_temp,
            y='Rainfall',
            x='Year',
            color='Year',
            labels={'Rainfall': 'Rainfall'},
            title='Rainfall Distribution by Year',
            width=800,
            height=500
        )
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Rainfall',
            boxmode='group',
            title_font=font_title,
            xaxis_title_font=font_labels,
            yaxis_title_font=font_labels,
            title_x=0.5
        )

        return fig

    fig = px.box(
        long_df,
        x='measurement',
        y='value',
        color='Year',
        labels={'value': val},
        title=f'{val} Readings by Year',
        width=800,
        height=500
    )

    fig.update_layout(
        legend_title_text='Year',
        title_font=font_title,
        xaxis_title_font=font_labels,
        yaxis_title_font=font_labels,
        title_x=0.5
    )

    return fig


@my_app.callback(
    Output('reg-plot', 'figure'),
    [Input('xaxis-dropdown-reg', 'value'),
     Input('yaxis-dropdown-reg', 'value')]
)
def update_reg_plot(xaxis_name, yaxis_name):
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 20, 'color': 'darkred'}
    fig = px.scatter(df, x=xaxis_name, y=yaxis_name, trendline="ols", title=f'Im plot {xaxis_name} vs {yaxis_name}')

    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))

    fig.update_traces(selector=dict(type='scattergl', mode='lines'),
                      line=dict(color='red', width=4))
    fig.update_layout(
        title_font=font_title,
        xaxis_title_font=font_labels,
        yaxis_title_font=font_labels,
        title_x=0.5
    )

    return fig


@my_app.callback(
    Output('strip-plot', 'figure'),
    Input('strip-tabs', 'value')
)
def update_strip_plot(selected_column):
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 20, 'color': 'darkred'}
    fig = px.strip(df, x=selected_column, y='Humidity3pm')
    fig.update_layout(
        title_font=font_title,
        xaxis_title_font=font_labels,
        yaxis_title_font=font_labels,
        title_x=0.5
    )
    fig.update_traces(marker=dict(size=5,
                                  line=dict(width=1,
                                            color='Darkblue')))
    return fig


@my_app.callback(
    Output('area-plot', 'figure'),
    [Input('area-tabs', 'value')]
)
def update_area_plot(input_value):
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 20, 'color': 'darkred'}
    if input_value == 'MaxMinTemp':
        df_yearly = df_og[['MaxTemp', 'MinTemp', 'Date']].resample('Y', on='Date').mean()

        fig = px.area(df_yearly, x=df_yearly.index, y=['MaxTemp', 'MinTemp'],
                      labels={'value': 'Average Temperature (°C)', 'Date': 'Year'})

        fig.update_layout(title='Yearly Mean Max and Min Temperatures')
    elif input_value == 'Rainfall':
        df_yearly = df_og[['Rainfall', 'Date']].resample('Y',
                                                         on='Date').sum()

        fig = px.area(df_yearly, x=df_yearly.index, y='Rainfall',
                      labels={'Rainfall': 'Total Rainfall (mm)', 'Date': 'Year'})

        fig.update_layout(title='Yearly Total Rainfall')
    elif input_value == 'WindGustSpeed':
        df_yearly = df_og[['WindGustSpeed', 'Date']].resample('Y',
                                                              on='Date').sum()

        fig = px.area(df_yearly, x=df_yearly.index, y='WindGustSpeed',
                      labels={'WindGustSpeed': 'Average WindGustSpeed', 'Date': 'Year'})

        fig.update_layout(title='Yearly average WindGustSpeed')
    elif input_value == 'Sunshine':
        df_yearly = df_og[['Sunshine', 'Date']].resample('Y', on='Date').mean()

        fig = px.area(df_yearly, x=df_yearly.index, y='Sunshine',
                      labels={'Sunshine': 'Average Sunshine', 'Date': 'Year'})

        fig.update_layout(title='Yearly average Sunshine')
    elif input_value == 'Evaporation':
        df_yearly = df_og[['Evaporation', 'Date']].resample('Y', on='Date').mean()

        fig = px.area(df_yearly, x=df_yearly.index, y='Evaporation',
                      labels={'Evaporation': 'Average Evaporation', 'Date': 'Year'})

        fig.update_layout(title='Yearly average Evaporation')

    fig.update_layout(
        title_font=font_title,
        xaxis_title_font=font_labels,
        yaxis_title_font=font_labels,
        title_x=0.5
    )
    return fig


@my_app.callback(
    Output('hex-plot', 'figure'),
    [Input('hex-in1', 'value'),
     Input('hex-in2', 'value')]
)
def update_hexbin_plot(input1, input2):
    font_title = {'family': 'serif', 'size': 30, 'color': 'blue'}
    font_labels = {'family': 'serif', 'size': 20, 'color': 'darkred'}
    fig = px.density_heatmap(df, x=input1, y=input2, nbinsx=40, nbinsy=40)

    fig.update_layout(
        title=f'Hexbin-like Plot of {input1} vs {input2}',
        xaxis_title=input1,
        yaxis_title=input2,
        title_font=font_title,
        xaxis_title_font=font_labels,
        yaxis_title_font=font_labels,
        title_x=0.5
    )
    return fig


@my_app.callback(
    Output('joint-plot', 'src'),
    [Input('joint-tabs', 'value')]
)
def render_image(tab):
    if tab == 'tab-1':
        return 'assets/jointplot1.png'
    elif tab == 'tab-2':
        return 'assets/jointplot2.png'
    elif tab == 'tab-3':
        return 'assets/jointplot3.png'
    elif tab == 'tab-4':
        return 'assets/jointplot4.png'
    elif tab == 'tab-5':
        return 'assets/jointplot5.png'
    elif tab == 'tab-6':
        return 'assets/jointplot6.png'


if not os.path.exists('assets'):
    os.mkdir('assets')


if __name__ == '__main__':
    my_app.run_server(debug=True, host='0.0.0.0', port=8080)