#!/usr/bin/env python
# coding: utf-8

# # Red Bull Data Science Case Study

# ### Loading libraries

# In[213]:


import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from datetime import timedelta

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from numpy import asarray
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot

import ipywidgets as widgets
from IPython.display import display
import networkx as nx


# ### Importing dataset

# In[210]:


df = pd.read_excel("C:/Users/paris/OneDrive/Desktop/RedBull_Assignment/purchase_orders_case_study.xlsx")


# ### Exploratory Data Analysis (EDA)

# In[186]:


df.head()


# In[187]:


df.shape


# In[188]:


df.info()


# In[42]:


# Check for missing values
MVs = df.isnull().sum()
MVs = MVs[MVs > 0]

if not MVs.empty:
    print('Columns with missing values:')
    for column, count in MVs.items():
        print(f'{column}: {count} missing values')


# In[43]:


# Check for duplicate data
if df.duplicated().any():
    print(f"There are as many as {df.duplicated.sum()} duplicated data.")
else:
    print('There are no duplicate data.')       


# In[211]:


# Casting
df['Product_ID'] = df['Product_ID'].astype(str)
df['order_date'] = pd.to_datetime(df['order_date'])


# In[45]:


# The distinct number of some variables
columns = ['Vendor','Vendor_City','Subsidiary','Product_ID', 'Purchase_Order_No']
for column in columns:
    distinct_values = df[column].nunique()
    print(f' The number of distinct values for {column}: {distinct_values}')


# In[205]:


# Bar charts
columns = ['Vendor','Vendor_City','Subsidiary','Product_ID']
for column in columns:
    column_counts = df[column].value_counts().reset_index()
    column_counts.columns = [column, 'Count']
    column_counts = column_counts.sort_values(by='Count', ascending=False)
    fig = px.bar(column_counts, x = column, y = 'Count', title = f"{column} count")
       
fig.show()


# In[11]:


counts = df.groupby(['Vendor', 'Subsidiary']).size().reset_index(name='count')
counts = counts.sort_values(by = 'count', ascending=False)

fig = go.Figure()

for vendor, group in counts.groupby('Vendor'):
    fig.add_trace(go.Bar(
        x = group['Vendor'] + ' - ' + group['Subsidiary'],
        y = group['count'],
        name = vendor
    ))

fig.update_layout(
    title = 'Vendor and Subsidiary',
    xaxis_title = 'Vendor - Subsidiary',
    yaxis_title = 'Count',
    barmode = 'group',
    width = 1000,
    height = 600
)

fig.show()


# In[12]:


grouped_df = df[['Vendor', 'Vendor_City']].groupby('Vendor')
grouped_df['Vendor_City'].unique()


# In[95]:


# Number of orders placed by each Subsidiary
df_subsidiary = df[['Product_ID', 'Subsidiary']].groupby('Subsidiary').count().reset_index()  
df_subsidiary.rename(columns = {'Product_ID':'Num_Order_Placed'}, inplace = True)
df_subsidiary


# In[47]:


# Lead time
df['Lead_Time']= df['Goods_Receipt_Date']-df['order_date']
df = df.dropna(subset=['Lead_Time'])
df['Lead_Time'] = df['Lead_Time'].astype(str)
df['Lead_Time'] = df['Lead_Time'].str.extract('(\d+)', expand=False).astype(int)


# ### Visualization of product flow through sypply network

# In[190]:


# Defining a function to calculate the coordinates of each city
def coordinates_lat_long(city_name):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(city_name)
    if location:
        latitude = location.latitude
        longitude = location.longitude
        return latitude, longitude
    else:
        return None, None
    
df = df.dropna(subset=['Vendor_City'])

Vendor_City = df['Vendor_City'].unique()
Subsidiary = df['Subsidiary'].unique()

geoinf_Vendor_City = []

for i in Vendor_City:
    latitude, longitude = coordinates_lat_long(i)
    geoinf_Vendor_City.append([i,latitude, longitude])
    
geoinf_subsidiaries = []   

for j in Subsidiary:
    latitude, longitude = coordinates_lat_long(j)
    geoinf_subsidiaries.append([j,latitude, longitude])


# In[191]:


# Add coordinates to the dataframe
VendorCity_df = pd.DataFrame(geoinf_Vendor_City, columns=['Vendor_City', 'Vendor_Latitude', 'Vendor_Longitude'])
Subsidiary_df = pd.DataFrame(geoinf_subsidiaries, columns=['Subsidiary', 'Subsidiary_Latitude', 'Subsidiary_Longitude'])
                             
merged_df = pd.merge(df, VendorCity_df, on='Vendor_City', how='left')
merged_df = pd.merge(merged_df, Subsidiary_df, on='Subsidiary', how='left')             

merged_df.head()


# In[193]:


# Visualize Product-flow Network
df_Net = merged_df[['Product_ID', 'Vendor', 'Vendor_City', 'Subsidiary', 'Vendor_Latitude', 'Vendor_Longitude', 'Subsidiary_Latitude', 'Subsidiary_Longitude']].drop_duplicates()

product_dropdown = widgets.Dropdown(
    options=df_Net['Product_ID'].unique(),
    description='Product ID:',
    disabled=False,
)

fig = go.FigureWidget()
display(fig)

# Define a function to generate the network graph
def generate_network_graph(product_id):
    filtered_df = df_Net[df_Net['Product_ID'] == product_id]

    G = nx.DiGraph()

    edges = [(row['Vendor_City'], row['Subsidiary']) for _, row in filtered_df.iterrows()]
    G.add_edges_from(edges)
    
    pos = {}
    for index, row in filtered_df.iterrows():
        vendor_city = row['Vendor_City']
        vendor_latitude = row['Vendor_Latitude']
        vendor_longitude = row['Vendor_Longitude']

        subsidiary_city = row['Subsidiary']
        subsidiary_latitude = row['Subsidiary_Latitude']
        subsidiary_longitude = row['Subsidiary_Longitude']

        pos[vendor_city] = (vendor_longitude, vendor_latitude)
        pos[subsidiary_city] = (subsidiary_longitude, subsidiary_latitude)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_colors = []
    node_symbols = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node in filtered_df['Vendor_City'].unique():
            node_colors.append('red')
            node_symbols.append('circle-dot')
        else:
            node_colors.append('green')
            node_symbols.append('triangle-up')
        
    node_labels = list(G.nodes())
    fig.data = []
    fig.add_trace(go.Scattergeo(
        lon=edge_x,
        lat=edge_y,
        mode='lines',
        line=dict(width=1, color='blue'),
        hoverinfo='none'
    ))

    fig.add_trace(go.Scattergeo(
        lon=node_x,
        lat=node_y,
        mode='markers',
        marker=dict(
            symbol=node_symbols,
            size=10,
            color=node_colors,
            line=dict(width=0.5, color='rgb(50,50,50)')
        ),
        text=node_labels,
        hoverinfo='text'
    ))

    fig.update_layout(title=f'Product Flow Network for Product_ID: {product_id}',
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

def dropdown_change_handler(change):
    if change['type'] == 'change' and change['name'] == 'value':
        product_id = change['new']
        generate_network_graph(product_id)

product_dropdown.observe(dropdown_change_handler)
display(product_dropdown)


# ### Key Performance Indicators- KPIs

# ####  KPI 1: Vendor Diversity
# - This KPI shows us the degree of diversity of vendors for each subsidiary in this supply chain.
# - Diversifying the supplier base helps mitigate risks associated with dependence on a single supplier or group of suppliers.

# In[46]:


df_grouped_3 = df[['Product_ID', 'Vendor', 'Subsidiary']].groupby(['Subsidiary', 'Product_ID'])['Vendor'].nunique().reset_index()
df_grouped_3.rename(columns = {'Vendor': 'num_Vendor_Orderd'}, inplace = True) 
df_grouped_2 = df[['Product_ID', 'Vendor']].groupby('Product_ID')['Vendor'].nunique().reset_index()
df_grouped_2.rename(columns = {'Vendor': 'num_Vendor'}, inplace = True) 
df_merged = pd.merge(df_grouped_2, df_grouped_3, on = 'Product_ID', how = 'inner')
data = []

for value in list(df['Subsidiary'].unique()):
    df_filtered = df_merged[df_merged['Subsidiary'] == value]
    sum1 = df_filtered['num_Vendor'].sum()
    sum2 = df_filtered['num_Vendor_Orderd'].sum()
    Vendor_Diversity = round((sum2/sum1)*100, 2)
    data.append({'Subsidiary': value, 'Vendor_Diversity': Vendor_Diversity})

result_df = pd.DataFrame(data)
result_df

# Share_of_Total
df_2_grouped = df[['Subsidiary', 'Product_ID']].groupby('Subsidiary')['Product_ID'].size().reset_index()
df_2_grouped.rename(columns = {'Product_ID': 'Total_num_Product_ID'}, inplace = True)

product_count_per_vendor = df.groupby(['Vendor', 'Product_ID']).size().reset_index(name='Count')
total_products_delivered = product_count_per_vendor['Count'].sum()

df_2_grouped ['Share_of_Total'] = round((df_2_grouped['Total_num_Product_ID']/total_products_delivered)*100, 2)

# Excluded_Vendors
df_New = pd.merge(df_2_grouped, result_df, on = 'Subsidiary', how = 'inner' )
df_New['Excluded_Vendors'] = 100 - df_New['Vendor_Diversity']

df_New


# In[49]:


for sub in df_New['Subsidiary'].unique():
    subset_df = df_New[df_New['Subsidiary'] == sub]
    
    # Create figure with two subplots, pie chart for Vendor Diversity and Excluded Vendors
    labels = ['Vendor Diversity', 'Excluded Vendors']
    values = [subset_df['Vendor_Diversity'].iloc[0], subset_df['Excluded_Vendors'].iloc[0]]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title_text=f'Vendor Diversity for {sub}', title_x=0.5, height=400, width=500)
    fig.show()


# #### KPI 2: Quantity order distribution
# - Distribution of the quantities of products ordered over time shows us the frequency, variability, order patterns of  quantities.
# - Order distribution in this supply chain shows a bimodal pattern.

# In[55]:


grouped = df.groupby(['Product_ID', 'Quantity']).size().reset_index(name='Frequency')
product_ids = grouped['Product_ID'].unique()

fig = go.Figure()

for product_id, group_data in grouped.groupby('Product_ID'):
    fig.add_trace(go.Bar(x=group_data['Quantity'], y=group_data['Frequency'], name=product_id, visible=False))

fig.data[0].visible = True

fig.update_layout(
    title='Frequency of Quantity for Each Product ID',
    xaxis=dict(title='Quantity'),
    yaxis=dict(title='Frequency'),
    barmode='group',
    legend=dict(title='Product ID'))

buttons = []
for i, product_id in enumerate(product_ids):
    button = dict(
        label=product_id,
        method="update",
        args=[{"visible": [False] * len(product_ids)},
              {"title": f"Frequency of Quantity for Product ID: {product_id}"}],
    )
    button["args"][0]["visible"][i] = True
    buttons.append(button)

fig.update_layout(
    updatemenus=[dict(buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.5, xanchor="left", y=1.15, yanchor="top")])

fig.show()


# #### KPI 3:  Vendor Lead Time Coefficient of Variation 
# - This KPI refers to the degree of variation in the time it takes for a vendor to fulfill an order.  

# In[81]:

average_leadtime_vendor = df.groupby('Vendor')['Lead_Time'].mean().reset_index()
average_leadtime_vendor = average_leadtime_vendor.rename(columns = {'Lead_Time' : 'Mean Lead Time'})

variability_leadtime_vendor = df.groupby('Vendor')['Lead_Time'].std().reset_index()
variability_leadtime_vendor = variability_leadtime_vendor.rename(columns = {'Lead_Time' : 'Variability Lead Time'})

vendor_product = df.groupby('Vendor')['Product_ID'].nunique().reset_index()
vendor_product = vendor_product.rename(columns = {'Product_ID' : 'Num unique Products'})

df_vendor = pd.merge(vendor_product, pd.merge(average_leadtime_vendor, variability_leadtime_vendor,  on = 'Vendor', how='inner'), on = 'Vendor', how='inner')
df_vendor['CV Lead Time %'] = round(df_vendor['Variability Lead Time']/df_vendor['Mean Lead Time']*100, 2)
df_vendor


# #### KPI 4: Quantity Order Position Index
# - This KPI for each product in a supply chain is a measure of relative distribution within the range defined by the minimum and maximum order quantities.

# In[212]:


df_stat_product = df.groupby('Product_ID').agg(
    min_Quantity=('Quantity', 'min'),
    max_Quantity=('Quantity', 'max'),
    median_Quantity=('Quantity', 'median'),
    mean_Quantity=('Quantity', 'mean'),
    std_Quantity=('Quantity', 'std')
).reset_index()

df_stat_product[['mean_Quantity', 'std_Quantity']] = df_stat_product[['mean_Quantity', 'std_Quantity']].round(2)
df_stat_product ['Quantity Order Position Index'] = abs(df_stat_product['median_Quantity'] - df_stat_product['max_Quantity'])/abs(df_stat_product['median_Quantity'] - df_stat_product['min_Quantity'])

fig = px.bar(df_stat_product, x='Product_ID', y='Quantity Order Position Index',
             title='Quantity Order Position Index for Each Product')

fig.show()


# In[89]:


fig = px.box(df_stat_product, 
             x='Product_ID', 
             y=['min_Quantity', 'max_Quantity', 'median_Quantity', 'mean_Quantity', 'std_Quantity'],
             title='Box Plot of Quantity for Each Product')

fig.show()


# ### Modeling

# #### XG BOOST
# Steps:
# - Transforming a time series dataset into a supervised learning dataset.
# - Implementing walk-forward validation for univariate data.
# - Spliting dataset into train/test sets.
# - fitting an xgboost model and making a one step prediction.
# - Evaluating the model performance.

# In[108]:


# Transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    
    for i in range(n_in, 0, -1):     # input sequence (t-n, ... t-1)
        cols.append(df.shift(i))
        
    for i in range(0, n_out):        # forecast sequence (t, t+1, ... t+n)
        cols.append(df.shift(-i))

        agg = concat(cols, axis=1)
        
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# In[183]:


# Implement walk-forward validation for data
def walk_forward_validation(data, n_test):
    
    predictions = list()
    train, test = train_test_splitt(data, n_test)
    history = [x for x in train]
    
    for i in range(len(test)):

        testX, testy = test[i, :-1], test[i, -1]
        yhat = xgboost_forecast(history, testX)
        predictions.append(yhat)
        history.append(test[i])
        
  #      print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, 1], predictions


# In[147]:


# Splitting the dataset into train/test sets
def train_test_splitt(data, n_test):
    
    return data[:-n_test, :], data[-n_test:, :]


# In[99]:


# fitting an xgboost model and making a one step prediction
def xgboost_forecast(train, testX):
    
    train = asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    
    yhat = model.predict([testX])
    return yhat[0]


# In[184]:



unique_product_ids = df['Product_ID'].unique()
product_ids = unique_product_ids[unique_product_ids != 12]

for i in product_ids:
    df_filtered = []
    df_filtered = df[df['Product_ID'] == i][['order_date','Quantity']].sort_values(by = 'order_date')
    df_filtered = df_filtered.dropna()
    data = df_filtered
    data = list(data['Quantity'])
    
    series = pd.Series(data) 
    values = list(series.values)

    data = series_to_supervised(values, n_in=1, n_out=1, dropnan=True)
    
    n_test = round(len(data)*0.2)
    train, test = train_test_splitt(data, n_test)
    
    mae, y, yhat = walk_forward_validation(data, n_test)
    print('Product_ID:', i)
    print('MAE: %.3f' % mae)
    
    pyplot.plot(y, label='Expected')
    pyplot.plot(yhat, label='Predicted')
    pyplot.legend()
    pyplot.show()

