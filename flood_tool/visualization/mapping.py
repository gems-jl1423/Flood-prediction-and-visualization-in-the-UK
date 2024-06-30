import sys
sys.path.append('..')

import os
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .. import geo
from folium import FeatureGroup
from pykrige.uk import UniversalKriging
from folium.plugins import HeatMap
import branca
import scipy.interpolate as interp
import scipy.ndimage as ndimage
import geojsoncontour
from folium import LayerControl
import base64
import branca.colormap as cm
from ..tool import _data_dir
present_path = os.path.abspath(__file__)
parent_folder = os.path.dirname(present_path)


# List of functions to be imported when using "from flood_tool.visualization import *"
all = ["display_features", "UniversalK_display", "add_countours"]



def display_features(path = f'{_data_dir}/postcodes_labelled.csv', path_selected=True, data_selected=False, 
                     data=None, parameter='riskLabel'):

    """Function to display the risk label of each postcode on the map
    Parameters
    ----------
    path : str
        The path of the postcode file
    path_selected : bool
        If True, the path is selected
    data_selected : bool
        If True, the dataframe is selected
    data : pandas.DataFrame
        The dataframe of the data
    parameter : str
        The column name of the target variable, by default 'riskLabel'

    Returns
    -------
    m : folium.Map

    """
    #Defining global variable m 
    global m 

    # Read the postcode file
    if path_selected == True:
        df = pd.read_csv(path)
    
    # Read the dataframe
    elif data_selected == True:
        df = data
    
    # Raise error if the path or the dataframe is not selected
    elif path_selected == False and data_selected == False:
        raise ValueError("Please specify the path of the  file or the dataframe itself")
    
    # Create a new dataframe to store the district and street of each postcode
    df_post = df.copy()
    district = df['postcode'].apply(lambda x: x.split(' ')[0]).to_list()
    street = df['postcode'].apply(lambda x: x.split(' ')[1]).to_list()
    df_post['district'] = district
    df_post['street'] = street
    dis_num = np.unique(district)

    # Create a new dataframe to store the easting and northing of each district
    df_new_postlabel = pd.DataFrame(data = np.array([dis_num,np.zeros(len(dis_num)),np.zeros(len(dis_num))]).T,columns = ['postcode','X','Y'])

    # Calculate the mean easting and northing of each district
    for word in dis_num:
        x = np.mean(df_post[df_post['district']==word]['easting'])
        y = np.mean(df_post[df_post['district']==word]['northing']) 
        df_new_postlabel.loc[df_new_postlabel['postcode'] == word,'X'] = x
        df_new_postlabel.loc[df_new_postlabel['postcode'] == word,'Y']= y
    y,x = geo.get_gps_lat_long_from_easting_northing(df_new_postlabel['X'],df_new_postlabel['Y'])

    # Create a map
    
    m = folium.Map(location=[np.mean(y), np.mean(x)], zoom_start=7)

    # Create a feature group to store the postcode of each district
    feature_group1 = FeatureGroup(name='Postcode of district', show=False)
    # Create a feature group to store the risk label of each district
    feature_group2 = FeatureGroup(name='Risk Labels Postcode',show = False)
    # Create a feature group to store the risk area of each district
    feature_group3 = FeatureGroup(name='Risk Area', show=False)
    # Create a feature group to store the contour line of selected parameters
    feature_group4 = FeatureGroup(name = f'{parameter} contour',show=False)

    # Loop over each district to add the postcode to the map
    for i in range(df_new_postlabel.shape[0]):
        folium.Circle(location=(y[i], x[i]),
                  popup=f'{df_new_postlabel.iloc[i,0]}'
                  ).add_to(feature_group1)
    
    colormap = cm.StepColormap(colors=['green','yellow','orange','red'] ,
                                       index=[1,3,5,7,10], vmin= 1,vmax=10)
    
    # Get latitude and longitude from easting and northing
    df.loc[:,'northing'], df.loc[:,'easting'] = geo.get_gps_lat_long_from_easting_northing(df["easting"],df["northing"])

    # Loop over each postcode to add the risk label to the map
    for loc, risk, postcode in zip(zip(df["northing"],df["easting"]), df["riskLabel"], df["postcode"]):
        folium.Circle(
        location=loc,
        radius=2,      
        fillOpacity=0.8,
        color=colormap(risk)
        ).add_child(folium.Popup(postcode + "\n" + " Risk level:" + str(risk))).add_to(feature_group2)
    #Add legend to the map
    colormap.caption = 'Risk level'
    colormap.add_to(m)

    risk_area(feature_group3, df_new_postlabel)
    
    # Add the feature groups to the map
    feature_group1.add_to(m)
    feature_group2.add_to(m)
    feature_group3.add_to(m)
    
    # Add the contour line of selected parameters to the map
    if parameter == None:
        add_countours(feature_group4, param='riskLabel')
    else:
        add_countours(feature_group4, param=parameter)
    feature_group4.add_to(m)

    # Control the layer display
    LayerControl().add_to(m)

    return m


def risk_area(input_map,data):
    """Function to display the risk area of each district on the map"""

    def encode_image(image_path):
        """Function to encode the image to base64 format"""
        with open(image_path, "rb") as image_file:
            
            return base64.b64encode(image_file.read()).decode()
    
    # Districts with risk area
    risk = ['NE6','TS14','BD3','OL6','S13','LE9','NN8','NR18']
    data_risk = pd.DataFrame(data = np.array(data.iloc[:,1:]),index = data['postcode'].to_numpy(),columns = ['X','Y'])
    data_risk = data_risk.loc[risk]
    
    # Loop to add the risk area to the map
    for name in risk:
        y,x = geo.get_gps_lat_long_from_easting_northing(data_risk.loc[name,'X'],data_risk.loc[name,'Y'])
        # encode the image to base64 format
        encoded_image = encode_image(f"{parent_folder}/popup image/heat_map_{name}.png")
        html = f'<img src="data:image/png;base64,{encoded_image}"style="width:400; height:400;">'
        # Create a popup
        iframe = branca.element.IFrame(html=html,width=400, height=400)
        popup = folium.Popup(iframe, max_width=400)
        folium.Marker(location = (y,x),popup = popup).add_to(input_map)
 

def UniversalK_display(data, x_target = 'easting',y_target = 'northing',phi_target = 'risk'):
    """"Function to display the universal kriging of the data
    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe of the data
    x_target : str
        The column name of the easting
    y_target : str
        The column name of the northing
    phi_target : str
        The column name of the target variable, by default 'risk'
    Returns
    -------
    m : folium.Map
    """
    # Get the minimum number of the target variable
    _,min_num = np.unique(data[phi_target],return_counts = True)
    min_num = np.sum(min_num[1:])
    # Get the data with the minimum number of the target variable
    data_smaller = pd.concat((data[data[phi_target] == 1].sample(min_num), data[data[phi_target] !=1]))
    # Get the latitude and longitude from easting and northing
    y,x = geo.get_gps_lat_long_from_easting_northing(data_smaller[x_target],data_smaller[y_target])
    
    # Universal kriging
    phi = data_smaller[phi_target]
    UK = UniversalKriging(
        x, 
        y, 
        phi, 
        variogram_model='exponential',
        verbose=True,
        enable_plotting=True,
        nlags=1000,
        drift_terms=['constant', 'linear', 'quadratic']
        )
        
    # Create a map
    gridx = np.arange(min(x), max(x), 0.01, dtype='float64')
    gridy = np.arange(min(y),max(y),0.01, dtype='float64')
    # Get the universal kriging result
    ustar, uss = UK.execute("grid", gridx, gridy)
    m = folium.Map(location=[np.mean(y), np.mean(x)], zoom_start=7) 
    heat_data = [[gridy[i], gridx[j], ustar.data[i,j]] for i in range(len(gridy)) for j in range(len(gridx)) if not np.isnan(ustar.data[i,j])]
    
    # Add the universal kriging result to the map
    HeatMap(heat_data,min_opacity=0.1,max_opacity=0.2).add_to(m)

    return m

def get_coor_stations(data, station=f'{_data_dir}/stations.csv'):
 
    """"Function to get the coordinates of the stations based on the station reference
    Parameters
    ----------
    """
    #if station == None: 
    #    station = self._stations.copy()
    #create a column of latitude in dataframe and fill it with latitudes from station based on stationReference 
    
    station = pd.read_csv(station)
    #Add a new column to store the latitude and longitude of each station
    data['latitude'] = data['stationReference'].map(station.set_index('stationReference')['latitude'])
    data['longitude'] = data['stationReference'].map(station.set_index('stationReference')['longitude'])
    
    return data

def add_countours(input_map=None, param='riskLabel', geographic=False,
                  risk_or_rainfall="risk",
                  path_postcode = f'{_data_dir}/postcodes_labelled.csv',
                  rainfall_river_data=f'{_data_dir}/typical_day.csv',
                  coord_EastNorth=["easting", "northing"], 
                  coord_LatLong=["latitude", "longitude"],
                  ):
    """
    Add risk countours on a base map (creating a new folium map instance if necessary).
    The default parameter to be plotted is riskLabel(1-10).
    
    If geographic is False, convert easting and northing to latitude and longitude, otherwise use latitude and longitude.
    Using path to specify the path of the postcode file. By default, it is the postcode file in the data folder.

    Using rainfall_river_data to specify the path of the rainfall or river data file. 
    By default, it is the typical_day.csv file in the data folder.
    
    User should specify the risk_or_rainfall parameter to specify the data type to be plotted. 
    By default, it is risk, from the postcode file. If user want to plot rainfall or river data, set risk_or_rainfall to rainfall_river.

    Parameters:
    -----------
    input_map: folium.Map
    param: str
    path_postcode: str
    rainfall_river_data: str
    coord_EastNorth: list
    coord_LatLong: list
    
    Returns:
    --------
    Folium map object

    """
    # Create a new folium map instance if necessary

    if risk_or_rainfall == "risk":
        
        # Read the postcode file
        data = pd.read_csv(path_postcode)
        # Drop rows with missing values
        data = data.dropna()

        # If geographic is False, convert easting and northing to latitude and longitude
        if geographic is False:
            geo_coord = geo.get_gps_lat_long_from_easting_northing(data[coord_EastNorth[0]].to_list(),
                                                                data[coord_EastNorth[1]].to_list())

            data['latitude'] = geo_coord[0]
            data['longitude'] = geo_coord[1]
    
        # Create a colormap for risk level
        if param == "riskLabel":

            # Create a colormap for risk level    
            colors = ['springgreen','yellowgreen','yellow','orange','red','purple']
            vmin, vmax = data['riskLabel'].min(), data['riskLabel'].max()
            colormap = cm.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(10)
            
            # Count the number of risk level 1 and other risk levels
            _,inbClases = np.unique(data['riskLabel'],return_counts=True)

            # Find the number of risk level 1
            sum_inbClases = sum(inbClases[1:])

            # Create a smaller dataframe with equal number of risk level 1 and other risk levels
            df2_smaller = pd.concat((data[data['riskLabel'] == 1].sample(sum_inbClases), data[data['riskLabel'] !=1]))

            # Define x, y, z coordinates for countour plot
            x_coord = df2_smaller[coord_LatLong[1]].to_numpy()
            y_coord = df2_smaller[coord_LatLong[0]].to_numpy()
            z_coord = df2_smaller["riskLabel"].to_numpy()

            # Create the grid
            x_grid, y_grid = np.meshgrid(np.linspace(x_coord.min(), x_coord.max(), 3000), np.linspace(y_coord.min(), y_coord.max(), 3000))

            # Interpolate the data for each grid point
            z_grid = interp.griddata((x_coord, y_coord), z_coord, (x_grid, y_grid), method='linear')

            # Smooth the data
            z_grid = ndimage.gaussian_filter(z_grid, sigma=20, order=0, mode='wrap')

            # Create contour plot to map
            contours = plt.contourf(x_grid, y_grid, z_grid, levels=len(colors), alpha=0.7, colors=colors, vmin=vmin, vmax=vmax)
            plt.close()

        else:

            #Create a coloprmap for rainfall, elevation or tide level data

            colormap = cm.linear.PuRd_09.scale(data[param].min(), data[param].max()).to_step(10)
            vmin, vmax = data[param].min(), data[param].max()

            x_coord = data[coord_LatLong[1]].to_numpy()
            y_coord = data[coord_LatLong[0]].to_numpy()
            z_coord = data[param].to_numpy()

            # Create the grid
            x_grid, y_grid = np.meshgrid(np.linspace(x_coord.min(), x_coord.max(), 3000), np.linspace(y_coord.min(), y_coord.max(), 3000))

            # Interpolate the data for each grid point
            z_grid = interp.griddata((x_coord, y_coord), z_coord, (x_grid, y_grid), method='linear')

            # Smooth the data
            z_grid = ndimage.gaussian_filter(z_grid, sigma=20, order=0, mode='wrap')

            # Create contour plot to map
            contours = plt.contourf(x_grid, y_grid, z_grid, levels=len(colormap.index), colors=colormap.colors, vmin=vmin, vmax=vmax)
            plt.close()

        # Converting matplotlib contourf to geojson so that it can be plotted on folium map
        geojson = geojsoncontour.contourf_to_geojson(contourf=contours, min_angle_deg=3, fill_opacity=0.85)

        # Plot the contour plot on folium
        folium.GeoJson(
            geojson,
            style_function=lambda x: {
                'color':     x['properties']['stroke'],
                'weight':    x['properties']['stroke-width'],
                'fillColor': x['properties']['fill'],
                'opacity':   0.7,
            }).add_to(input_map)
        
        #Add legend
        colormap.caption = f'{param}'
        colormap.add_to(m)
    
    elif risk_or_rainfall == 'rainfall_river':
        
        if input_map is None:
            
            input_map = folium.Map(location=[53.368998, -1.602912], zoom_start=7)
        
            rainfall_river_data = pd.read_csv(rainfall_river_data)
        
            # Get the latitude and longitude by using the station reference
            rainfall_river_data_coord = get_coor_stations(rainfall_river_data, station=f'{_data_dir}/stations.csv')

            rainfall_river_data_coord = rainfall_river_data_coord.loc[rainfall_river_data_coord['parameter'] == param]
            
            # Replace every row containing string '|' with NaN in column 'value'        
            rainfall_river_data_coord['value'] = rainfall_river_data_coord['value'].apply(lambda x: np.nan if '|' in str(x) else x)

            # Drop rows with missing values
            rainfall_river_data_coord = rainfall_river_data_coord.dropna()
            
            # Convert the data type of column 'value' from string to float
            rainfall_river_data_coord['value'] = rainfall_river_data_coord['value'].astype(float)
            rainfall_river_data_coord = rainfall_river_data_coord[rainfall_river_data_coord['value']>=0]

            #Define colormap for rainfall, elevation or tide level data
            colormap = cm.linear.PuRd_09.scale(rainfall_river_data_coord['value'].min(), 
                                               rainfall_river_data_coord['value'].max()).to_step(10)

            #Define x, y, z coordinates for countour plot
            vmin, vmax = rainfall_river_data_coord['value'].min(), rainfall_river_data_coord['value'].max()

            x_coord = rainfall_river_data_coord['longitude'].to_numpy()
            y_coord = rainfall_river_data_coord['latitude'].to_numpy()
            z_coord = rainfall_river_data_coord['value'].to_numpy()

            # Remove NaN values from the coordinates and values
            valid_indices = ~np.isnan(x_coord) & ~np.isnan(y_coord) & ~np.isnan(z_coord)
            x_coord = x_coord[valid_indices]
            y_coord = y_coord[valid_indices]
            z_coord = z_coord[valid_indices]

            # Create the grid
            x_grid, y_grid = np.meshgrid(np.linspace(x_coord.min(), x_coord.max(), 2000), np.linspace(y_coord.min(), y_coord.max(), 2000))

            # # Interpolate the data for each grid point
            z_grid = interp.griddata((x_coord, y_coord), z_coord, (x_grid, y_grid), 
                                      fill_value=np.nanmean(z_coord), method='linear')

            # # Smooth the data
            z_grid = ndimage.gaussian_filter(z_grid, sigma=20, order=0, mode='wrap')

            # # Create contour plot to map
            contours = plt.contourf(x_grid, y_grid, z_grid, levels=len(colormap.index), colors=colormap.colors, vmin=vmin, vmax=vmax)
            plt.close()

            # # Converting matplotlib contourf to geojson so that it can be plotted on folium map
            geojson = geojsoncontour.contourf_to_geojson(contourf=contours, min_angle_deg=3, fill_opacity=0.85)

            # Plot the contour plot on folium
            folium.GeoJson(
                geojson,
                style_function=lambda x: {
                    'color':     x['properties']['stroke'],
                    'weight':    x['properties']['stroke-width'],
                    'fillColor': x['properties']['fill'],
                    'opacity':   0.7,
                }).add_to(input_map)
            
            #Add legend
            colormap.caption = f'{param}'
            colormap.add_to(input_map)

            return input_map
        
def add_countours_predictions(data, parameter='annual_flood_risk'):

    map_predictions = folium.Map(location=[53.368998, -1.602912], zoom_start=7)

    data = data.dropna()

    geo_coord = geo.get_gps_lat_long_from_easting_northing(data['easting'].to_list(),
                                                                data['northing'].to_list())  
    data['latitude'] = geo_coord[0]
    data['longitude'] = geo_coord[1]
        
    #Define colormap for rainfall, elevation or tide level data
    colormap = cm.linear.PuRd_09.scale(data[parameter].min(), 
                                               data[parameter].max()).to_step(10)

            
    vmin, vmax = data[parameter].min(), data[parameter].max()

    x_coord = data['longitude'].to_numpy()
    y_coord = data['latitude'].to_numpy()
    z_coord = data[parameter].to_numpy()

    # Remove NaN values from the coordinates and values
    valid_indices = ~np.isnan(x_coord) & ~np.isnan(y_coord) & ~np.isnan(z_coord)
    x_coord = x_coord[valid_indices]
    y_coord = y_coord[valid_indices]
    z_coord = z_coord[valid_indices]

     # Create the grid
    x_grid, y_grid = np.meshgrid(np.linspace(x_coord.min(), x_coord.max(), 2000), np.linspace(y_coord.min(), y_coord.max(), 2000))

    # # Interpolate the data for each grid point
    z_grid = interp.griddata((x_coord, y_coord), z_coord, (x_grid, y_grid), 
                                      fill_value=np.nanmean(z_coord), method='linear')

    # # Smooth the data
    z_grid = ndimage.gaussian_filter(z_grid, sigma=20, order=0, mode='wrap')

    # # Create contour plot to map
    contours = plt.contourf(x_grid, y_grid, z_grid, levels=len(colormap.index), colors=colormap.colors, vmin=vmin, vmax=vmax)
    plt.close()

     # # Converting matplotlib contourf to geojson so that it can be plotted on folium map
    geojson = geojsoncontour.contourf_to_geojson(contourf=contours, min_angle_deg=3, fill_opacity=0.85)

            # Plot the contour plot on folium
    folium.GeoJson(
                geojson,
                style_function=lambda x: {
                    'color':     x['properties']['stroke'],
                    'weight':    x['properties']['stroke-width'],
                    'fillColor': x['properties']['fill'],
                    'opacity':   0.7,
                }).add_to(map_predictions)
            
            #Add legend
    colormap.caption = f'{parameter}'
    colormap.add_to(map_predictions)

    return map_predictions