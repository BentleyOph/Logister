import os
os.environ["OMP_NUM_THREADS"] = "2"

import time
import requests
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict
import folium
from folium.plugins import MarkerCluster, AntPath , PolyLineTextPath,FeatureGroupSubGroup
import networkx as nx
import osmnx as ox
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from functools import lru_cache
from jinja2 import Template


@lru_cache(maxsize=10000)
def osrm_distance(lat1, lon1, lat2, lon2):
    url = f"http://localhost:5000/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data["code"] == "Ok":
            distance = data["routes"][0]["distance"]
            duration = data["routes"][0]["duration"]
            geometry = data["routes"][0].get("geometry")
            return distance, duration, geometry
        else:
            print(f"OSRM request failed: {data.get('message', 'Unknown error')} for points ({lat1}, {lon1}) to ({lat2}, {lon2})")
            return float('inf'), float('inf'), None
    except requests.exceptions.RequestException as e:
        print(f"OSRM request failed due to network error: {e}")
        return float('inf'), float('inf'), None

def lat_lon_to_cartesian(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    R = 6371000
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z

def cartesian_to_lat_lon(x, y, z):
    R = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / R)
    lon = np.arctan2(y, x)
    return np.degrees(lat), np.degrees(lon)

def cluster_and_rank_locations(locations, n_clusters=6):
    if len(locations) < n_clusters:
        n_clusters = len(locations)
    
    cartesian_coords = np.array([lat_lon_to_cartesian(lat, lon) for _, _, lat, lon in locations])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(cartesian_coords)
    
    centers_cartesian = kmeans.cluster_centers_
    centers_lat_lon = [cartesian_to_lat_lon(x, y, z) for x, y, z in centers_cartesian]
    
    clusters = kmeans.predict(cartesian_coords)
    
    cluster_locations = defaultdict(list)
    
    for i, (shop_id, shop_name, lat, lon) in enumerate(locations):
        cluster = clusters[i]
        center_lat, center_lon = centers_lat_lon[cluster]
        
        distance, duration, _ = osrm_distance(lat, lon, center_lat, center_lon)
        
        if distance != float('inf'):
            cluster_locations[cluster].append((shop_id, shop_name, lat, lon, distance, duration))
        else:
            print(f"Skipping unreachable location: Shop ID {shop_id} - {shop_name}")
    
    for cluster in cluster_locations:
        cluster_locations[cluster].sort(key=lambda x: x[4])
    
    return centers_lat_lon, cluster_locations # cluster_locations is a dictionary with cluster number as key and list of locations as value

def optimize_routes(distance_matrix, depot, names=None):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, depot) 
    # The depot is the starting point of the route, in this case it is the first point in the distance matrix
    routing = pywrapcp.RoutingModel(manager) 
    #routing's objective is to find the shortest path between the points in the distance matrix and manager's job is to manage the indices of the points in the distance matrix

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add distance dimension
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack which means there is no waiting time at the locations
        3000000,  # vehicle maximum travel distance in meters
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name) # Get the distance dimension from the routing model
    distance_dimension.SetGlobalSpanCostCoefficient(100)  #global span is the difference between the max and min values across all the nodes

    # Set up search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30
    search_parameters.log_search = True 

    solution = routing.SolveWithParameters(search_parameters)

    def print_solution(manager, routing, solution, names):
        print(f"Objective: {solution.ObjectiveValue()}") #solution.ObjectiveValue() returns the total distance of the optimized route
        index = routing.Start(0) 
        plan_output = "Route:\n"
        route_distance = 0
        while not routing.IsEnd(index): #routing.IsEnd(index) returns True if the index is the end of the route
            node_index = manager.IndexToNode(index)
            plan_output += f" {node_index} ({names[node_index]}) ->"
            previous_index = index
            index = solution.Value(routing.NextVar(index)) # solution.Value(routing.NextVar(index)) returns the index of the next point in the optimized route
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        node_index = manager.IndexToNode(index)
        plan_output += f" {node_index} ({names[node_index]})\n"
        print(plan_output)
        print(f"Total distance: {route_distance}m")

    if solution:
        print_solution(manager, routing, solution, names)
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route
    else:
        print("No solution found.")
        return []    
def plot_route(coordinates, shop_names, distances_from_start, durations_from_start, dist=15000, zoom_start=15):
    median_tuple = tuple(np.median(np.array(coordinates), axis=0)) # median of the coordinates
    distances = [np.linalg.norm(np.array(t) - np.array(median_tuple)) for t in coordinates] # distance of each coordinate from the median
    closest_index = np.argmin(distances) # index of the coordinate closest to the median
    closest_tuple = coordinates[closest_index] # the coordinate closest to the median

    graph = ox.graph_from_point(center_point=closest_tuple, dist=dist, network_type="all") #download the graph from the closest point to the median

    map_center = closest_tuple
    mymap = folium.Map(location=map_center, zoom_start=zoom_start)

    num_stops = len(coordinates)
    colors = [
        '#FF0000', '#FF3300', '#FF6600', '#FF9900', '#FFCC00', '#FFFF00', 
        '#CCFF00', '#99FF00', '#66FF00', '#33FF00', '#00FF00'
    ]
    color_map = folium.LinearColormap(colors=colors, vmin=0, vmax=num_stops-1)

    for i in range(num_stops - 1): #num_stops - 1 because we are calculating the route between each stop and the next stop
        start_node = ox.distance.nearest_nodes(graph, coordinates[i][1], coordinates[i][0]) #find the nearest node in the graph to the start coordinate
        end_node = ox.distance.nearest_nodes(graph, coordinates[i + 1][1], coordinates[i + 1][0]) # find the nearest node in the graph next to the end coordinate 
        #end_node is the nearest node in the graph to the next stop
        route = nx.shortest_path(graph, start_node, end_node, weight="length")
        route_coordinates = [(graph.nodes[node]["y"], graph.nodes[node]["x"]) for node in route]

        color = color_map(i)
        route_polyline = folium.PolyLine(locations=route_coordinates, color=color, weight=4, opacity=0.8)
        mymap.add_child(route_polyline)

        PolyLineTextPath(
            route_polyline,
            '\u25BA',
            repeat=True,
            offset=10,
            attributes={'font-weight': 'bold', 'font-size': '14'}
        ).add_to(mymap)

    # Create a MarkerCluster for all stops
    all_stops_cluster = folium.FeatureGroup(name="All Stops")
    mymap.add_child(all_stops_cluster)

    # Create individual feature groups for each stop
    for i, (coord, name, distance, duration) in enumerate(zip(coordinates, shop_names, distances_from_start, durations_from_start)):
        popup_text = f"""
        <b>Stop {i+1}: {name}</b><br>
        Lat: {coord[0]:.6f}, Lon: {coord[1]:.6f}<br>
        Distance from start: {distance:.2f}m<br>
        Duration from start: {duration:.2f}s
        """
        marker = folium.Marker(
            location=coord,
            icon=folium.Icon(color='blue', icon='info-sign'),
            popup=popup_text
        )
        
        
        # Create a feature group for this stop
        fg = FeatureGroupSubGroup(all_stops_cluster,name=f"Stop {i+1}: {name}")
        marker.add_to(fg)
        mymap.add_child(fg)

   

    # Add start and end markers
    folium.Marker(location=coordinates[0], icon=folium.Icon(color="green", icon="play"), popup="Start").add_to(mymap)
    folium.Marker(location=coordinates[-1], icon=folium.Icon(color="red", icon="stop"), popup="End").add_to(mymap)

    # Add color scale legend
    color_map.add_to(mymap)
    color_map.caption = 'Route Progress'

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(mymap)

    return mymap


def remove_outliers(data, columns, factor=1.5):
    Q1 = data[columns].quantile(0.25)
    Q3 = data[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return data[~((data[columns] < lower_bound) | (data[columns] > upper_bound)).any(axis=1)]

def main():
    try:
        df = pd.read_excel("locations.xlsx")
        grouped_data = df.groupby("cluster_code")
        for cluster_code, group in grouped_data:
            print(f"\nProcessing main cluster: {cluster_code}")
            #number of locations  before removing outliers
            print(f"Number of locations before removing outliers: {len(group)}")
            try:
                group_without_outliers = remove_outliers(group, ['latitude', 'longitude'])
                #number of locations after removing outliers
                print(f"Number of locations after removing outliers: {len(group_without_outliers)}")
                locations = group_without_outliers[["shop_id", "shop_name", "latitude", "longitude"]].values.tolist()
            
                centers, sub_clusters = cluster_and_rank_locations(locations)
            
                for sub_cluster, points in sub_clusters.items():
                    #points is a list of tuples with shop_id, shop_name, lat, lon, distance, duration
                    distance_matrix = [[osrm_distance(p1[2], p1[3], p2[2], p2[3])[0] for p2 in points] for p1 in points]
                    #The distance matrix looks smth like this :
                    #[[0, 100, 200, 300],
                    #[100, 0, 150, 250],
                    #[200, 150, 0, 100],
                    #[300, 250, 100, 0]]


                    duration_matrix = [[osrm_distance(p1[2], p1[3], p2[2], p2[3])[1] for p2 in points] for p1 in points]
                    names = [p[1] for p in points]  # Extracting shop names
                    optimized_route = optimize_routes(distance_matrix, 0, names) # return a list of indices of the points in the optimized route 
                    # for example if the optimized route is [0, 3, 2, 1], it means that the first stop is the first point in the points list, the second stop is the 4th point in the points list and so on
                    
                    route_coordinates = [points[i][:4] for i in optimized_route] #[:4] means that we are taking the shop_id, shop_name, lat, lon
                    shop_names = [p[1] for p in route_coordinates]
                    
                    # Calculate distances and durations from start
                    distances_from_start = [0]
                    durations_from_start = [0]
                    for i in range(1, len(optimized_route)):
                        prev = optimized_route[i-1]
                        curr = optimized_route[i]
                        distances_from_start.append(distances_from_start[-1] + distance_matrix[prev][curr]) #an index of -
                        durations_from_start.append(durations_from_start[-1] + duration_matrix[prev][curr])
                    
                    m = plot_route([(p[2], p[3]) for p in route_coordinates], shop_names, distances_from_start, durations_from_start)
                    
                    safe_cluster_code = ''.join(e for e in f"{cluster_code}_{sub_cluster}" if e.isalnum())
                    file_name = f"route_map_{safe_cluster_code}.html"
                    
                    os.makedirs('route_maps', exist_ok=True)
                    m.save(os.path.join('route_maps', file_name))
                    
                    print(f"Route map saved for cluster {cluster_code}, sub-cluster {sub_cluster}")
            
            except Exception as e:
                print(f"Error processing cluster {cluster_code}: {e}")
                continue

        print("\nOSRM Cache Info:")
        print(osrm_distance.cache_info())

    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
