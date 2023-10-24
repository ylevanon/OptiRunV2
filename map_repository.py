import osmnx as ox
import pandas as pd
import networkx as nx
import warnings
from collections import deque
from shapely.errors import ShapelyDeprecationWarning
from scipy.spatial.distance import cdist
from geopy.geocoders import Nominatim


warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
ox.config(use_cache=True, log_console=True)


def get_closest_coordinates(point, points):
    """
    Find the closest point from a list of points to a given point.

    Parameters:
    - point (tuple): The coordinates (x, y) of the target point.
    - points (list): List of points to search for the closest one.

    Returns:
    - get_closest_coordinates (tuple): The coordinates of the closest point.
    """
    return points[cdist([point], points).argmin()]


def match_value(df, col1, x, col2):
    """
    Match a value x from col1 row to the corresponding value in col2.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col1 (str): The column name from which to match the value.
    - x: The value to match.
    - col2 (str): The column name to retrieve the matched value.

    Returns:
    - matched_value: The matched value from col2.
    """
    return df[df[col1] == x][col2].values[0]


def split(a, n):
    """
    Split a list into n sublists.

    Parameters:
    - a (list): The list to be split.
    - n (int): The number of sublists.

    Returns:
    - generator: Generator of n sublists.
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def get_lat_long(address):
    """
    Get latitude and longitude coordinates for a given address using Geopy.

    Parameters:
    - address (str): The address to geocode.

    Returns:
    - tuple: The (latitude, longitude) coordinates.
    """
    geolocator = Nominatim(user_agent="my_geocoder")
    location = geolocator.geocode(address)

    if location:
        lat, lon = location.latitude, location.longitude
        return lat, lon
    else:
        print(f"Could not find coordinates for address: {address}")
        return None


def find_get_closest_coordinates(node_df, address):
    """
    Find the closest node in the network to a given address.

    Parameters:
    - node_df (pd.DataFrame): DataFrame containing node information.
    - address (str): The target address.

    Returns:
    - start_node: The closest node to the given address.
    """
    coordinates = get_lat_long(address)
    lat, lng = coordinates[0], coordinates[1]
    lat_lng_df = pd.DataFrame(
        {
            "x": [lat],
            "y": [lng],
        }
    )
    node_df["point"] = [(x, y) for x, y in zip(node_df["x"], node_df["y"])]
    lat_lng_df["point"] = [(x, y) for x, y in zip(lat_lng_df["x"], lat_lng_df["y"])]
    lat_lng_df["closest"] = [
        get_closest_coordinates(x, list(node_df["point"])) for x in lat_lng_df["point"]
    ]
    lat_lng_df["node from"] = [
        match_value(node_df, "point", x, "node from") for x in lat_lng_df["closest"]
    ]
    start_node = lat_lng_df["node from"].values.tolist()
    return start_node[0]


def get_model_start_point(street_adj, node_adj, start_node):
    """
    Perform Breadth-First Search to find a suitable starting node.

    Parameters:
    - street_adj (dict): Dictionary representing the street count matrix.
    - node_adj (dict): Dictionary representing the adjacency matrix.
    - start_node: The initial node to start the search.

    Returns:
    - suitable_start_node: A suitable starting node for the route.
    """
    if street_adj[start_node] >= 4.0:
        return start_node

    queue = deque([start_node])
    visited = set([start_node])

    while queue:
        current_node = queue.popleft()

        if street_adj[current_node] >= 4.0:
            return current_node

        for neighbor in node_adj[current_node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)

    # If no suitable starting node is found, return None or handle accordingly
    return None


def path_to_start(start_node, end_node, G):
    """
    Find the shortest path from start_node to end_node in the OSMnx graph.

    Parameters:
    - start_node: The starting node.
    - end_node: The destination node.
    - G (networkx.Graph): The OSMnx graph.

    Returns:
    - list: List representing the shortest path from start_node to end_node.
    """
    if start_node == end_node:
        return []
    else:
        return nx.shortest_path(G, start_node, end_node)


def find_route_length(selected_edges, dist_dict):
    """
    Calculate the total length of a route based on selected edges.

    Parameters:
    - selected_edges (list): List of edges selected for the route.
    - dist_dict (dict): Dictionary representing the distance matrix.

    Returns:
    - float: Total length of the route.
    """
    route_length = 0
    for i in selected_edges:
        route_length += dist_dict[i[0], i[1]]
    return route_length


def convert_node_ids_to_coordinates(G, tour):
    """
    Convert node IDs in a tour to latitude and longitude coordinates.

    Parameters:
    - G (networkx.Graph): The OSMnx graph.
    - tour (list): List of node IDs representing the tour.

    Returns:
    - list: List of (latitude, longitude) coordinates representing the tour.
    """
    tour_coordinates = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in tour]
    return tour_coordinates


def create_ordered_tour_from_edges(selected_edges):
    """
    Create an ordered tour from a list of selected edges.

    Parameters:
    - G (networkx.Graph): The OSMnx graph.
    - selected_edges (list): List of edges selected for the route.

    Returns:
    - list: List of tuples representing the ordered tour.
    """
    edge_list = [(edge[0], edge[1]) for edge in selected_edges]
    input_dict = dict(edge_list)
    start_point = edge_list[0][0]

    tour_list = []
    for _ in range(len(edge_list)):
        tour_list.append((start_point, input_dict[start_point]))
        start_point = input_dict[start_point]

    return tour_list


def create_final_tour(ord_tour, intro, end):
    """
    Create the final tour by incorporating introduction and end nodes.

    Parameters:
    - ord_tour (list): List of tuples representing the ordered tour.
    - intro (list): List of introduction nodes.
    - end: End node.

    Returns:
    - list: List representing the final tour.
    """
    final_tour = [step[0] for step in ord_tour]
    final_tour.append(final_tour[0])

    if len(intro) > 0:
        final_tour = final_tour[1:]
        while final_tour[-1] != end:
            last_node = final_tour.pop(0)
            final_tour.append(last_node)
        final_tour.insert(0, end)

        for i in range(len(intro) - 2, -1, -1):
            final_tour.append(intro[i])

    return final_tour


def map_creation(G, lst, fol_map):
    """
    Create a Folium map of the route.

    Parameters:
    - G (networkx.Graph): The OSMnx graph.
    - lst (list): List of node IDs representing the route.
    - fol_map: The Folium map to which the route will be added.

    Returns:
    - folium.Map: The Folium map with the added route.
    """
    return ox.plot_route_folium(
        G, lst, route_map=fol_map, color="blue", weight=5, opacity=0.7
    )
