import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import warnings
from shapely.errors import ShapelyDeprecationWarning
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
ox.config(use_cache=True, log_console=True)


def initialize_map(coord, distance):
    G = ox.graph_from_point(coord, distance, network_type='walk')
    return G

def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]

def match_value(df, col1, x, col2):
    """ Match value x from col1 row to value in col2. """
    return df[df[col1] == x][col2].values[0]

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def find_closest_point(node_df, lat, lng):
    lat_lng_df = pd.DataFrame(
    {'x': [lat],
     'y': [lng],
    })
    node_df['point'] = [(x, y) for x,y in zip(node_df['x'], node_df['y'])]
    lat_lng_df['point'] = [(x, y) for x,y in zip(lat_lng_df['x'], lat_lng_df['y'])]
    lat_lng_df['closest'] = [closest_point(x, list(node_df['point'])) for x in lat_lng_df['point']]
    lat_lng_df['node from'] = [match_value(node_df, 'point', x, 'node from') for x in lat_lng_df['closest']]
    start_node = lat_lng_df["node from"].values.tolist()
    return start_node[0]


def node_database(G, lat):
    raw_node_data = list(G.nodes(data=True))
    node_db = pd.DataFrame(raw_node_data)
    node_db = node_db.set_axis(['node from', 'codes'], axis=1, inplace=False)
    node_db = pd.concat([node_db, node_db["codes"].apply(pd.Series)], axis=1)
    node_db = node_db.drop(columns="codes")
    node_db = node_db.rename(columns={'x': 'y', 'y': 'x'})
    node_db.to_excel('node_db.xlsx')
    return node_db, list(G.nodes(data=False))



def edge_database(G):
    edges = list(G.edges(data=True))
    raw_edge_data = list(G.edges(data=True))
    raw_edge_df = pd.DataFrame(raw_edge_data)
    transformed_edge_df = raw_edge_df.set_axis(['node from', 'node too', 'codes'], axis=1, inplace=False)
    edge_df = pd.concat([transformed_edge_df, transformed_edge_df["codes"].apply(pd.Series)], axis=1)
    database = edge_df.drop(columns="codes")
    database.to_csv('edge_db.xlsx')
    return database

def get_distance_matrix(database):
    database['from too'] = list(zip(database['node from'], database['node too']))
    dist = dict(zip(database['from too'], database['length']))
    return dist

def get_adjacency_matrix(database):
    Gz = nx.from_pandas_edgelist(database, 'node from', 'node too', create_using=nx.DiGraph())
    adj = nx.adjacency_matrix(Gz)
    adj = nx.convert.to_dict_of_lists(Gz)
    return adj

def get_street_count_matrix(adj_matrix):
    street_dict = {x:len(adj_matrix[x]) for x in adj_matrix}
    return street_dict

def BFS(street_adj, node_adj, start_node):
    print(street_adj[start_node])
    if street_adj[start_node] >= 4.0:
        return start_node
    my_queue = []
    for  i in node_adj[start_node]:
        my_queue.append(i)
    visited = [start_node]
    while len(my_queue) > 0:
        end_node = my_queue.pop(0)
        if street_adj[end_node] == 4.0:
            return end_node
        else:
            for j in node_adj[end_node]:
                if j not in visited:
                    my_queue.append(j)
            visited.append(end_node)

def path_to_start(start_node, end_node, G):
    if start_node == end_node:
        return []
    else:
        return nx.shortest_path(G, start_node, end_node)

def find_length(selected_edges, dist_dict):
    route_length = 0
    for i in selected_edges:
        route_length += (dist_dict[i[0], i[1]])
    return route_length


def id_to_cord(G, tour):
    tour_cord = []
    for i in tour:
        tour_cord.append((G.nodes[i]['y'], G.nodes[i]['x']))
    return tour_cord


def ordered_tour(G, selected_edges):
    master_list = []
    for i in selected_edges:
        new_list = [i[0], i[1]]
        master_list.append(new_list)
    input_dict = dict(master_list)  # Convert list of `tuples` to dict
    elem = master_list[0][0]  # start point in the new list
    # ox.plot_graph_routes(G, master_list, route_colors='red')
    tour_list = []  # List of tuples for holding the values in required order
    for i in range(len(master_list)):
        tour_list.append((elem, input_dict[elem]))
        elem = input_dict[elem]
    return tour_list


def make_final_tour(ord_tour, intro, end):
    f_tour = []
    for i in ord_tour:
        f_tour.append(i[0])
    f_tour.append(f_tour[0])
    print('this is f_tour before appending intro')
    print(f_tour)
    if len(intro) <= 0:
        return f_tour
    else:
        f_tour = f_tour[1:]
        while f_tour[-1] != end:
            a = f_tour.pop(0)
            f_tour.append(a)
        f_tour.insert(0, end)
        print('f_tour sorted')
        print(f_tour)
        for i in range(len(intro) -2, -1, -1):
            print(intro[i])
            f_tour.append(intro[i])
        return f_tour


def map_creation(G, lst, fol_map):
    return ox.plot_route_folium(G, lst,route_map=fol_map, color='blue', weight=5, opacity=0.7)
