import osmnx as ox
import pandas as pd
import numpy as np
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def initalize_map(coord, distance):
    G = ox.graph_from_point(coord, distance, network_type='walk')
    return G


def start_node(G, coord):
    start_node = ox.nearest_nodes(G, coord[0], coord[1], return_dist=True)
    return start_node

def node_database(G, lat):
    raw_node_data = list(G.nodes(data=True))
    node_db = pd.DataFrame(raw_node_data)
    transform_node_db = node_db.set_axis(['node from', 'codes'], axis=1, inplace=False)
    edge_df = pd.concat([transform_node_db, transform_node_db["codes"].apply(pd.Series)], axis=1)
    database = edge_df.drop(columns="codes")
    # print(database)
    start = database.iloc[(database['y'] - lat).abs().argsort()[0],:]
    # database.to_csv('spread_node.csv')
    # print(start['node from'])
    return start['node from'], start['y'], start['x']

def get_nodes(G):
    return list(G.nodes(data=False))


def edge_database(G):
    edges = list(G.edges(data=True))
    raw_edge_data = list(G.edges(data=True))
    raw_edge_df = pd.DataFrame(raw_edge_data)
    transformed_edge_df = raw_edge_df.set_axis(['node from', 'node too', 'codes'], axis=1, inplace=False)
    edge_df = pd.concat([transformed_edge_df, transformed_edge_df["codes"].apply(pd.Series)], axis=1)
    database = edge_df.drop(columns="codes")
    # database.to_csv('ts.csv')
    return database


def get_distance_matrix(database):
    outlist = [(i, j)
               for i in database['node from']
               for j in database['node too']]
    new_database = pd.DataFrame(data=outlist, columns=['node from', 'node too'])
    overflow_database = pd.concat([database, new_database])
    matrix_database = overflow_database.drop_duplicates(subset=['node from', 'node too'], keep='first')
    matrix_database['length'] = matrix_database['length'].replace(np.nan, 10 ** 9)
    matrix_database['from too'] = list(zip(matrix_database['node from'], matrix_database['node too']))
    dist = dict(zip(matrix_database['from too'], matrix_database['length']))
    return dist


def get_distance_matrix_2(database):
    database['from too'] = list(zip(database['node from'], database['node too']))
    dist = dict(zip(database['from too'], database['length']))
    return dist


def id_to_cord(G, tour):
    tour_cord = []
    for i in tour:
        tour_cord.append((G.nodes[i]['y'], G.nodes[i]['x']))
    return tour_cord


def ordered_tour(master_list):
    input_dict = dict(master_list)  # Convert list of `tuples` to dict
    elem = master_list[0][0]  # start point in the new list
    new_list = []  # List of tuples for holding the values in required order
    for i in range(len(master_list)):
        new_list.append((elem, input_dict[elem]))
        elem = input_dict[elem]
    return new_list


