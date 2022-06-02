import time
time_start = time.time()
import Data_Functions as dfunc
import gurobipy as gp
from gurobipy import GRB
import folium
import osmnx as ox
# import networkx as nx
# import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import graphviz_layout


def model_builder(lat, lng, d):
    time_start = time.time()
    print(time.time() - time_start)
    time_start = time.time()
    G = dfunc.initalize_map((lat, lng), d/6)
    print(time.time() - time_start)
    time_start = time.time()
    # start = dfunc.start_node(G, (lat, long))
    nodes = dfunc.get_nodes(G)
    print(time.time() - time_start)
    time_start = time.time()
    edge_df = dfunc.edge_database(G)
    print(time.time() - time_start)
    time_start = time.time()
    start, lat, lng = dfunc.node_database(G, lat)
    print(time.time() - time_start)
    time_start = time.time()
    edge_df = dfunc.edge_database(G)
    print(time.time() - time_start)
    time_start = time.time()
    dist = dfunc.get_distance_matrix_2(edge_df)
    print(time.time() - time_start)
    time_start = time.time()
    m = gp.Model()

    # Variables: is city 'i' adjacent to city 'j' on the tour?
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')
    seq = m.addVars(nodes, obj=nodes, vtype=GRB.INTEGER, name='u')
    print(lat)
    print(lng)
    root = start
    print(root)
    # Variables: Is each node visited? Will use in constraints

    # Constraints: At most, two edges incident to each city
    for i0 in nodes:
        m.addConstr(
            gp.quicksum([vars[i, j] for i, j in dist.keys() if i == i0]) ==
            gp.quicksum([vars[j, i] for j, i in dist.keys() if i == i0]))

    m.addConstr(gp.quicksum([vars[j, i] for j, i in dist.keys() if i == root]) == 1)

    m.addConstr(gp.quicksum([vars[i, j] * dist[i, j] for i, j in vars.keys()]) >= int(d))

    for i, j in dist.keys():
        if i != root and j != root:
            m.addConstr(seq[i] - seq[j] + len(nodes) * vars[i, j] <= len(nodes) - 1)

    m.addConstr(seq[root] == 1)
    m.addConstrs(seq[i] >= 2 for i in nodes if i != root)
    m.addConstrs(seq[i] <= len(nodes) for i in nodes if i != root)

    m._vars = vars
    m.setObjective(gp.quicksum([vars[i, j] * dist[i, j] for i, j in vars.keys()]), GRB.MINIMIZE)
    m.optimize()
    print(time.time() - time_start)
    time_start = time.time()
    # Retrieve solution
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    # Print Route Distance
    route_length = 0
    for i in selected:
        route_length += (dist[i[0], i[1]])
    print(route_length)
    print(time.time() - time_start)
    time_start = time.time()
    # List to OSMNX Graph and Plot graph
    master_list = []
    for i in selected:
        new_list = [i[0], i[1]]
        master_list.append(new_list)
    print("number of edges {}.".format(len(master_list)))
    # ox.plot_graph_routes(G, master_list, route_colors='red')

    # Prepping  and making Folium Map

    tour_pairs = dfunc.ordered_tour(master_list)

    print(tour_pairs)
    new_list = []
    for i in tour_pairs:
        new_list.append(i[0])

    new_list.append(new_list[0])
    for i in new_list:
        print(i)
    print(time.time() - time_start)
    time_start = time.time()
    test_map = folium.Map(location=(lat, lng), zoom_start=60)
    test_map = ox.folium.plot_route_folium(G, new_list,route_map=test_map, color='blue', weight=5, opacity=0.7)
    test_map.save('C:/Users/yoav_/PycharmProjects/Opti_Run_App/templates/run.html')
    print(time.time() - time_start)
    time_start = time.time()
    '''
    Gi = nx.DiGraph()
    Gi.add_edges_from(tour_pairs)
    nx.draw(Gi, pos=graphviz_layout(Gi),  with_labels=True, font_size=8)
    pos = nx.spring_layout(Gi)
    '''
    return 'route.html'

# Press the green button in the gutter to run the script.

# knapsack algorithm