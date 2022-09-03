"""
Last Date: 8/13/2022
Author: Yoav Levanoni
Project: OptiRun - The Smart Running App
"""
import time
import Data_Functions as dfunc
import gurobipy as gp
from gurobipy import GRB
import folium
from memory_profiler import profile
import networkx as nx
import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import graphviz_layout

@profile
def model_builder(lat, lng, d, address):
    time_start = time.time()
    d = d * 1609.34
    G = dfunc.initialize_map((lat, lng), d/3)
    print('G = dfunc.initalize_map((lat, lng), d/4)')
    print(time.time() - time_start)
    node_df, nodes = dfunc.node_database(G, lat)
    edge_df = dfunc.edge_database(G)
    dist = dfunc.get_distance_matrix(edge_df)
    adj_mtrx = dfunc.get_adjacency_matrix(edge_df)
    street_mtrx = dfunc.get_street_count_matrix(adj_mtrx)
    start = dfunc.find_closest_point(node_df, lat, lng)
    print(start)
    end = dfunc.BFS(street_mtrx, adj_mtrx, start)
    print(end)
    intro = dfunc.path_to_start(start, end, G)
    print(intro)
    m = gp.Model()
    # Variables: vars is the set of edges in the graph, seq is the set of nodes in the graph
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')
    seq = m.addVars(nodes, obj=nodes, vtype=GRB.INTEGER, name='u')
    if start != end:
        root = end
    else:
        root = start

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
    m.setParam('MIPGap', 0.15)
    m.optimize()

    # Retrieve solution
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    # Print Route Distance
    route_length = dfunc.find_length(selected, dist)
    tour_pairs = dfunc.ordered_tour(G, selected)
    print(start)
    print(end)
    print(intro)
    final_tour = dfunc.make_final_tour(tour_pairs, intro, end)
    test_map = folium.Map(location=(lat, lng), zoom_start=100, width='100%', height='55%')
    iframe = folium.IFrame(address,
                       width=100,
                       height=50)

    popup = folium.Popup(iframe,
                     max_width=200)

    marker = folium.Marker([lat, lng],
                       popup=popup).add_to(test_map)
    test_map = dfunc.map_creation(G, final_tour, test_map)
    test_map.save('C:\Users\yoav_\PycharmProjects\Opti_Run_App\run.html')

    '''
    Gi = nx.DiGraph()
    Gi.add_edges_from(tour_pairs)
    nx.draw(Gi, pos=graphviz_layout(Gi),  with_labels=True, font_size=8)
    pos = nx.spring_layout(Gi)
    nx.draw(Gi, pos, with_labels=True)
    plt.show()
    '''
    return 'route.html', round(route_length/1609.34, 2)

# model_builder(47.696210, -122.128250, 3, "10717 159th Ct Ne Redmond Wa")

