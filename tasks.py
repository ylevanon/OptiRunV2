import folium
from model import Model, Run, Graph, RouteParser


def add_user(form_data):
    distance = int(form_data["distance"])
    graph = Graph(distance=distance, address=form_data["address"])
    run = Run(distance=form_data["distance"], address=form_data["address"], graph=graph)
    lat_lng = run.get_lat_long()
    route_parser = RouteParser()
    start_of_path = route_parser.path_to_start(
        run.starting_node, run.model_root_node, graph.graph
    )
    model = Model()
    dist_mtrx = graph.get_distance_matrix()
    selected = model.build_model(
        graph.get_distance_matrix(), graph.get_nodes(), run.model_root_node, distance
    )

    route_length = route_parser.find_route_length(selected, dist_mtrx)
    tour_pairs = route_parser.create_ordered_tour_from_edges(selected)
    final_tour = route_parser.create_final_tour(
        tour_pairs, start_of_path, run.model_root_node
    )
    test_map = folium.Map(
        location=lat_lng,
        zoom_start=100,
        width="100%",
        height="55%",
    )
    iframe = folium.IFrame(run.address, width=100, height=50)

    popup = folium.Popup(iframe, max_width=200)

    folium.Marker(lat_lng, popup=popup).add_to(test_map)
    test_map = route_parser.map_creation(graph.graph, final_tour, test_map)
    test_map.save("new_route.html")
    return "route.html", round(route_length / 1609.34, 2)
