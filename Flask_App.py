from __future__ import print_function # In python 2.7
import sys
from flask import Flask, redirect, url_for, render_template, request
import folium
# from OptiRun.Model import main
import Model
import osmnx as ox
import pandas as pd
import numpy as np
import warnings
import time
time_start = time.time()
import Data_Functions as dfunc
import gurobipy as gp
from gurobipy import GRB
import folium
import osmnx as ox
import jsonify
from shapely.errors import ShapelyDeprecationWarning



app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/map")
def map():
    return render_template('map.html')

@app.route("/preference")
def preference():
    return render_template('preference.html')

@app.route("/route", methods=["POST"])
def route():
    latitude = request.form.get("latitude", type=float)
    longitude = request.form.get("longitude", type=float)
    street_crossing = request.form.get("street_crossing")
    distance = request.form.get("distance", type=float)

    # longitude = request.args.get('longitude', type=float)
    # latitude = request.args.get('latitude', type=float)

    return render_template(Model.model_builder(latitude, longitude, distance), latitude=latitude, longitude=longitude,
                           distance=distance, street_crossing=street_crossing)


# @app.route("/Mile_SLO")
# def Mile_SLO():
#     return render_template('Mile_SLO.html')


# @app.route('/folium_map')
# def folium_map():
#     #map using stamen terrain
#     start_coords = (35.28053391188845, -120.66373491879064)
#     pin_coords = (35.29053391188845, -120.67373491879064)
#     # attempted map
#     map = folium.Map(location=start_coords,
#     # tiles="Stamen Toner",
#     zoom_start=15)
#
#     folium.Marker(
#     pin_coords,
#     popup="Click anywhere for coordinates, then copy them. After, return back to home screen.",
#     tooltip="Click Here for Help",
#     icon=folium.Icon(color='red')
#     ).add_to(map)
#
#     # map.add_child(folium.ClickForMarker(popup="Starting Location"))
#     folium.LatLngPopup().add_to(map)
#
#     return map._repr_html_()

if __name__ == "__main__":
    app.run()

