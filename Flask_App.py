from __future__ import print_function # In python 2.7
import osmnx as ox
from flask import Flask, redirect, url_for, render_template, request, session
import folium
import geocoder
import Model as Model
import time
time_start = time.time()


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


@app.route("/preference", methods=["POST"])
def preference():
    address = request.form['address']
    print(address)
    location = geocoder.osm(address)
    if  location.lat is None:
        error_statement = "Please enter a valid address."
        return render_template("index.html", error_statement=error_statement)
    print(location)
    print(type(location))
    print(location.lat)
    print(location.lng)
    latitude = location.lat
    longitude = location.lng
    my_map = folium.Map(location=(latitude, longitude), zoom_start=100, width='100%', height='55%')
    iframe = folium.IFrame(address,
                       width=100,
                       height=50)

    popup = folium.Popup(iframe,
                     max_width=200)

    marker = folium.Marker([latitude, longitude],
                       popup=popup).add_to(my_map)
    my_map.save('/Opti_Run_App/templates/gmap.html')
    return render_template('preference.html', address=address, latitude=latitude, longitude=longitude)


@app.route("/route", methods=["POST"])
def route():
    street_crossing = request.form.get("street_crossing")
    distance = request.form.get("distance", type=float)
    print(distance)
    latitude = request.form.get("latitude", type=float)
    longitude = request.form.get("longitude", type=float)
    address = request.form.get("address",type = str)
    print("address")
    print(address)

    if not distance:
        error_statement = "All Form Fields Required."
        return render_template("preference.html", error_statement=error_statement, latitude=latitude, longitude=longitude, address=address)
    if distance < 0:
        error_statement = "Must enter a distance greater than 0."
        return render_template("preference.html", error_statement=error_statement, latitude=latitude, longitude=longitude, address=address)

    my_tup = Model.model_builder(latitude, longitude, distance, address)
    return render_template(my_tup[0], latitude=latitude, longitude=longitude,
                           distance=my_tup[1], street_crossing=street_crossing)


if __name__ == "__main__":
    app.run(debug=True)

