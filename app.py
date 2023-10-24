from __future__ import print_function  # In python 2.7
import os
import osmnx as ox
from flask import Flask, jsonify, render_template, request, redirect, url_for
import folium
from rq import Queue
from tasks import add_user
from worker import conn


app = Flask(__name__)
q = Queue(connection=conn)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/route")
def route():
    return render_template("route.html")


@app.route("/input", methods=["GET", "POST"])
def input():
    if request.method == "POST":
        form_data = request.form
        job = q.enqueue(add_user, form_data)
        return redirect(url_for("loading", task_id=job.id))
    return render_template("input.html")


@app.route("/loading/<task_id>")
def loading(task_id):
    # Check if the Celery task is finished

    job = q.fetch_job(task_id)
    status = job.get_status()
    if status in ["queued", "started", "deferred", "failed"]:
        return render_template("loading_screen.html", result=status, refresh=True)
    elif status == "finished":
        print(job.get_status())
        return render_template("route.html")


if __name__ == "__main__":
    app.run(port=8000, debug=True)
