function initMap() {
    // Create a Leaflet map with a default location and zoom level
    var map = L.map('map').setView([51.505, -0.09], 13);

    // Add a tile layer (you may need to replace this with a suitable tile layer)
    L.tileLayer('https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.{ext}', {
        attribution: '&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://www.stamen.com/" target="_blank">Stamen Design</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        ext: 'png'
    }).addTo(map);

    // Declare a variable to store the marker
    var marker;

    // Initialize Google Places Autocomplete for the search input
    var input = document.getElementById('address');
    var autocomplete = new google.maps.places.Autocomplete(input);

    // Listen for the 'place_changed' event to handle the selected place
    autocomplete.addListener('place_changed', function () {
        var place = autocomplete.getPlace();

        // Check if place has geometry (location information)
        if (place.geometry) {
            // Get the latitude and longitude
            var lat = place.geometry.location.lat();
            var lng = place.geometry.location.lng();

            // If there's an existing marker, remove it
            if (marker) {
                map.removeLayer(marker);
            }

            // Create a new marker and add it to the map
            marker = L.marker([lat, lng]).addTo(map);

            // Bind a popup with the address to the marker
            marker.bindPopup(place.formatted_address).openPopup();

            // Update the Leaflet map view
            map.setView([lat, lng], 13);
        }
    });
}

// Attach the callback function to the window object
window.initMap = initMap;
