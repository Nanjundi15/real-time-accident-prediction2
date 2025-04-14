import requests
import folium
import webbrowser
from IPython.display import IFrame

# ====== API Configuration ======
api_key = "8cd0273b24c077ce74c48408382a76b9"  # Your OpenWeatherMap API Key

# ====== List of Cities in India with Coordinates ======
locations = [
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Delhi", "lat": 28.7041, "lon": 77.1025},
    {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462}
]

# ====== Create Folium Map ======
# Center the map around India
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# ====== Fetch Weather Data and Add Markers ======
for loc in locations:
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={loc['lat']}&lon={loc['lon']}&appid={api_key}&units=metric"

    response = requests.get(weather_url)

    if response.status_code == 200:
        weather_data = response.json()

        # Extract weather details
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        humidity = weather_data['main']['humidity']
        pressure = weather_data['main']['pressure']
        weather_desc = weather_data['weather'][0]['description'].capitalize()
        wind_speed = weather_data['wind']['speed']

        # Weather Info Popup
        popup_text = f"""
        🌦️ <b>{loc['name']} Weather</b><br>
        🌡️ Temp: {temp}°C (Feels like: {feels_like}°C)<br>
        💧 Humidity: {humidity}%<br>
        🌬️ Wind Speed: {wind_speed} m/s<br>
        📉 Pressure: {pressure} hPa<br>
        ☁️ Conditions: {weather_desc}
        """

        # Add marker with weather info
        folium.Marker(
            location=[loc['lat'], loc['lon']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color="blue", icon="cloud")
        ).add_to(m)

        print(f"✅ Added weather data for {loc['name']}")
    else:
        print(f"❌ Failed to fetch weather data for {loc['name']}")

# ====== Save & Display Map ======
map_file = "india_weather_map.html"
m.save(map_file)

# Display map directly in Colab
IFrame(map_file, width=1000, height=600)
