
import serial
import time
import requests
import random
from haversine import haversine, Unit
import flexpolyline as fp
from datetime import datetime
import math

# === CONFIGURATION ===
USE_MOCK_LOCATION = False  # ✅ Toggle to simulate GPS
MOCK_COORDINATES = (55.86586456400274, -4.282926941808761)

# Deviation simulation settings
SIMULATE_DEVIATION = False  # Enable simulated deviation from route
DEVIATION_START_TIME = 30  # Start deviating after 30 seconds
DEVIATION_DURATION = 20    # Deviate for 20 seconds
DEVIATION_AMOUNT = 0.005   # Amount to deviate (in degrees)

HERE_API_KEY = "pIpkl3NsvuYVap99tvBB3fkdFGQzAzm0NcaGJq1mmGk"
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 4800
DEVIATION_THRESHOLD_METERS = 30
ROUTING_URL = "https://router.hereapi.com/v8/routes"

# Simulation state variables
start_time = None
is_deviating = False
deviation_direction = None
current_simulated_position = MOCK_COORDINATES
route_index = 0
current_route = []

# === UTIL ===
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# === GPS HELPERS ===
def nmea_to_decimal(degree_min, direction):
    degrees = float(degree_min[:2])
    minutes = float(degree_min[2:])
    decimal = degrees + minutes / 60
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_lat_lon(nmea_line):
    if nmea_line.startswith('$GPRMC') and ',A,' in nmea_line:
        parts = nmea_line.split(',')
        lat = nmea_to_decimal(parts[3], parts[4])
        lon = nmea_to_decimal(parts[5], parts[6])
        return lat, lon
    return None

def generate_deviated_position(base_position, route):
    """Generate a position that deviates from the route"""
    global deviation_direction
    
    if not deviation_direction:
        # Choose a random direction to deviate (perpendicular to the route)
        if len(route) > route_index + 1:
            # Calculate direction vector of the route
            next_point = route[min(route_index + 1, len(route) - 1)]
            current_point = route[route_index]
            
            # Vector from current to next point
            dx = next_point[1] - current_point[1]
            dy = next_point[0] - current_point[0]
            
            # Perpendicular vector (rotate 90 degrees)
            perp_dx = -dy
            perp_dy = dx
            
            # Normalize and set the deviation direction
            magnitude = math.sqrt(perp_dx**2 + perp_dy**2)
            if magnitude > 0:
                deviation_direction = (perp_dy/magnitude, perp_dx/magnitude)
            else:
                # Fallback if points are too close
                deviation_direction = (1, 0)
        else:
            # Simple default deviation direction
            deviation_direction = (1, 0)  # Deviate east
    
    # Calculate the deviated position
    lat = base_position[0] + deviation_direction[0] * DEVIATION_AMOUNT
    lon = base_position[1] + deviation_direction[1] * DEVIATION_AMOUNT
    
    return (lat, lon)

def simulate_movement_along_route(route):
    """Simulate movement along the route or deviation from it"""
    global route_index, current_simulated_position, start_time, is_deviating
    
    if not start_time:
        start_time = time.time()
    
    elapsed_time = time.time() - start_time
    
    # Check if we should start deviating
    if SIMULATE_DEVIATION and elapsed_time >= DEVIATION_START_TIME and elapsed_time <= (DEVIATION_START_TIME + DEVIATION_DURATION):
        if not is_deviating:
            is_deviating = True
            log("🔄 Simulating deviation from route")
        
        # Generate a position that's off the route
        if len(route) > 0:
            base_position = route[min(route_index, len(route) - 1)]
            current_simulated_position = generate_deviated_position(base_position, route)
        else:
            # If no route available, just deviate from the current position
            current_simulated_position = (
                MOCK_COORDINATES[0] + DEVIATION_AMOUNT,
                MOCK_COORDINATES[1] + DEVIATION_AMOUNT
            )
    else:
        if is_deviating and elapsed_time > (DEVIATION_START_TIME + DEVIATION_DURATION):
            is_deviating = False
            log("🔄 Returning to route after deviation")
            # Reset deviation direction for next time
            deviation_direction = None
        
        # Move along the route normally
        if len(route) > 0:
            route_index = min(route_index + 1, len(route) - 1)
            current_simulated_position = route[route_index]
    
    return current_simulated_position

def read_gps_position(serial_conn):
    global current_route
    
    if USE_MOCK_LOCATION:
        if SIMULATE_DEVIATION and current_route:
            position = simulate_movement_along_route(current_route)
            log(f"🧪 Using mock GPS position: {position} {'(DEVIATING)' if is_deviating else ''}")
            return position
        else:
            return MOCK_COORDINATES

    while True:
        line = serial_conn.readline().decode('ascii', errors='replace').strip()
        if line:
            log(f"Raw NMEA: {line}")
        if line.startswith('$GPRMC'):
            if ',A,' in line:
                pos = extract_lat_lon(line)
                if pos:
                    log(f"✅ Parsed GPS: {pos}")
                    return pos
            else:
                log("⚠️  GPRMC has no valid GPS fix (status=V)")

# === ROUTING & DEVIATION ===
def get_truck_route(origin, destination):
    global current_route, route_index
    
    log(f"🚛 Routing from {origin} to {destination}")
    
    # Format coordinates correctly (lat,lon without parentheses)
    origin_str = f"{origin[0]},{origin[1]}"
    dest_str = f"{destination[0]},{destination[1]}"
    
    params = {
        "origin": origin_str,
        "destination": dest_str,
        "transportMode": "truck",
        "routingMode": "fast",
        "apiKey": HERE_API_KEY,
        "return": "polyline,summary",
    }
    # Add truck parameters as integers instead of floats, and without brackets in the keys
    params["truck.height"] = 4
    params["truck.width"] = 2
    params["truck.length"] = 18
    params["truck.weight"] = 40000
    
    try:
        response = requests.get(ROUTING_URL, params=params)
        
        # Add debug information
        log(f"Request URL: {response.url}")
        log(f"Response status: {response.status_code}")
        
        if response.ok:
            data = response.json()
            log("✅ Route fetched successfully")
            
            # Extract the polyline safely
            if 'routes' in data and len(data['routes']) > 0:
                if 'sections' in data['routes'][0] and len(data['routes'][0]['sections']) > 0:
                    if 'polyline' in data['routes'][0]['sections'][0]:
                        polyline = data['routes'][0]['sections'][0]['polyline']
                        decoded_route = decode_polyline(polyline)
                        log(f"Route decoded with {len(decoded_route)} points")
                        # Reset the route index when we get a new route
                        current_route = decoded_route
                        route_index = 0
                        return decoded_route
                    else:
                        log("❌ No polyline in response")
                else:
                    log("❌ No sections in route")
            else:
                log("❌ No routes in response")
                
            # If we got here, something was wrong with the response structure
            log(f"Response content: {data}")
            return []
        else:
            log(f"❌ Error response from API: {response.status_code}")
            log(f"Error details: {response.text}")
            return []
    except Exception as e:
        log(f"❌ Exception during routing request: {str(e)}")
        return []

def decode_polyline(polyline_str):
    try:
        points = fp.decode(polyline_str)
        # flexpolyline returns points as [lat, lng, z] where z is optional
        # Convert to simple (lat, lng) tuples
        return [(point[0], point[1]) for point in points]
    except Exception as e:
        log(f"❌ Error decoding polyline: {str(e)}")
        return []

def is_deviated(current_pos, route):
    if not route:
        log("⚠️ No route to check deviation against")
        return True
    
    # Check if any point on the route is close to our current position
    min_distance = float('inf')
    for waypoint in route:
        distance = haversine(current_pos, waypoint, unit=Unit.METERS)
        min_distance = min(min_distance, distance)
        if distance <= DEVIATION_THRESHOLD_METERS:
            return False
    
    log(f"⚠️ Distance to nearest route point: {min_distance:.1f} meters (threshold: {DEVIATION_THRESHOLD_METERS} meters)")
    return True

# === MAIN LOOP ===
def main():
    try:
        end_lat, end_lon = 51.50696501689779, -0.1275170888122869  # Destination: London
        destination = [end_lat, end_lon]

        log("Opening GPS serial port...")
        
        # Only open serial port if not using mock location
        if not USE_MOCK_LOCATION:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        else:
            ser = None

        log("⏳ Waiting for valid GPS lock (GPRMC with status A)...")
        initial_pos = read_gps_position(ser)
        log(f"🚦 Starting from GPS: {initial_pos}")

        route = get_truck_route(initial_pos, destination)
        
        # Initial route check
        if not route:
            log("❌ Could not get initial route, retrying in 5 seconds...")
            time.sleep(5)
            route = get_truck_route(initial_pos, destination)
            
        while True:
            current_pos = read_gps_position(ser)

            if is_deviated(current_pos, route):
                log("🚨 Off-route detected. Recalculating...")
                route = get_truck_route(current_pos, destination)
                if not route:
                    log("⚠️ Could not recalculate route, will try again later")
            else:
                log("🛣️  On route.")

            # if route:
            #     next_waypoints = route[:5]
            #     log(f"➡️  Next waypoints for control: {next_waypoints}")

            if route:
                next_waypoints = route[:5]
                log(f"➡️  Next waypoints for control: {next_waypoints}")
                
                # Display distances between consecutive waypoints
                if len(next_waypoints) > 1:
                    log("📏 Distances between waypoints:")
                    for i in range(len(next_waypoints) - 1):
                        distance = haversine(next_waypoints[i], next_waypoints[i+1], unit=Unit.METERS)
                        log(f"   • {i+1} to {i+2}: {distance:.1f} meters")

            else:
                log("⚠️ No route available")

            time.sleep(2)

    except KeyboardInterrupt:
        log("🛑 Interrupted by user. Shutting down.")
    except Exception as e:
        log(f"❌ Error: {e}")
    finally:
        if 'ser' in locals() and ser is not None and ser.is_open:
            ser.close()
            log("📴 Serial port closed.")

if __name__ == "__main__":
    main()



# import serial
# import time
# import requests
# from haversine import haversine, Unit
# import flexpolyline as fp
# from datetime import datetime

# # === CONFIGURATION ===
# USE_MOCK_LOCATION = True  # ✅ Toggle to simulate GPS
# MOCK_COORDINATES = (55.86586456400274, -4.282926941808761)

# HERE_API_KEY = "pIpkl3NsvuYVap99tvBB3fkdFGQzAzm0NcaGJq1mmGk"
# SERIAL_PORT = '/dev/ttyUSB0'
# BAUD_RATE = 4800
# DEVIATION_THRESHOLD_METERS = 30
# ROUTING_URL = "https://router.hereapi.com/v8/routes"

# # === UTIL ===
# def log(msg):
#     print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# # === GPS HELPERS ===
# def nmea_to_decimal(degree_min, direction):
#     degrees = float(degree_min[:2])
#     minutes = float(degree_min[2:])
#     decimal = degrees + minutes / 60
#     if direction in ['S', 'W']:
#         decimal = -decimal
#     return decimal

# def extract_lat_lon(nmea_line):
#     if nmea_line.startswith('$GPRMC') and ',A,' in nmea_line:
#         parts = nmea_line.split(',')
#         lat = nmea_to_decimal(parts[3], parts[4])
#         lon = nmea_to_decimal(parts[5], parts[6])
#         return lat, lon
#     return None

# def read_gps_position(serial_conn):
#     if USE_MOCK_LOCATION:
#         log(f"🧪 Using mock GPS position: {MOCK_COORDINATES}")
#         return MOCK_COORDINATES

#     while True:
#         line = serial_conn.readline().decode('ascii', errors='replace').strip()
#         if line:
#             log(f"Raw NMEA: {line}")
#         if line.startswith('$GPRMC'):
#             if ',A,' in line:
#                 pos = extract_lat_lon(line)
#                 if pos:
#                     log(f"✅ Parsed GPS: {pos}")
#                     return pos
#             else:
#                 log("⚠️  GPRMC has no valid GPS fix (status=V)")

# # === ROUTING & DEVIATION ===
# def get_truck_route(origin, destination):
#     log(f"🚛 Routing from {origin} to {destination}")
    
#     # Format coordinates correctly (lat,lon without parentheses)
#     origin_str = f"{origin[0]},{origin[1]}"
#     dest_str = f"{destination[0]},{destination[1]}"
    
#     params = {
#         "origin": origin_str,
#         "destination": dest_str,
#         "transportMode": "truck",
#         "routingMode": "fast",
#         "apiKey": HERE_API_KEY,
#         "return": "polyline,summary",
#     }
#         # Add truck parameters as integers instead of floats, and without brackets in the keys
#     params["truck.height"] = 4
#     params["truck.width"] = 2
#     params["truck.length"] = 18
#     params["truck.weight"] = 40000
    
#     try:
#         response = requests.get(ROUTING_URL, params=params)
        
#         # Add debug information
#         log(f"Request URL: {response.url}")
#         log(f"Response status: {response.status_code}")
        
#         if response.ok:
#             data = response.json()
#             log("✅ Route fetched successfully")
            
#             # Extract the polyline safely
#             if 'routes' in data and len(data['routes']) > 0:
#                 if 'sections' in data['routes'][0] and len(data['routes'][0]['sections']) > 0:
#                     if 'polyline' in data['routes'][0]['sections'][0]:
#                         polyline = data['routes'][0]['sections'][0]['polyline']
#                         decoded_route = decode_polyline(polyline)
#                         log(f"Route decoded with {len(decoded_route)} points")
#                         return decoded_route
#                     else:
#                         log("❌ No polyline in response")
#                 else:
#                     log("❌ No sections in route")
#             else:
#                 log("❌ No routes in response")
                
#             # If we got here, something was wrong with the response structure
#             log(f"Response content: {data}")
#             return []
#         else:
#             log(f"❌ Error response from API: {response.status_code}")
#             log(f"Error details: {response.text}")
#             return []
#     except Exception as e:
#         log(f"❌ Exception during routing request: {str(e)}")
#         return []

# def decode_polyline(polyline_str):
#     try:
#         points = fp.decode(polyline_str)
#         # flexpolyline returns points as [lat, lng, z] where z is optional
#         # Convert to simple (lat, lng) tuples
#         return [(point[0], point[1]) for point in points]
#     except Exception as e:
#         log(f"❌ Error decoding polyline: {str(e)}")
#         return []

# def is_deviated(current_pos, route):
#     if not route:
#         log("⚠️ No route to check deviation against")
#         return True
        
#     for waypoint in route:
#         if haversine(current_pos, waypoint, unit=Unit.METERS) <= DEVIATION_THRESHOLD_METERS:
#             return False
#     return True

# # === MAIN LOOP ===
# def main():
#     try:
#         end_lat, end_lon = 51.50696501689779, -0.1275170888122869  # Destination: London
#         destination = [end_lat, end_lon]

#         log("Opening GPS serial port...")
        
#         # Only open serial port if not using mock location
#         if not USE_MOCK_LOCATION:
#             ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
#         else:
#             ser = None

#         log("⏳ Waiting for valid GPS lock (GPRMC with status A)...")
#         initial_pos = read_gps_position(ser)
#         log(f"🚦 Starting from GPS: {initial_pos}")

#         route = get_truck_route(initial_pos, destination)
        
#         # Initial route check
#         if not route:
#             log("❌ Could not get initial route, retrying in 5 seconds...")
#             time.sleep(5)
#             route = get_truck_route(initial_pos, destination)
            
#         while True:
#             current_pos = read_gps_position(ser)

#             if is_deviated(current_pos, route):
#                 log("🚨 Off-route detected. Recalculating...")
#                 route = get_truck_route(current_pos, destination)
#                 if not route:
#                     log("⚠️ Could not recalculate route, will try again later")
#             else:
#                 log("🛣️  On route.")

#             if route:
#                 next_waypoints = route[:5]
#                 log(f"➡️  Next waypoints for control: {next_waypoints}")
#             else:
#                 log("⚠️ No route available")

#             time.sleep(2)

#     except KeyboardInterrupt:
#         log("🛑 Interrupted by user. Shutting down.")
#     except Exception as e:
#         log(f"❌ Error: {e}")
#     finally:
#         if 'ser' in locals() and ser is not None and ser.is_open:
#             ser.close()
#             log("📴 Serial port closed.")

# if __name__ == "__main__":
#     main()
