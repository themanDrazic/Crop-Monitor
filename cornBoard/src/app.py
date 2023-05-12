import sys
from flask import Flask, render_template, jsonify
from tinydb import TinyDB, Query
import datetime
import threading
import time
import serverRover
import os
import json
sys.path.append("../templates")

app = Flask(__name__)

HOST = '0.0.0.0'
PORT = 12345
server = serverRover.RoverServer(HOST, PORT)

# New database for historical rovers
historical_db_path = '../data/historical_rovers.json'
if not os.path.exists(historical_db_path) or os.path.getsize(historical_db_path) == 0:
    with open(historical_db_path, 'w') as f:
        json.dump({"_default": {}}, f)
historical_db_lock = threading.Lock()
historical_db = TinyDB(historical_db_path)

def server_loop():
    server.start()
    client_timestamps = {}  # Dictionary to store the last received data timestamp for each client
    timeout_seconds = 5

    try:
        while True:
            for addr in list(server.clients.keys()):
                server.send_instruction(addr, "$SEND_ALL_DATA$")
                data = server.get_data(addr)

                if data:
                    print(f"Data from {addr}: {data}")
                    parsed_data = data.strip('[').strip(']').strip('\'').split(',')
                    with db_lock:
                        db.insert({'ID': parsed_data[0], 'TIME24hr': parsed_data[2], 'DATE': parsed_data[1]})
                    client_timestamps[addr] = time.time()  # Update the last received data timestamp
                    with historical_db_lock:  # Acquire the lock for historical database
                        Rover = Query()
                        if not historical_db.contains(Rover.ID == parsed_data[0]):
                            historical_db.insert({'ID': parsed_data[0], 'Connected': True})
                        else:
                            historical_db.update({'Connected': True}, Rover.ID == parsed_data[0])
                elif addr in client_timestamps and time.time() - client_timestamps[addr] > timeout_seconds:
                    # If the client has not sent data within the timeout window, remove it
                    del server.clients[addr]
                    del client_timestamps[addr]
                    print(f"Client disconnected: {addr}")

            with historical_db_lock:
                for rover in historical_db.all():
                    if rover['ID'] in client_timestamps:
                        if time.time() - client_timestamps.get(rover['ID'], 0) > timeout_seconds:
                            historical_db.update({'Connected': False}, Rover.ID == rover['ID'])
            time.sleep(1)
    except KeyboardInterrupt:
        server.close()

threading.Thread(target=server_loop).start()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_clients')
def get_clients():
    with historical_db_lock:
        clients = [rover['ID'] for rover in historical_db.all()]
    return jsonify(clients)

@app.route('/get_data')
def get_data():
    with db_lock:  # Acquire the lock
        data = db.all()
    return jsonify(data)


if __name__ == '__main__':
    db_path = '../db/' + str(datetime.datetime.now()) + '.json'
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        with open(db_path, 'w') as f:
            json.dump({"_default": {}}, f)
    db_lock = threading.Lock()
    db = TinyDB(db_path)
    app.run(debug=True, use_reloader=False)