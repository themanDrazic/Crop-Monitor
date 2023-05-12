import socket
import time

IDENTIFIER = "BETA-02"
IP = "10.49.98.163"
PORT = 12345


class RoverClient:
    def __init__(self, ip, port, identifier):
        self.ip = ip
        self.port = port
        self.id = identifier
        self.client = None

    def start(self):
        while True:
            try:
                self.connect_to_server()
                self.listen()
            except (ConnectionError, OSError):
                print("Connection lost, attempting to reconnect")
                time.sleep(5)

    def connect_to_server(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.client.connect((self.ip, self.port))
                print("Connected to the server")

                break
            except:
                print("Failed to connect, retrying in 5 seconds")
                time.sleep(5)

    def listen(self):
        while True:
            instruction = self.client.recv(1024).decode()
            if not instruction:  # Check if instruction is empty
                raise ConnectionError("Server disconnected")  # Raise an exception to trigger reconnection

            print(f"Received instruction: {instruction}")

            if instruction == "$CLOSE_SERVER$":
                self.exit()
            if instruction == "$SEND_ALL_DATA$":
                DATE = time.strftime("%Y-%m-%d")
                TIME = time.strftime('%H:%M:%S')
                data = [self.id, DATE, TIME]
                self.client.send(str(data).encode())

    def exit(self):
        self.client.close()


client = RoverClient(IP, PORT, IDENTIFIER)
client.start()
