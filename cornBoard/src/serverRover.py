import socket
import threading


class RoverServer:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Add this line
        self.server.bind((self.ip, self.port))
        self.clients = {}
        self.client_data = {}
        self.running = False
        self.rovers = {}

    def start(self):
        self.server.listen()
        self.running = True
        print(f"Server is listening on {self.ip}:{self.port}")
        threading.Thread(target=self.accept_client, daemon=True).start()

    def accept_client(self):
        while self.running:
            client_conn, client_addr = self.server.accept()
            print(f"Client connected: {client_addr}")
            self.clients[client_addr] = client_conn
            threading.Thread(target=self.handle_client, args=(client_addr, client_conn), daemon=True).start()

    def handle_client(self, addr, conn):
        while self.running:
            try:
                data = conn.recv(1024).decode()
                if data:
                    self.client_data[addr] = data
            except Exception as e:
                print(f"Error: {e}")
                conn.close()
                del self.clients[addr]
                if addr in self.client_data:  # Remove the client's data after the connection is reset
                    del self.client_data[addr]
                break

    def send_instruction(self, addr, instruction_string):
        if addr in self.clients:
            self.clients[addr].send(instruction_string.encode())

    def get_data(self, addr):
        return self.client_data.get(addr)

    def close(self):
        self.running = False
        for conn in list(self.clients.values()):  # Create a copy of the values before iterating
            conn.close()
        self.server.close()
        print("Server closed")
