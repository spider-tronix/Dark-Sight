import SocketServer
from subprocess import Popen,PIPE

class Handler(SocketServer.BaseRequestHandler):
    def handle(self):
        if not hasattr(self, 'Proc'):
            self.Proc = Popen('r.sh', stdout=PIPE)
        socket = self.request[1]
        socket.sendto(self.Proc.stdout.readline(),self.client_address)

if __name__ == "__main__":                         
    HOST, PORT = "0.0.0.0", 6001

    server = SocketServer.UDPServer((HOST, PORT), Handler)
    server.serve_forever()