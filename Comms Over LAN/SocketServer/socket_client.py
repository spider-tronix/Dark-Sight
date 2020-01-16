import socket
data = " ".join(sys.argv[1:])
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(data + "\n", ("192.168.1.1", 6001))
try:
    received = sock.recv(1024)
while True:
    print "Sent:     {}".format(data)
    print "Received: {}".format(received)
    sock.sendto('more' + "\n", ("192.168.1.1", 6001))
    received = sock.recv(1024)
except:
    print "No more messages"