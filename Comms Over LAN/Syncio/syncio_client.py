"""import asyncio

async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 8888)

    print(f'Send: {message!r}')
    writer.write(message.encode())

    data = await reader.read(100)
    print(f'Received: {data.decode()!r}')

    print('Close the connection')
    writer.close()

inf = input("Givve something : ")
asyncio.run(tcp_echo_client(inf))"""


from socket import *
from threading import Thread

host = '127.0.0.1'
port = 8888
s = socket(AF_INET, SOCK_STREAM)
s.connect((host, port))

def Listener():
    try:
        while True:
            data = s.recv(1024).decode('utf-8')
            print('>', data)
    except ConnectionAbortedError:
        pass

t = Thread(target=Listener)
t.start()

try:
    while True:
        message = input('>')
        s.send(message.encode('utf-8'))
except EOFError:
    pass
finally:
    s.close()