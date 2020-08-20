import multiprocessing 
import subprocess

def worker(file):
    subprocess.Popen(['python', file])

if __name__ == "__main__": 
    files = [r'/media/sudhar/D Drive/Spider/Dark-sight/Dark-Sight/Comms Over LAN/thermal_client.py', r'/media/sudhar/D Drive/Spider/Dark-sight/Dark-Sight/Comms Over LAN/testing.py']
    for i in files: 
        p = multiprocessing.Process(target = worker(i))
        p.start()