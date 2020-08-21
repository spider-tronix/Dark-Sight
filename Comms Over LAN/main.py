import multiprocessing 
import subprocess
import os

dirname = os.path.dirname(__file__)
def worker(file):
    subprocess.Popen(['python', file])

def main():
    files = [os.path.join(dirname,'thermal_client.py'), os.path.join(dirname,'picam_client.py')]
    for i in files: 
        p = multiprocessing.Process(target = worker(i))
        p.start()

if __name__ == "__main__": 
    main()
    