import glob
import os

counter = 0
data = ""
string1 = ""
string2 = ""
string3 = ""

for folder in glob.glob("./Dataset/*Cannon*"):
    for filename in glob.glob(os.path.join(folder, "*small*long*.JPG")):
        string1 = filename
    for filename in glob.glob(os.path.join(folder, "*small*short*.JPG")):
        string2 = filename
    for filename in glob.glob(os.path.join(folder, "temp.txt")):
        string3 = filename
    data = string1 + "\t" + string2 + "\t" + string3 + "\n" + data

    print(string1 + " " + string2)
    counter += 1

with open("myFile_jpg.txt", "w") as myFile:
    myFile.write(data)

data = ""
for folder in glob.glob("./Dataset/*Cannon*"):
    for filename in glob.glob(os.path.join(folder, "*raw*long*.CR3")):
        string1 = filename
    for filename in glob.glob(os.path.join(folder, "*raw*short*.CR3")):
        string2 = filename
    for filename in glob.glob(os.path.join(folder, "temp.txt")):
        string3 = filename
    data = string1 + "\t" + string2 + "\t" + string3 + "\n" + data
    print(string1 + " " + string2)
    counter += 1

with open("myFile_raw.txt", "w") as myFile:
    myFile.write(data)

data = ""
for folder in glob.glob("./Dataset/*Cannon*"):
    for filename in glob.glob(os.path.join(folder, "*raw*long*.JPG")):
        string1 = filename
    for filename in glob.glob(os.path.join(folder, "*raw*short*.JPG")):
        string2 = filename
    for filename in glob.glob(os.path.join(folder, "temp.txt")):
        string3 = filename

    data = string1 + "\t" + string2 + "\t" + string3 + "\n" + data
    print(string1 + " " + string2)
    counter += 1

with open("myFile_raw_jpg.txt", "w") as myFile:
    myFile.write(data)
