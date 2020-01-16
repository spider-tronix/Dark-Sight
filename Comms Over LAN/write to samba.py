f = open("/run/user/1000/gvfs/smb-share:server=192.168.43.239,share=share/a.txt", "a")
f.write("Now the file has more content!")
f.close()