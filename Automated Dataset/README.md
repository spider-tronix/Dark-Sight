## Notes

[Samba Setup](https://www.linuxbabe.com/ubuntu/install-samba-server-ubuntu-16-04)

Mount the server locally 
```bash
sudo mount -t cifs -o username=syzygianinfern0,password=,uid=1000,gid=1000,forceuid,forcegid, //192.168.43.156/sambashare/ ~/sambaishere
```
