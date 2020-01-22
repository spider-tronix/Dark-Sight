import paramiko

ssh_client = paramiko.SSHClient()

ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# hostname = input("Hostname: ")
# usename = input("Username: ")
# password = input("Password: ")
hostname = '192.168.43.185'
usename = 'pi'
password = 'ni6ga2rd'
ssh_client.connect(hostname=hostname, username=usename, password=password)

print(ssh_client)
