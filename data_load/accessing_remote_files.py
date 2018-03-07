import paramiko
ssh_client=paramiko.SSHClient()
# ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname='kamiak.wsu.edu',\
	username='n.kannappanjayakodi',password='Ni11hil@n1')

stdin,stdout,stderr=ssh_client.exec_command('ls')

print("output ", stdout.readlines(), \
	"error ", stderr.readlines())