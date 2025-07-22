import paramiko
import os
# Connection info
hostname = "merlin.fit.vutbr.cz"
username = "xsmida06"
password = "Te5Hejvofe"

remote_path = "/home/matko/Desktop/neighbors_image_path"
local_folder = "/home/matko/Desktop/download_neighbors"


# Connect to the server
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname, username=username, password=password)
#
# Open SFTP session
sftp = client.open_sftp()
# sftp.chdir(remote_path)

# List and download files
for line in open(remote_path):
    remote_filename = line.strip()

    local_file = os.path.join(local_folder, os.path.basename(remote_filename))
    if remote_filename != "not_found":
        print(f"Downloading {remote_filename} -> {local_folder}")
        sftp.get(remote_filename, local_file)

# Clean up
sftp.close()
client.close()