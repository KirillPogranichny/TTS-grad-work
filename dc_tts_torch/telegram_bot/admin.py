import socket

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
os.makedirs('/tmp/my_ipc_socket', exist_ok=True)
sock.bind('/tmp/my_ipc_socket')
sock.listen(1)