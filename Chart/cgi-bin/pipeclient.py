import win32file
import json


PIPE_NAME = r'\\.\pipe\nnpipe'
PIPE_BUFFER_SIZE = 14

file_handle = win32file.CreateFile(PIPE_NAME,
                                   win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                                   win32file.FILE_SHARE_WRITE, None,
                                   win32file.OPEN_EXISTING, 0, None)
try:
    msg = win32file.ReadFile(file_handle, PIPE_BUFFER_SIZE, None)
    msg = msg[1].decode()
    data = msg.split('#')
finally:
    win32file.CloseHandle(file_handle)


result = {'batch': int(data[0]), 'loss': float(data[1])}

json_result = json.dumps(result)  # 将字典转换为json
print("content-type:text/json")
print()
print(json_result)