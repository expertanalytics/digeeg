import subprocess

result = subprocess.check_call(["ls", "-ljangfin"])
print(type(result))
