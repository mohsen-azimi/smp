import os
import subprocess
import socket


def git_push():
    hostname = socket.gethostname()
    if hostname == "APSD-CDW-CIVL37" or "APC":
        subprocess.run(["git", "add", "*.py"])
        subprocess.run(["git", "commit", "-m", "updated"])
        subprocess.run(["git", "push"])
        print("Changes Pushed to GitHub")
    else:
        print("Code not running on Mohsen's Laptop, GitHub is not updated")



if __name__ == '__main__':
    git_push()