import shutil
import os

# Source: The formatted VisDrone folder we just built
SRC_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-test-dev"

# Destination: The default MOT directory the code expects
DST_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\mot"

print(f"Copying files from:\n{SRC_DIR}\nto:\n{DST_DIR}\n")
print("This might take a minute depending on your drive speed...")

# copytree with dirs_exist_ok=True safely merges the folders without deleting existing files
try:
    shutil.copytree(SRC_DIR, DST_DIR, dirs_exist_ok=True)
    print("\nSuccess! All videos, images, and annotations have been merged into the default 'mot' directory.")
except Exception as e:
    print(f"\nAn error occurred: {e}")