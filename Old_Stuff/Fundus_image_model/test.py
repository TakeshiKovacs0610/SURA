import os

def print_jpeg_files(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpeg') or filename.lower().endswith('.jpg'):
            print(filename)

# Example usage:
directory = '../data/BigOne/sample'  # Replace with your directory path
print_jpeg_files(directory)
