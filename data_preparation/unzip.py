# unzip the datasets.zip file

import zipfile
import os

# Define the path
path = 'datasets'
file = 'datasets.zip'

# Remove the zip file
# os.remove(file)

# Unzip the file
with zipfile.ZipFile(file, 'r') as zip_ref:
    zip_ref.extractall(path)
    
