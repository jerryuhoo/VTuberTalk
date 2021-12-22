import os

path="./"
files=os.listdir(path)

if os.path.exists("./text.txt"):
    os.remove("./text.txt")

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

files = sorted_alphanumeric(files)

for file in files:
    if file.endswith('.txt'):
        pass
    else:
        continue
    position=path+file
    print(position)
    with open(position ,'r') as f:
        for line in f.readlines():
            with open("./text.txt","a") as p:
                p.write(str(file) + " " + line + "\n")
