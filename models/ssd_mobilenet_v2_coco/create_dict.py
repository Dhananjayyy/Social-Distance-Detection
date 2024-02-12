import json

d = {}

with open("coco_91cl_bkgr.txt") as f:
    counter = 0
    for line in f:
        d[counter] = line.strip()
        counter += 1

print(d)

with open("classes.json", "w") as f:
    f.write(json.dumps(d))