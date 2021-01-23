import json


def read_json(file):
    with open(file, 'r') as jfile:
        data = json.load(jfile)

    return data