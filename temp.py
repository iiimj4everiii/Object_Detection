import json


def switch(arg):
    switcher = {
        0:  0,
        1:  2,
        2:  3,
        3:  1,
        4:  10,
        5:  10,
        6:  10,
        7:  10,
        8:  10,
        9:  10,
        10: 10,
        11: 8
    }

    return switcher.get(arg, "nothing")


with open('annotations.json') as f:
    data = json.load(f)

for anno in data['annotations']:
    anno['category_id'] = switch(anno['category_id'])

for cat in data['categories']:
    if cat['name'] == 'obstacles':
        cat['name'] = 'N/A'
    elif cat['name'] == 'biker':
        cat['name'] = 'bicycle'
    elif cat['name'] == 'pedestrian':
        cat['name'] = 'person'
    elif cat['name'] == 'trafficLight':
        cat['name'] = 'traffic light'
    elif cat['name'] == 'trafficLight-Green':
        cat['name'] = 'traffic light'
    elif cat['name'] == 'trafficLight-GreenLeft':
        cat['name'] = 'traffic light'
    elif cat['name'] == 'trafficLight-Red':
        cat['name'] = 'traffic light'
    elif cat['name'] == 'trafficLight-RedLeft':
        cat['name'] = 'traffic light'
    elif cat['name'] == 'trafficLight-Yellow':
        cat['name'] = 'traffic light'
    elif cat['name'] == 'trafficLight-YellowLeft':
        cat['name'] = 'traffic light'

with open('annotations_out.json', 'w') as json_file:
    json.dump(data, json_file, indent=4, sort_keys=True)

print()
