import json
import os

VizWiz_ANN_PATH = './annotations/'
splits = ['train', 'val']

for split in splits:
    valid_img = set()
    with open(os.path.join(VizWiz_ANN_PATH, split+'.json'), 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    data['annotations'] = []
    imgs = data['images']
    data['images'] = []

    for caption in annotations:
        if caption['is_rejected'] or caption['is_precanned']:
            continue
        else:
            data['annotations'].append(caption)
            valid_img.add(caption['image_id'])

    for img in imgs:
        if img['id'] in valid_img:
            data['images'].append(img)

    with open(os.path.join('./data', split+'.json'), 'w') as f:
        json.dump(data, f)
