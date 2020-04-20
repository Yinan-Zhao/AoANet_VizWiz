import json
import os

VizWiz_ANN_PATH = './annotations/'
splits = ['test']

caption_id = 155905

for split in splits:
    with open(os.path.join(VizWiz_ANN_PATH, split+'.json'), 'r') as f:
        data = json.load(f)

    data['annotations'] = []
    
    for img in data['images']:
        caption = {}
        caption['caption'] = 'the'
        caption['image_id'] = img['id']
        caption['id'] = caption_id
        caption['is_precanned'] = False
        caption['is_rejected'] = False
        caption['text_detected'] = False
        data['annotations'].append(caption)
        caption_id += 1

    with open(os.path.join('./data', split+'.json'), 'w') as f:
        json.dump(data, f)
