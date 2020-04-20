"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

### source activate py2.7 ###

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import sys
import re
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import skimage.io
from PIL import Image

sys.path.append('COCOAPI/')
from pycocotools.coco import COCO

VizWiz_ANN_PATH = 'data/'
COCO_TRAIN_VOCAB_PATH = 'data/cocotalk_vocab.json'

corrupt_list = []

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

def split_sentence(sentence):
  """ break sentence into a list of words and punctuation """
  toks = []
  for word in [s.strip().lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
    # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
    if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
      toks += list(word)
    else:
      toks.append(word)
  # Remove '.' from the end of the sentence - 
  # this is EOS token that will be populated by data layer
  if toks[-1] != '.':
    return toks
  return toks[:-1]

def load_caption_vocab(vocab_path=COCO_TRAIN_VOCAB_PATH):
  info = json.load(open(vocab_path))
  ix_to_word = info['ix_to_word']
  vocab = []
  vocab_size = len(ix_to_word)
  for i in range(vocab_size):
    vocab.append(ix_to_word[str(i+1)])
  return vocab

def build_vocab(params, base_vocab):
  vocabulary = load_caption_vocab(vocab_path=base_vocab)
  offset = len(vocabulary)
  print("number of words in the base vocab: %d\n" % (offset))
  return vocabulary

def encode_captions(params, data_split, wtoi):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """
  max_length = params['max_length']
  img_count = 0
  caption_count = 0
  test_count = 0
  for dataset in data_split:
    annFile='%s/%s.json' % (VizWiz_ANN_PATH, dataset)
    coco = COCO(annFile)
    for image_id,anns in coco.imgToAnns.iteritems():
      if image_id in corrupt_list:
        continue
      img_count += 1
      caption_count += len(anns)
  N = img_count
  M = caption_count  # total number of captions

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1
  img_counter = 0
  imgInfo = []
  
  for dataset in data_split:
    annFile='%s/%s.json' % (VizWiz_ANN_PATH, dataset)
    coco = COCO(annFile)
    for image_id,anns in coco.imgToAnns.iteritems():
      if image_id in corrupt_list:
        continue
      image_info = coco.imgs[image_id]
      #image_path = '%s/%s' % (image_info['file_name'].split('_')[1], image_info['file_name'])
      image_path = '%s' % (image_info['file_name'])
      jimg = {}
      if dataset == 'train' or dataset == 'val':
        jimg['split'] = 'train'
      elif dataset == 'test':
        jimg['split'] = 'test'
      jimg['file_path'] = image_path
      jimg['id'] = image_info['id']
      n = len(anns)
      assert n > 0, 'error: some image has no captions'
      Li = np.zeros((n, max_length), dtype='uint32')
      for j, ann in enumerate(anns):
        caption_sequence = split_sentence(ann['caption'])
        label_length[caption_counter] = min(max_length, len(caption_sequence))
        caption_counter += 1
        for k,w in enumerate(caption_sequence):
          if k < max_length:
            if not w in wtoi:
              w = 'UNK'
            Li[j,k] = wtoi[w]
      # note: word indices are 1-indexed, and captions are padded with zeros
      label_arrays.append(Li)
      label_start_ix[img_counter] = counter
      label_end_ix[img_counter] = counter + n - 1
      img_counter += 1
      counter += n
      imgInfo.append(jimg)
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print('encoded captions to array of size ', L.shape)
  return N, L, label_start_ix, label_end_ix, label_length, imgInfo

def main(params):
  seed(123) # make reproducible
  
  # create the vocab integrating MSCOCO 
  vocab = build_vocab(params, base_vocab=COCO_TRAIN_VOCAB_PATH)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table 
  # encode captions in large arrays, ready to ship to hdf5 file
  N, L, label_start_ix, label_end_ix, label_length, imgInfo = encode_captions(params, ['train', 'val', 'test'], wtoi)
  # create output h5 file
  f_lb = h5py.File(params['output_h5']+'_pretrained_label.h5', "w")
  f_lb.create_dataset("labels", dtype='uint32', data=L)
  f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
  f_lb.close()
  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['images'] = imgInfo  
  json.dump(out, open(params['output_json']+'_pretrained.json', 'w'))
  print('wrote ', params['output_json']+'_pretrained.json')

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--output_json', default='data/vizwiztalk', help='output json file')
  parser.add_argument('--output_h5', default='data/vizwiztalk', help='output h5 file')
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')

  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
