# Prepare data

## Requirements

- Python 2.7

## VizWiz

### Download COCO vocabulary

Download preprocessed coco vocabulary from [here](http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/data/cocotalk_vocab.json) and put it in `data/`. It is necessary when fine-tuning the model which was originally trained on MSCOCO-Captions to VizWiz-Captions. 

### Download preprocessed VizWiz captions

Download preprocessed VizWiz captions for [train](http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/data/train.json), [val](http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/data/val.json) and [test](http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/data/test.json) split. We remove the pre-canned and rejected captions in the train and val split, and create a dummy caption for each test image. We add dummy captions for the test split as placeholders to make it compatible with the dataloader.

Or download the raw [annotations](http://ivc.ischool.utexas.edu/VizWiz_final/caption/annotations.zip) and preprocess them by running:

```bash
$ python scripts/remove_reject.py
$ python scripts/create_test_dummy.py
```

### Extract image meta information and build vocabulary

Build image meta and vocabulary for `VizWiz-Captions` and `VizWiz-Captions + MSCOCO-Captions` by running:

```bash
$ python scripts/prepro_labels_vizwiz.py
```

`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. For `VizWiz-Captions` , the image information and vocabulary are dumped into `data/vizwiztalk.json` and discretized caption data are dumped into `data/vizwiztalk_label.h5`. For `VizWiz-Captions + MSCOCO-Captions`, they are dumped into `data/vizwiztalk_withCOCO.json` and `data/vizwiztalk_withCOCO_label.h5`. The processed files for `VizWiz-Captions` are used for training from scrach and those for `VizWiz-Captions + MSCOCO-Captions` are used for fine-tuning.

To use the pretrained models on `MSCOCO-Captions` directly for evaluation, first build image meta and vocabulary by running:

```bash
$ python scripts/prepro_labels_vizwiz_pretrained.py
```

### Download Bottom-up features for images

Download pre-extracted feature from [here](http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/data/tsv.zip) or use the [bottom-up attention model](https://github.com/peteanderson80/bottom-up-attention) to extract features (We mainly modified `tools/generate_tsv.py` to adapt to VizWiz-Captions. We also put our modified `generate_tsv.py` in `scripts/` of this repo).

Then run:

```bash
python scripts/make_bu_data_vizwiz.py
```

This will create `data/vizwizbu_fc`, `data/vizwizbu_att` and `data/vizwizbu_box`. 


### Pre-process N-Gram for captions

Pre-process the dataset and get the cache for calculating cider score for [SCST](https://arxiv.org/abs/1612.00563):

For training from scratch, run:

```bash
$ python scripts/prepro_ngrams_vizwiz.py --dict_json data/vizwiztalk.json --output_pkl data/vizwiz-train
```

This will create `data/vizwiz-train-idxs.p` and `data/vizwiz-train-words.p`

For fine-tuning, run:

```bash
$ python scripts/prepro_ngrams_vizwiz.py --dict_json data/vizwiztalk_withCOCO.json --output_pkl data/vizwiz-train-withCOCO
```

This will create `data/vizwiz-train-withCOCO-idxs.p` and `data/vizwiz-train-withCOCO-words.p`

