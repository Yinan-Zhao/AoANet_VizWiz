# Benchmarking AoANet on VizWiz-Captions

This repository includes the code for benchmarking [Attention on Attention for Image Captioning](https://arxiv.org/abs/1908.06954) on [VizWiz-Captions](https://vizwiz.org/tasks-and-datasets/image-captioning/).

## Requirements

- Python 3.6
- Java 1.8.0
- PyTorch 1.0
- tensorboardX


## Training AoANet

### Prepare data

See details in `data/README.md`. We combine both the train and val split of VizWiz-Captions for training.


### Training from scratch

```bash
$ CUDA_VISIBLE_DEVICES=0 sh train_vizwiz.sh
```

See `opts.py` for the options. You can also download our trained model [here](http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/log/log_aoanet_vizwiz_rl). 

### Fine-tuning models pretrained on MSCOCO-Captions

Download the pretrained models (log_aoanet_rl) from [here](https://drive.google.com/drive/folders/1ab0iPNyxdVm79ml-oozsIlH7H6t6dIVl?usp=sharing). 

Then run:

```bash
$ CUDA_VISIBLE_DEVICES=0 sh finetune_vizwiz.sh
```

## Evaluation

Generate predictions for the test split using the model pretrained on MSCOCO-Captions.

```bash
$ CUDA_VISIBLE_DEVICES=0 sh eval_pretrained.sh
```

Generate predictions for the test split using the model trained from scratch.

```bash
$ CUDA_VISIBLE_DEVICES=0 sh eval_scratch.sh
```

Generate predictions for the test split using the fine-tuned model.

```bash
$ CUDA_VISIBLE_DEVICES=0 sh eval_finetune.sh
```

The results will be saved in `vis/`

### Performance

Upload the generated results in `vis/` to the [evaluation server](https://evalai.cloudcv.org/web/challenges/challenge-page/525/overview) to evalute on the test split. See below for the scores of the model trained from scratch.

Model | Bleu-1 | Bleu-2 | Bleu-3 | Bleu-4 | ROUGE-L | METEOR | SPICE | CIDEr
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
from_scratch | 65.91 | 47.77 | 33.68 | 23.41 | 46.56 | 20.00 | 15.11 | 59.77



## Reference

If you find this repo helpful, please consider citing:

```
@article{gurari2020captioning,
  title={Captioning Images Taken by People Who Are Blind},
  author={Gurari, Danna and Zhao, Yinan and Zhang, Meng and Bhattacharya, Nilavra},
  journal={arXiv preprint arXiv:2002.08565},
  year={2020}
}

@inproceedings{huang2019attention,
  title={Attention on Attention for Image Captioning},
  author={Huang, Lun and Wang, Wenmin and Chen, Jie and Wei, Xiao-Yong},
  booktitle={International Conference on Computer Vision},
  year={2019}
}
```

## Contact

Contact Yinan Zhao (yinanzhao@utexas.edu) for any question.

## Acknowledgements

This repository is based on [AoANet](https://github.com/husthuaan/AoANet), and you may refer to it for more details about the code.
