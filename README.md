
This repo contains the source code for our CVPR'19 work
"Patch-based discriminative feature learning for unsupervised person re-identification"

Our implementation is based on [Pytorch](https://pytorch.org/).


### Prerequisites
1. Pytorch 0.4.1
2. Python 3.6



### Preparation
 
- Data preparation

```bash {.line-numbers}
mkdir data

ln -s [PATH TO MSMT17_V1] ./data/MSMT17_V1
ln -s [PATH TO DUKE] ./data/DukeMTMC-reID
ln -s [PATH TO Market] ./data/Market
```

- set the path of ImageNet pretrained models
```bash {.line-numbers}
ln -s [THE PATH OF IMAGENET PRE-TRAINED MODELS] imagenet_models
```
### Run the code
- For pretraining the model
```bash {.line-numbers}
cd ./src/train
python supervised_train.py --gpu [CHOOSE WHICH GPU TO RUN] --exp-name [YOUR EXP NAME]
```
- Optionally, you can download our pretrained model from [google drive](https://drive.google.com/file/d/1KKzR0CoGPmEq00Aw-evH5odOAlwxuwX8/view?usp=sharing) or [baidu cloud](https://pan.baidu.com/s/17uy7VnBa037f5luMG7cGzw) (kvkz)
and place it in `./snapshot/MSMT17_PRE/`

```bash
mkdir ./snapshot
mkdir ./snapshot/MSMT17_PRE
cp [PATH TO PRETRAINED MODEL] ./snapshot/MSMT17_PRE/
# it means the name of the experiment of pretraining is 'MSMT17_PRE'  
```

- For unsupervised training
```bash {.line-numbers}
cd ./src/unsupervised

# for market
python unsupervised_train.py --data MARKET --gpu [CHOOSE WHICH GPU TO RUN] \
--pre-name [THE EXP NAME OF PRE-TRIANED MODEL] --exp-name [YOUR EXP NAME] \
--batch-size 42 --scale 15 --lr 0.0001 

 # for duke
python unsupervised_train.py --data DUKE --gpu [CHOOSE WHICH GPU TO RUN] \
--pre-name [THE EXP NAME OF PRE-TRIANED MODEL] --exp-name [YOUR EXP NAME] \
--batch-size 40 --scale 5 --lr 0.0001 

```


### Reference

If you find our work helpful in your research,
please kindly cite our paper:

Qize Yang, Hong-Xing Yu, Ancong Wu, Wei-Shi Zheng, "Patch-based discriminative feature 
learning for unsupervised person re-identification",
In CVPR, 2019.

bib:
```
@inproceedings{yu2019unsupervised,
  title={Patch-based discriminative feature learning for unsupervised person re-identification},
  author={Yang, Qize and Yu, Hong-Xing and Wu, Ancong and Zheng, Wei-Shi},
  year={2019},
  booktitle={IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

If you have any problem/question, please feel free to contact me at chitse.yang@gmail.com
or open an issue. Thank you.
