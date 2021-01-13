# Self-training with Noisy Student improves ImageNet classification
## Dataset
```
data/
  imagenet/
    train/
      ...
      n021015556/
        ..
        n021015556_12124.jpeg
	..
      n021015557/
      ...
    val/
      ...
      n021015556/
        ...
	ILSVRC2012_val_00003054.jpeg
	...
      n021015557/
      ...
```
## Model
## Dependency
- I use python3 (3.5.2) and python2 is not supported. 
- I use PyTorch (1.1.0), though tensorflow-gpu is necessary to launch tensorboard.
## Install
```
git clone --recurse-submodules (this repo)
cd $REPO_NAME/code
(use python >= 3.5)
pip3 install -r requirements.txt
```

### When using docker

build & push & run
```
sudo ./setup-docker.sh
```
directory structure
```
/home/
 /code/
 /data/
```

## Data Folder Structure
```
code/
 cli.py : executable check_dataloading, training, evaluating script
 config.py: default configs
 ckpt.py: checkpoint saving & loading
 train.py : training python configuration file
 evaluate.py : evaluating python configuration file
 infer.py : make submission from checkpoint
 logger.py: tensorboard and commandline logger for scalars
 utils.py : other helper modules
 dataloader/ : module provides data loaders and various transformers
  load_dataset.py: dataloader for classification
  vision.py: image loading helper
 loss/ 
 metric/ : accuracy and loss logging 
 optimizer/
 ...
data/
```
### Functions
```
utils.prepare_batch: move to GPU and build target
ckpt.get_model_ckpt: load ckpt and substitue model weight and args
load_dataset.get_iterator: load data iterator {'train': , 'val': , 'test': }
```
## How To Use
### First check data loading
```
cd code
python3 cli.py check_dataloader
```

### Training
```
cd code
python3 cli.py train
```

### Evaluation
```
cd code
python3 cli.py evaluate --ckpt_name=$CKPT_NAME
````
- Substitute CKPT_NAME to your preferred checkpoint file, e.g., `ckpt_name=model_name_simclr_ckpt_3/loss_0.4818_epoch_15`
## Results

## Contact Me
To contact me, send an email to sally20921@snu.ac.kr
