# SALD: Sign Agnostic Learning with Derivatives

This repository contains an implementation to the ICLR 2021 paper SALD: Sign Agnostic Learning with Derivatives.

SALD is a method for learning implicit neural representations of shapes directly from raw data. We generalize sign agnostic learning (SAL) to include derivatives: given an unsigned distance function to the input raw data, we advocate a novel sign agnostic regression loss, incorporating both pointwise values and gradients of the unsigned distance function. Optimizing this loss leads to a signed implicit function solution, the zero level set of which is a high quality and valid manifold approximation to the input 3D data.

For more details, please visit: https://openreview.net/pdf?id=7EDgLu9reQD.

### Installation Requirmenets
The code is compatible with python 3.6 and pytorch 1.10. Conda environment file is provided at ```./envsald.yml```.
### Usage

##### Data
The DFaust raw scans can be downloaded from http://dfaust.is.tue.mpg.de/downloads.
In order to be able to run the training process, the raw scans need to be preprocessed using:

```
cd ./code
python preprocess/preprocess_dfaust.py 
```

##### Training
For training, run:
```
cd ./code
python training/exp_runner.py
```

##### Predicting meshed surfaces with SALD trained network
We have uploaded SALD trained network. To produce predictions of unseen test scans, run:
```
cd ./code
python evaluate/evaluate.py  --parallel
```

### Citation
If you find our work useful in your research, please consider citing:

       @inproceedings{atzmon2021sald,
	author    = {Matan Atzmon and
	             Yaron Lipman},
  	title     = {{SALD:} Sign Agnostic Learning with Derivatives},
  	booktitle = {9th International Conference on Learning Representations, {ICLR} 2021},
  	year      = {2021}
	}
			
