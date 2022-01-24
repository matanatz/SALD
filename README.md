# SALD: Sign Agnostic Learning with Derivatives

This repository contains an implementation to the ICLR 2021 paper SALD: Sign Agnostic Learning with Derivatives.

SALD is a method for learning implicit neural representations of shapes directly from raw data. We generalize sign agnostic learning (SAL) to include derivatives: given an unsigned distance function to the input raw data, we advocate a novel sign agnostic regression loss, incorporating both pointwise values and gradients of the unsigned distance function. Optimizing this loss leads to a signed implicit function solution, the zero level set of which is a high quality and valid manifold approximation to the input 3D data. The motivation behind SALD is that incorporating derivatives in a regression loss leads to a lower sample complexity, and consequently better fitting. In addition, we provide empirical evidence, as well as theoretical motivation in 2D that SAL enjoys a minimal surface property, favoring minimal area solutions. More importantly, we are able to show that this property still holds for SALD, i.e.,  with derivatives included.

For more details visit: https://openreview.net/pdf?id=7EDgLu9reQD.


### Usage
#### Learning shape space from the D-Faust dataset raw scans

##### Data
The raw scans can be downloaded from http://dfaust.is.tue.mpg.de/downloads.
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
We have uploaded SALD trained network. To produce predictions on unseen test scans, run:
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
			
