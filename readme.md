# Number Recognition
In this project I tied to train model for number recognition in image with height 28, as in MNIST dataset (width can be larger then 28).
Was used MNIST and SVHN dataset becouse MNIST have poor accaracy for computer font and SVHN has such images for trainin. Also artificial images with digits used for augmentation.
### Folders and files
*data with datasets for training models
*imges with images for testing pre-trained models
*models with pre-trained models
*notebooks with jupyter notebook for training model(with all outputs)
*scripts with .py files for training models
*mean_std.py file with mean and std for pre-trained models
*testModels.py and testModels.ipynd with testing pre-trained models(with visualization) on images from "images/"
*numberRecognition.py script for using recognizer
### Models
In models folder exists such models:
-aug_model trained on artificiol data only (lowest accuracy)
-MNIST_model trained only on MNIST dataset (not good on real images as I want)
-MNIST_aug_model' trained om MNIST and artificial data (same result)
-SVHM_model trained on SVHN dataset onle (better then MNIST)
-SVHM_aug_model' trained om SVHN and artificial data (unexpect but wors then pure SVHN)
-MNIST_SVHM_model trained ON MNIST and SVHN datasets (bad as previous)
-MNIST_SVHM_aug_model best one traind on all data (use in final script as default)
### Neural network
Was used LeNet architecture for network. In model exist such layers:
*Convolution x2
*Batch Normalization x2
*Relu Activation x2
*Max Pooling x2
*Linear with ReLu x3
### Usege
For usege script for number recognition in derectory shuold exists such folders and files:
*models at leat with MNIST_SVHM_aug_model.pth file
*mean_std.py
*numberRecognizer.py
Examples:
+```python
python numberRecognizer.py image_name
```
where image_name is image Path like images/04770.png
+```python
python numberRecognizer.py image_name model_path mean std
```
where model_name is model Path like models/aug_model.pth, mean and std numbers (can be 0.0 and 0.0)