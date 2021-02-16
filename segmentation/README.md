# Goodgame Dataset

The repository contains:

* Procedures to use Goodgame Dataset with two tasks: Object detection and semantic segmentation.
* How to implement into your custom dataset.

### Download datasets

* [Object detection](https://drive.google.com/file/d/1NGrKWHc1z_4bOh2huWHC8kZsUZFXOku-/view)
* [Semantic segmentation](https://drive.google.com/file/d/1X-onXnGbrIwuXTt03rK-6FV3w2bGNyK8/view?usp=sharing)

### Documentation

#### Object detection
Object detection folder contains: dataset folder, two files .csv (test.csv, train.csv). The train.csv has 12,764 images and the test.csv has 2561 images. 

```
Object Detection 
    │─── Data
         |───000000_10.png
         |───000001_10.png
         |─── ...
    │─── test.csv
    │─── train.csv
```

The structure of .csv: 

| filename | xmin | ymin | xmax | ymax | class_id |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 00072.jpg     | 148     | 53     | 159     | 63     | 4     |

In object detection task, we have 6 traffic signs (6 classes): Turn left, Turn Right, Straight, Stop, No Turn Left, No Turn Right. 

![](https://i.imgur.com/jrmCOEW.png)

#### Semantic segmentation

Setup sementation data folders

```
Segmentation
    │─── GGDataSet
         |─── train_frames
             |─── train
                 |─── train_000001.png
         |─── train_masks
             |─── train
                 |─── train_000001.png         
         |─── val_frames
             |─── val
                 |─── val_000001.png         
         |─── val_masks
             |─── val
                 |─── val_000001.png         
         |─── label_colors.txt
    │─── model_pb
    │─── models
    │─── train.py
    │─── convert_pb.py
```

Goodgame segmentation dataset has 6,240 training images, 1,448 validation images. In our task, we have to predict 3 classes: Background, Line, Road. 
The label_colors.txt contains RGB color code of classes and we handle it to convert classes into one-hot vector. 

![](https://i.imgur.com/XbJsBE0.png)


### How to use Goodgame Dataset

#### Object detection

In Goodgame experiment, we implement Single Shot Detection (SSD). You do not need to follow our instructions if you want to handle the data for only your purposes.

1. Upload **Goodgame Dataset** to Google Drive
2. Upload **object_detection.ipynb** to Google Colab 
3. Modify your dataset locate path in Google Drive and your dataset path link to images and .csv files.

You can run notebook in local with requirements: Keras version 2.2.4, Tensorflow version 1.15 and git clone this repository: 

``` git clone https://github.com/pierluigiferrari/ssd_keras ```

#### Semantic segmentation

In Goodgame experiment, we implement PSPNet and use combine-loss is Dice Loss and Focal Loss. We use a [library](https://github.com/qubvel/segmentation_models) segmentation, you can read the docs to modify segmentation architecture. 

To train segmentation task by Goodgame dataset, run the following commands

```python=1
# Keras 2.2.4, Tensorflow >= 1.15

pip install -U segmentation-models

# Modify dataset path in train.py

img_dir = 'your_path/GGDataSet/'

DATA_PATH = 'your_path/GGDataSet/'

# Train

python3 train.py

# If you want to run pretrained model faster, you need to convert model to frozen graph 

python3 convert_pb.py

```

### How to implement into your custom dataset

In object detection task, we use the [labelimg](https://github.com/tzutalin/labelImg) tool and the [labelme](https://github.com/wkentaro/labelme) tool to label segmentation dataset.

To object detection, we need to convert .xml files to .csv as (train.csv above). This [xml_to_csv.py](https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py) to help you handle it, remember that in our .csv we only contains (filename, xmin,ymin,xmax,ymax,class_id).

To semantic segmentation, the labelme tool export .json, we need to convert .json files to .png. Run the following commands

```python=1
git clone https://github.com/wkentaro/labelme.git

cd labelme/examples/semantic_segmentation

./labelme2voc.py data_annotated data_dataset_voc --labels labels.txt
```

You can see the label PNG file in `data_dataset_voc/SegmentationClassPNG/` folder.
Modify `data_annotated` folder to `your_dataset`. 
