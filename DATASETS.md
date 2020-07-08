# Dataset preparation

### Configuration
Please create a config file called `semantic_segmentation.cfg` that specifies the paths
to the datasets. Note that they are optional; you don't need to download and provide all datasets,
only the ones you intend to use. Replace the paths below with something that works for you:

```
[paths]
camvid=/datasets/camvid/CamVidData.zip
cityscapes=/datasets/cityscapes\cityscapes_segmentation.zip
isic2017=/datasets/isic2017/isic2017_segmentation_248x248.zip
pascal_voc=/datasets/pascal_voc2012/VOCdevkit/VOC2012
```

Note that the CamVid, Cityscapes and ISIC 2017 datasets must be converted to a ZIP-based format
prior to use.
You must run the provided conversion utilities to create these ZIP files.


#### Pascal VOC 2012

1. Download the [Pascal VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)
(use the 'training/validation data' link).
2. You will also want the augmented labels
([download here](http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip))
so you can use the augmented Pascal dataset (used in [Mittal et al.](https://arxiv.org/abs/1908.05724) and
 [Hung et al.](https://arxiv.org/abs/1802.07934))
3. Decompress the main main dataset file `VOCtrainval_11-May-2012.tar`
4. Unzip `SegmentationClassAug.zip` within the `VOCdevkit/VOC2012` directory that was created by
unpacking the main dataset.
5. Edit the `semantic_segmentation.cfg` configuration file and provide a path for the `pascal_voc` setting.
6. Now run: `python download_pascal_aug_names.py` to download some index files

The specific split used in [Mittal et al.](https://arxiv.org/abs/1908.05724) can be found
in `data/splits/pascal_aug/split_0.pkl`. This file was taken
[as-is](https://github.com/sud0301/semisup-semseg/tree/master/splits/voc)
from their [repo](https://github.com/sud0301/semisup-semseg).
 

#### Cityscapes

1. Sign up for a cityscapes account at [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)
2. Download the input images file `leftImg8bit_trainvaltest.zip`
3. Download the ground truth file `gtFine_trainvaltest.zip`.
4. Edit the `semantic_segmentation.cfg` configuration file and point the `cityscapes` to a place where you
 want the converted Cityscapes ZIP to live 
5. Run: `python convert_cityscapes.py /path/to/leftImg8bit_trainvaltest.zip /path/to/gtFine_trainvaltest.zip`

The conversion process will downsample all images by a factor of 2 as in [Mittal et al.](https://arxiv.org/abs/1908.05724) and
 [Hung et al.](https://arxiv.org/abs/1802.07934).


#### ISIC 2017

1. Download the ISIC 2017 zip files: `ISIC-2017_Training_Data.zip`, `ISIC-2017_Training_Part1_GroundTruth.zip`,
`ISIC-2017_Validation_Data.zip` and `ISIC-2017_Validation_Part1_GroundTruth.zip` to a directory called
e.g. `/path/to/isic_zips_directory`.
2. Run: `python convert_isic.py /path/to/isic_zips_directory`

The conversion process will scale all images to a default size of `248x248`.
(Use the `--out_size=<height>,<width>` when running `convert_isic` to change this).
