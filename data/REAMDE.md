# DATA

In this folder, you will find the selected data and the undersampling performed for the project. The initial dataset contains several thousand images across 350 labels, which is quite substantial given the resources available to us.

## Contents of [/data](.)

| Folder/Files       | Description                                                                  |
|--------------------|------------------------------------------------------------------------------|
| img/               | Folder containing the constructed images, separated by set and label.        |
| size_10000/        | Initial JSON file with the coordinates of the images and their labels.       |
| __init__.py        | File necessary for Python to treat the directory as a module.                |
| dataset_builder.py | Script used to create the img/ folder.                                       |
| labels.txt         | The selected labels.                                                         |
| README.md          | File explaining the contents of the folder.                                  |

## Usage Guide for `dataset_builder.py`

This file is used to create our training, testing, and validation datasets.

To display the available options:

```bash
# Assuming you are at the root of the project
$ python data/dataset_builder.py -h
```
```
Create the QuickDraw dataset from the json files

positional arguments:
  input_path            The folder containing the json files
  output_path           The folder where the images will be saved

options:
  -h, --help            show this help message and exit
  --max_images MAX_IMAGES
                        The number of images to save per json file
  --labels LABELS [LABELS ...]
                        The labels to consider
  --delete_on_error     Delete the output folder if an error occurs
  --test_size TEST_SIZE
                        The proportion of the dataset to include in the test split
  --val_size VAL_SIZE
                        The proportion of the dataset to include in the validation split
```

To construct your dataset from the size_10000 folder to the data folder, taking, for example, 30% for test and 30% for validation:

```bash
python data/dataset_builder.py /<PATH_TO_PROJECT>/data/size_10000 /<PATH_TO_PROJECT>/data --max_images 250 --labels /<PATH_TO_PROJECT>/data/labels.txt --test_size 0.3 --val_size 0.3
