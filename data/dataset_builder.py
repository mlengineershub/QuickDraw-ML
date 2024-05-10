from PIL import Image, ImageDraw
from typing import List, Dict, Optional
from tqdm.auto import tqdm
import os
import json
import argparse


def data_to_img(o: Dict) -> Image:
    """
    Return an image from a drawing object
    (usually object["drawing in the point list"])

    @param o {dict} the object from the dataset
    @return {Image} a PIL Image of size (3, 256, 256)
    """

    cord = o["drawing"]

    img = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for i in range(len(cord)):
        line = cord[i]

        if len(line[0]) > 1:
            for j in range(len(line[0]) - 1):
                draw.line((line[0][j], line[1][j], line[0][j+1], line[1][j+1]),
                          fill=(0, 0, 0), width=3)
        else:
            draw.point((line[0][0], line[1][1]), fill=(0, 0, 0), width=3)

    return img


def json_to_img(json_folder: str,
                output_folder: str,
                max_img: int = 250,
                labels: Optional[List[str]] = None,
                delete_on_error: bool = False,
                test_size: float = 0.2,
                val_size: float = 0.2) -> None:
    """
    Create images from json files and split into train, test, and validation
    sets without using sklearn.

    @param json_folder {str} - The folder containing the json files.
    @param output_folder {str} - The folder where the images will be saved.
    @param max_img {int} - The number of images to save per json file.
    @param labels {Optional[List[str]]} - The labels to consider,
                                          None for all labels.
    @param delete_on_error {bool} - Delete the output folder if an error
                                    occurs.
    @param test_size {float} - The proportion of the dataset to include in the
                               test split.
    @param val_size {float} - The proportion of the dataset to include in the
                              validation split.
    """

    if not os.path.exists(json_folder):
        raise FileNotFoundError(f"{json_folder} not found")

    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    try:
        for f in tqdm(files, desc="Processing files"):
            label = f.split('.')[0]
            if labels is not None and label not in labels:
                continue

            with open(f"{json_folder}/{f}", "r") as file:
                data = json.load(file)

            num_images = min(max_img, len(data))
            num_test = int(num_images * test_size)
            num_val = int(num_images * val_size)
            num_train = num_images - num_test - num_val

            for i, d in enumerate(data):
                if not d["recognized"]:
                    continue

                if i < num_train:
                    set_name = "train"
                elif i < num_train + num_test:
                    set_name = "test"
                else:
                    set_name = "val"

                set_output_folder = f"{output_folder}/img/{set_name}/{label}"
                os.makedirs(set_output_folder, exist_ok=True)

                img = data_to_img(d)
                img.save(f"{set_output_folder}/{i}.png")

                if i >= num_images - 1:
                    break

    except Exception as e:

        if delete_on_error:
            os.system(f"rm -rf {output_folder}")
        raise e


if __name__ == "__main__":

    # Examples
    # python data/dataset_builder.py -h
    # python data/dataset_builder.py data/size_10000 data --max_images 250
    #                                                 --labels data/labels.txt

    parser = argparse.ArgumentParser(
        description="Create the QuickDraw dataset from the json files"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="The folder containing the json files")
    parser.add_argument(
        "output_path",
        type=str,
        help="The folder where the images will be saved")
    parser.add_argument(
        "--max_images",
        type=int,
        default=250,
        help="The number of images to save per json file"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="The labels to consider"
    )
    parser.add_argument(
        "--delete_on_error",
        action="store_true",
        help="Delete the output folder if an error occurs"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="The proportion of the dataset to include in the test split"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="The proportion of the dataset to include in the validation split"
    )

    args = parser.parse_args()

    if args.labels is not None:
        if len(args.labels) == 1 and args.labels[0].endswith(".txt"):
            with open(args.labels[0], "r") as file:
                args.labels = file.read().split("\n")

    if args.test_size + args.val_size >= 1:
        raise ValueError("test_size + val_size should be less than 1")

    if args.test_size < 0 or args.val_size < 0:
        raise ValueError("test_size and val_size should be positive")

    json_to_img(args.input_path, args.output_path, args.max_images,
                args.labels, args.delete_on_error, args.test_size,
                args.val_size)
