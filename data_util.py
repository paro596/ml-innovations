"""Created by pmaji at 8/22/2020"""
import random
import unicodedata
import csv
import string
import time
from os import getcwd

PROJECT_DIR = getcwd().replace("naive_bayes", "")
DATASET_FN = PROJECT_DIR + "data/new_name_gender_dataset.csv"
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.0


# helpers functions
def clean_str(s):
    uncoded = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in string.ascii_letters
    )
    return uncoded.lower()


def time_since(since):
    now = time.time()
    s = now - since
    hours, rem = divmod(now - since, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}h {:0>2}m {:0>2}s".format(int(hours), int(minutes), int(seconds))


def gender_features(name):
    name = name.lower()
    features = {"first2": name[:2],
                "suffix3": name[-3:],
                "suffix4": name[-4:],
                "length_of_name": len(name)
                }
    return features


# data accessors
def load_names(filename=DATASET_FN):
    """loads all names and genders from the dataset
    :param filename: path to dataset file (default: DATASET_FN)
    :return: names and genders read from the data fil
    """
    names = []
    genders = []

    with open(filename) as csv_data_file:
        csv_reader = csv.reader(csv_data_file)
        for row in csv_reader:
            names.append(row[0])
            genders.append(row[1])

    return names, genders


def load_dataset(shuffled, filename=DATASET_FN):
    """Returns the name->gender dataset ready for processing
    :param filename: path to dataset file (default: DATASET_FN)
    :param shuffled (Boolean, optional): set to False to return the dataset unshuffled
    :return: namelist (list(String,String)): list of (name, gender) records
    """
    names, genders = load_names(filename)
    namelist = list(zip(names, genders))
    if shuffled:
        random.shuffle(namelist)
    return namelist


def split_dataset(train_pct=TRAIN_SPLIT, val_pct=VAL_SPLIT, filename=DATASET_FN, shuffle=False):
    """

    :param train_pct: train dataset ration
    :param val_pct: validation dataset ratio
    :param filename: data dile name
    :param shuffle: random shuffLe status

    Returns: splited data set

    """
    dataset = load_dataset(shuffle, filename)
    n = len(dataset)
    tr = int(n * train_pct)
    va = int(tr + n * val_pct)
    return dataset[:tr], dataset[tr:va], dataset[va:]  # Trainset, Valset, Testset
