"""Created by pmaji at 8/21/2020"""
import os
import time
from naive_bayes.data_util import time_since, split_dataset, PROJECT_DIR
from naive_bayes.train import train
from naive_bayes.test import test
from naive_bayes.classify import predictor

DATASET = PROJECT_DIR + "data/new_name_gender_dataset.csv"
WEIGHTS = PROJECT_DIR + "weights/naive_bayes_weights"


def main(name):
    if not os.path.isfile(WEIGHTS):
        # # split the dataset into 70% train, 0%  val, 30% test
        TRAINSET, VALSET, TESTSET = split_dataset(0.7, 0, shuffle=True)
        print("Length of train data : ", len(TRAINSET))
        print("Length of test data : ", len(TESTSET))
        train(TRAINSET, VALSET, WEIGHTS)
        test(TESTSET, WEIGHTS)
    # predict gender from a first name
    predicted_gender = predictor(name, WEIGHTS)

    return predicted_gender


if __name__ == "__main__":
    start = time.time()
    names = ['Paramita', 'Tanvi', 'Swaroop', 'Pannaga', 'Shivam', 'Marina', 'Rabia', 'Mckyala', 'Michael', 'Elisha']
    for name in names:
        gender = main(name)
        print("%s -> %s " % (name, gender))
    print("\nClassified %d names (%s)" % (len(names), time_since(start)))
