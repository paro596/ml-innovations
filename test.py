"""Created by pmaji at 8/22/2020"""
import time
import nltk
from nltk.classify import apply_features
from naive_bayes.data_util import PROJECT_DIR, time_since, split_dataset, gender_features
from naive_bayes.classify import load_classifier

weights = PROJECT_DIR + "weights/naive_bayes_weights"
dataset = PROJECT_DIR + "data/new_name_gender_dataset.csv"
test_split = 0.25
_, _, testset = split_dataset(1 - test_split, 0, dataset, shuffle=False)


def test(testset=testset, weight_file=weights):
    """tests classifier on name->gender
    
    Args:
        train: % of examples to train with (e.g., 0.8)
    """
    start = time.time()
    classifier = load_classifier(weight_file)

    print("Testing Naive Bayes Classifer on %d examples" % (len(testset)))
    testset = apply_features(gender_features, testset, labeled=True)
    acc = nltk.classify.accuracy(classifier, testset)

    print("Testing accuracy is %.2f%% on %d examples (%s)" % (acc * 100, len(testset), time_since(start)))
    return acc
