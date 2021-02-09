"""Created by pmaji at 8/22/2020"""
import csv
import time
import pickle
from naive_bayes.data_util import PROJECT_DIR, clean_str, time_since, gender_features

weights = PROJECT_DIR + "weights/naive_bayes_weights"
verbose = True


def load_classifier(weight_file=weights, verbose=False):
    with open(weight_file, 'rb') as f:
        classifier = pickle.load(f)
        f.close()
    if verbose: print('Loaded weights from "%s"...\n' % (weight_file))
    return classifier


def _classify(name, classifier):
    _name = gender_features(clean_str(name))
    dist = classifier.prob_classify(_name)
    m, f = dist.prob("male"), dist.prob("female")
    d = {m: "male", f: "female"}
    prob = max(m, f)
    guess = d[prob]
    return guess


def predictor(name, weight_file=weights):
    classifier = load_classifier(weight_file)
    predicted_gender = _classify(name, classifier)
    return predicted_gender
