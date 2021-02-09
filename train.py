"""Created by pmaji at 8/21/2020"""
import time
import nltk
from nltk.classify import apply_features
import pickle
from naive_bayes.data_util import PROJECT_DIR, time_since, split_dataset, gender_features

weights = PROJECT_DIR + "weights/naive_bayes_weights"
dataset = PROJECT_DIR + "data/new_name_gender_dataset.csv"
train_split = 0.75  # default value
val_split = 0.05  # default value
trainset, valset, _ = split_dataset(train_split, val_split, dataset, shuffle=False)


def train(trainset=trainset, valset=valset, weight_file=weights):
    """trains classifier on name->gender
    
    Args:
        trainset: list of name->gender tuple pairs for training
        valset (opt): list of name->gender tuple pairs to validation
        weight_file: filename to save classifer weights

    """

    start = time.time()
    print("Training Naive Bayes Classifer on %d examples" % (len(trainset)))
    trainset = apply_features(gender_features, trainset, labeled=True)
    classifier = nltk.NaiveBayesClassifier.train(trainset)
    # classifier.show_most_informative_features(10)

    print("Training complete. (%s)" % (time_since(start)))

    # validation
    if valset is not None and len(valset) > 0:
        valset = apply_features(gender_features, valset, labeled=True)
        acc = nltk.classify.accuracy(classifier, valset)
        print("Validation accuracy is %.2f%% on %d examples (%s)" % (acc * 100, len(valset), time_since(start)))

    # save weights
    with open(weight_file, 'wb') as f:
        pickle.dump(classifier, f)
        f.close()
        print('Weights saved to "%s"' % weights)
