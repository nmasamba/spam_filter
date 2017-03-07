"""

Authors: Nyasha Masamba and Mariam Basajja
Unit: Introduction to Machine Learning (December 2016)

"""


from __future__ import print_function, division
import os, random, re
import nltk
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from nltk import word_tokenize, WordNetLemmatizer, classify, corpus
import sys
import pickle
import os.path
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from nltk.metrics.scores import f_measure, precision, recall
from nltk.metrics import ConfusionMatrix
from nltk.collocations import *

stoplist = stopwords.words('english')

# 1. TO DOWNLOAD NLTK CORPORA:
# python -m nltk.downloader all
# sudo python -m nltk.downloader -d /usr/local/share/nltk_data all (for Linux)
# interactively: >>> import nltk 
#                >>> nltk.download('all')

# 2. TO RUN
# python filter.py <testfile.txt>


def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r')
        a_list.append(f.read())
    f.close()
    return a_list

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in word_tokenize(unicode(sentence, errors='ignore'))]

def get_features(text, setting):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(preprocess(text))
    trigram_features = finder.nbest(trigram_measures.raw_freq, 2000)
    return {word: True for word in trigram_features if not word in stoplist}

"""
def printmeasures(label, refset, testset):
    print (label, 'precision:', precision(refset, testset))
    print (label, 'recall:', recall(refset, testset))
    print (label, 'F-measure:', f_measure(refset, testset))
"""

def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # initialise the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]

    # initiate classification pipeline  
    pipeline = Pipeline([('chi2', SelectKBest(chi2, k=1000)),
                        ('svm', LinearSVC())])
    
    # prepare to do cross val   
    cv = cross_validation.KFold(len(train_set), n_folds=10, shuffle=True, random_state=None)

    # train an instance of the SVM classifier and perform 10-fold cross validation
    # print out metrics 
    acc_per_svm_fold = []
    reflist = []
    testlist = []
    for traincv, testcv in cv:
        classifier = nltk.classify.SklearnClassifier(pipeline)
        classifier.train(train_set[traincv[0]:traincv[len(traincv)-1]])
        for (email, label) in test_set:
            reflist.append(label)
            testlist.append(classifier.classify(email))
        acc_per_svm_fold.append( nltk.classify.util.accuracy(classifier, train_set[testcv[0]:testcv[len(testcv)-1]]))
    """
    cm = ConfusionMatrix(reflist, testlist)
    print (cm)
    print ("Accuracy per SVM fold:", acc_per_svm_fold)
    print ("Average SVM accuracy:", sum(acc_per_svm_fold) / len(acc_per_svm_fold))
    """
    refham = set()
    refspam = set()
    testham = set()
    testspam = set()
    for i, label in enumerate(reflist):
        if label == 'ham': refham.add(i)
        if label == 'spam': refspam.add(i)
    for i, label in enumerate(testlist):
        if label == 'ham': testham.add(i)
        if label == 'spam': testspam.add(i)
    """
    printmeasures('spam', refspam, testspam)
    printmeasures('ham', refham, testham)
    """
    return train_set, test_set, classifier
 

def run_online(classifier, setting):
    with open(sys.argv[1], 'r') as f2:
        features = get_features(f2.read(), setting)
    print (classifier.classify(features))

 
if __name__ == "__main__":
    if (os.path.exists('finalised_clf_svm')):
        #load pre-trained classifier
        classifier = pickle.load(open('finalised_clf_svm', 'rb'))
        #classify your new email
        run_online(classifier, "")
    else:
        spam = init_lists('public/spam/')
        ham = init_lists('public/ham/')
        all_emails = [(email, 'spam') for email in spam]
        all_emails += [(email, 'ham') for email in ham]
        random.shuffle(all_emails)
        
     
        all_features = [(get_features(email, ''), label) for (email, label) in all_emails]
        train_set, test_set, classifier = train(all_features, 0.9)
     
        #store trained classifier results
        pickle.dump(classifier, open('finalised_clf_svm', 'wb'))
        #classify your new email
        run_online(classifier, "")
         
    
 
