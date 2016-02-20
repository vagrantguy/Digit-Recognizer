# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 00:01:12 2015

@author: ruoyang
"""
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
import time
import os

os.environ["SPARK_HOME"] = "/usr/local/spark/spark-1.5.2-bin-hadoop2.6"

#sc = SparkContext()

def parseLine(raw_data):
    
    line = raw_data.split(',')
    label = line[0]
    features = line[1:]
    #features = Vectors.dense([float(x) for x in parts[1].split(',')])

    return LabeledPoint(label, features)

time_start = time.time()
#data = sc.textFile('/usr/local/spark/spark-1.5.2-bin-hadoop2.6/data/mllib/sample_naive_bayes_data.txt').map(parseLine)
data = sc.textFile("train.csv")
first_raw = data.first()
data = data.filter(lambda x:x !=first_raw).map(parseLine)

# Split data aproximately into training (70%) and test (30%)
training, test = data.randomSplit([0.7, 0.3], seed = 0)

# Train a naive Bayes model.
model = NaiveBayes.train(training, 1.0)
#model = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},
#                                     numTrees=100, featureSubsetStrategy="auto",
#                                     impurity='gini', maxDepth=4, maxBins=32)
#model = DecisionTree.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},
#                                     impurity='gini', maxDepth=5, maxBins=32)

# Make prediction and test accuracy.
predictionAndLabel = test.map(lambda p : (model.predict(p.features), p.label))
predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
accuracy_1 = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test.count())

#accuracy_2 = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
time_end = time.time()
print("Training accuracy_1 = " + str(accuracy_1))
#print('Training accuracy_2 = ' + str(accuracy_2))
print("Training time = %.03f seconds" %(time_end - time_start))

