# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:16:00 2015

@author: Ryding
"""

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
import time

#sc = SparkContext()

def parseLine(raw_data):
    
    line = raw_data.split(',')
    label = line[0]
    features = line[1:]

    return LabeledPoint(label, features)

time_start = time.time()
data = sc.textFile("train.csv")
first_raw = data.first()
data = data.filter(lambda x:x !=first_raw).map(parseLine)

# Split data aproximately into training (70%) and test (30%)
training, test = data.randomSplit([0.7, 0.3], seed = 0)

# Train a naive Bayes model.
model = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=20, featureSubsetStrategy="auto",
                                     impurity='entropy', maxDepth=15)

# Make prediction and test accuracy.
predictionAndLabel = test.map(lambda p : (model.predict(p.features), p.label))
predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
accuracy = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(test.count())

time_end = time.time()
print("Training accuracy = " + str(accuracy))
print("Training time = %.03f seconds" %(time_end - time_start))