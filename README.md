# Digit-Recognizer

Purpose:
The goal of this project is to classify handwritten single digit (0-9). We use Spark and Azure to increase the running speed.

Solution:
Data
We get the data from Kaggle competition, Digit Recognizer. The data set contains gray-scale images of hand-drawn digits, from zero through nine. There are 41985 labeled records in the data set. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive. There is no missing value.

Preprocessing
Since there are 784 pixels, ie. attributes in total, we intend to apply SVD or PCA method to reduce dimensions prior to building models.

Model
Random Forest, KNN, SVM, Naive Bayes, Neural Network (Convolutional Neural Nets)

Evaluation
ROC, Confusion Matrix,  Accuracy Score. 
