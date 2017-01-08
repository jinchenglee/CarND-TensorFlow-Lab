
# coding: utf-8

# <h1 align="center">TensorFlow Neural Network Lab</h1>

# <img src="image/notmnist.png">
# In this lab, you'll use all the tools you learned from *Introduction to TensorFlow* to label images of English letters! The data you are using, <a href="http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html">notMNIST</a>, consists of images of a letter from A to J in differents font.
# 
# The above images are a few examples of the data you'll be training on. After training the network, you will compare your prediction model against test data. Your goal, by the end of this lab, is to make predictions against that test set with at least an 80% accuracy. Let's jump in!

# To start this lab, you first need to import all the necessary modules. Run the code below. If it runs successfully, it will print "`All modules imported`".

# In[1]:

import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile

print('All modules imported.')


#<<JC>> # The notMNIST dataset is too large for many computers to handle.  It contains 500,000 images for just training.  You'll be using a subset of this data, 15,000 images for each label (A-J).
#<<JC>> 
#<<JC>> # In[2]:
#<<JC>> 
#<<JC>> def download(url, file):
#<<JC>>     """
#<<JC>>     Download file from <url>
#<<JC>>     :param url: URL to file
#<<JC>>     :param file: Local file path
#<<JC>>     """
#<<JC>>     if not os.path.isfile(file):
#<<JC>>         print('Downloading ' + file + '...')
#<<JC>>         urlretrieve(url, file)
#<<JC>>         print('Download Finished')
#<<JC>> 
#<<JC>> # Download the training and test dataset.
#<<JC>> download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
#<<JC>> download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')
#<<JC>> 
#<<JC>> # Make sure the files aren't corrupted
#<<JC>> assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa',        'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
#<<JC>> assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9',        'notMNIST_test.zip file is corrupted.  Remove the file and try again.'
#<<JC>> 
#<<JC>> # Wait until you see that all files have been downloaded.
#<<JC>> print('All files downloaded.')
#<<JC>> 
#<<JC>> 
#<<JC>> # In[3]:
#<<JC>> 
#<<JC>> def uncompress_features_labels(file):
#<<JC>>     """
#<<JC>>     Uncompress features and labels from a zip file
#<<JC>>     :param file: The zip file to extract the data from
#<<JC>>     """
#<<JC>>     features = []
#<<JC>>     labels = []
#<<JC>> 
#<<JC>>     with ZipFile(file) as zipf:
#<<JC>>         # Progress Bar
#<<JC>>         filenames_pbar = tqdm(zipf.namelist(), unit='files')
#<<JC>>         
#<<JC>>         # Get features and labels from all files
#<<JC>>         for filename in filenames_pbar:
#<<JC>>             # Check if the file is a directory
#<<JC>>             if not filename.endswith('/'):
#<<JC>>                 with zipf.open(filename) as image_file:
#<<JC>>                     image = Image.open(image_file)
#<<JC>>                     image.load()
#<<JC>>                     # Load image data as 1 dimensional array
#<<JC>>                     # We're using float32 to save on memory space
#<<JC>>                     feature = np.array(image, dtype=np.float32).flatten()
#<<JC>> 
#<<JC>>                 # Get the the letter from the filename.  This is the letter of the image.
#<<JC>>                 label = os.path.split(filename)[1][0]
#<<JC>> 
#<<JC>>                 features.append(feature)
#<<JC>>                 labels.append(label)
#<<JC>>     return np.array(features), np.array(labels)
#<<JC>> 
#<<JC>> # Get the features and labels from the zip files
#<<JC>> train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
#<<JC>> test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')
#<<JC>> 
#<<JC>> # Limit the amount of data to work with a docker container
#<<JC>> docker_size_limit = 150000
#<<JC>> train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)
#<<JC>> 
#<<JC>> # Set flags for feature engineering.  This will prevent you from skipping an important step.
#<<JC>> is_features_normal = False
#<<JC>> is_labels_encod = False
#<<JC>> 
#<<JC>> # Wait until you see that all features and labels have been uncompressed.
#<<JC>> print('All features and labels uncompressed.')
#<<JC>> 
#<<JC>> 
#<<JC>> # <img src="image/mean_variance.png" style="height: 75%;width: 75%; position: relative; right: 5%">
#<<JC>> # ## Problem 1
#<<JC>> # The first problem involves normalizing the features for your training and test data.
#<<JC>> # 
#<<JC>> # Implement Min-Max scaling in the `normalize()` function to a range of `a=0.1` and `b=0.9`. After scaling, the values of the pixels in the input data should range from 0.1 to 0.9.
#<<JC>> # 
#<<JC>> # Since the raw notMNIST image data is in [grayscale](https://en.wikipedia.org/wiki/Grayscale), the current values range from a min of 0 to a max of 255.
#<<JC>> # 
#<<JC>> # Min-Max Scaling:
#<<JC>> # $
#<<JC>> # X'=a+{\frac {\left(X-X_{\min }\right)\left(b-a\right)}{X_{\max }-X_{\min }}}
#<<JC>> # $
#<<JC>> # 
#<<JC>> # *If you're having trouble solving problem 1, you can view the solution [here](https://github.com/udacity/CarND-TensorFlow-Lab/blob/master/solutions.ipynb).*
#<<JC>> 
#<<JC>> # In[4]:
#<<JC>> 
#<<JC>> # Problem 1 - Implement Min-Max scaling for grayscale image data
#<<JC>> def normalize_grayscale(image_data):
#<<JC>>     """
#<<JC>>     Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
#<<JC>>     :param image_data: The image data to be normalized
#<<JC>>     :return: Normalized image data
#<<JC>>     """
#<<JC>>     # TODO: Implement Min-Max scaling for grayscale image data
#<<JC>>     img_max = np.max(image_data)
#<<JC>>     img_min = np.min(image_data)
#<<JC>>     a = 0.1
#<<JC>>     b = 0.9
#<<JC>>     
#<<JC>>     img_normed = a + (b-a)*(image_data - img_min)/(img_max - img_min)
#<<JC>>     #print(img_normed)
#<<JC>>     return img_normed
#<<JC>> 
#<<JC>> 
#<<JC>> ### DON'T MODIFY ANYTHING BELOW ###
#<<JC>> # Test Cases
#<<JC>> np.testing.assert_array_almost_equal(
#<<JC>>     normalize_grayscale(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255])),
#<<JC>>     [0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
#<<JC>>      0.125098039216, 0.128235294118, 0.13137254902, 0.9],
#<<JC>>     decimal=3)
#<<JC>> np.testing.assert_array_almost_equal(
#<<JC>>     normalize_grayscale(np.array([0, 1, 10, 20, 30, 40, 233, 244, 254,255])),
#<<JC>>     [0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
#<<JC>>      0.896862745098, 0.9])
#<<JC>> 
#<<JC>> if not is_features_normal:
#<<JC>>     train_features = normalize_grayscale(train_features)
#<<JC>>     test_features = normalize_grayscale(test_features)
#<<JC>>     is_features_normal = True
#<<JC>> 
#<<JC>> print('Tests Passed!')
#<<JC>> 
#<<JC>> 
#<<JC>> # In[5]:
#<<JC>> 
#<<JC>> if not is_labels_encod:
#<<JC>>     # Turn labels into numbers and apply One-Hot Encoding
#<<JC>>     encoder = LabelBinarizer()
#<<JC>>     encoder.fit(train_labels)
#<<JC>>     train_labels = encoder.transform(train_labels)
#<<JC>>     test_labels = encoder.transform(test_labels)
#<<JC>> 
#<<JC>>     # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
#<<JC>>     train_labels = train_labels.astype(np.float32)
#<<JC>>     test_labels = test_labels.astype(np.float32)
#<<JC>>     is_labels_encod = True
#<<JC>> 
#<<JC>> print('Labels One-Hot Encoded')
#<<JC>> 
#<<JC>> 
#<<JC>> # In[6]:
#<<JC>> 
#<<JC>> assert is_features_normal, 'You skipped the step to normalize the features'
#<<JC>> assert is_labels_encod, 'You skipped the step to One-Hot Encode the labels'
#<<JC>> 
#<<JC>> # Get randomized datasets for training and validation
#<<JC>> train_features, valid_features, train_labels, valid_labels = train_test_split(
#<<JC>>     train_features,
#<<JC>>     train_labels,
#<<JC>>     test_size=0.05,
#<<JC>>     random_state=832289)
#<<JC>> 
#<<JC>> print('Training features and labels randomized and split.')
#<<JC>> 
#<<JC>> 
#<<JC>> # In[7]:
#<<JC>> 
#<<JC>> # Save the data for easy access
#<<JC>> pickle_file = 'notMNIST.pickle'
#<<JC>> if not os.path.isfile(pickle_file):
#<<JC>>     print('Saving data to pickle file...')
#<<JC>>     try:
#<<JC>>         with open('notMNIST.pickle', 'wb') as pfile:
#<<JC>>             pickle.dump(
#<<JC>>                 {
#<<JC>>                     'train_dataset': train_features,
#<<JC>>                     'train_labels': train_labels,
#<<JC>>                     'valid_dataset': valid_features,
#<<JC>>                     'valid_labels': valid_labels,
#<<JC>>                     'test_dataset': test_features,
#<<JC>>                     'test_labels': test_labels,
#<<JC>>                 },
#<<JC>>                 pfile, pickle.HIGHEST_PROTOCOL)
#<<JC>>     except Exception as e:
#<<JC>>         print('Unable to save data to', pickle_file, ':', e)
#<<JC>>         raise
#<<JC>> 
#<<JC>> print('Data cached in pickle file.')


# # Checkpoint
# All your progress is now saved to the pickle file.  If you need to leave and comeback to this lab, you no longer have to start from the beginning.  Just run the code block below and it will load all the data and modules required to proceed.

# In[8]:

#%matplotlib inline

# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Reload the data
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  print("train_features size: ", len(train_features))
  del pickle_data  # Free up memory


print('Data and modules loaded.')


# <img src="image/weight_biases.png" style="height: 60%;width: 60%; position: relative; right: 10%">
# ## Problem 2
# For the neural network to train on your data, you need the following <a href="https://www.tensorflow.org/resources/dims_types.html#data-types">float32</a> tensors:
#  - `features`
#   - Placeholder tensor for feature data (`train_features`/`valid_features`/`test_features`)
#  - `labels`
#   - Placeholder tensor for label data (`train_labels`/`valid_labels`/`test_labels`)
#  - `weights`
#   - Variable Tensor with random numbers from a truncated normal distribution.
#     - See <a href="https://www.tensorflow.org/api_docs/python/constant_op.html#truncated_normal">`tf.truncated_normal()` documentation</a> for help.
#  - `biases`
#   - Variable Tensor with all zeros.
#     - See <a href="https://www.tensorflow.org/api_docs/python/constant_op.html#zeros"> `tf.zeros()` documentation</a> for help.
# 
# *If you're having trouble solving problem 2, review "TensorFlow Linear Function" section of the class.  If that doesn't help, the solution for this problem is available [here](https://github.com/udacity/CarND-TensorFlow-Lab/blob/master/solutions.ipynb).*

# In[9]:

features_count = 784
labels_count = 10

hidden_layer_count = 1024

# TODO: Set the features and labels tensors
features = tf.placeholder(tf.float32, [None, features_count])
labels = tf.placeholder(tf.float32, [None, labels_count])

# TODO: Set the weights and biases tensors
weights_hidden_layer = tf.Variable(tf.truncated_normal([features_count, hidden_layer_count]))
biases_hidden_layer = tf.Variable(tf.zeros(hidden_layer_count))

weights = tf.Variable(tf.truncated_normal([hidden_layer_count, labels_count]))
biases = tf.Variable(tf.zeros(labels_count))

### DON'T MODIFY ANYTHING BELOW ###

#Test Cases
from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

assert features._shape == None or (    features._shape.dims[0].value is None and    features._shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
assert labels._shape  == None or (    labels._shape.dims[0].value is None and    labels._shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'
assert weights_hidden_layer._variable._shape == (784, 1024), 'The shape of weights is incorrect'
assert biases_hidden_layer._variable._shape == (1024), 'The shape of biases is incorrect'
assert weights._variable._shape == (1024, 10), 'The shape of weights is incorrect'
assert biases._variable._shape == (10), 'The shape of biases is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'

# Feed dicts for training, validation, and test session
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

# Linear Function WX + b
#tmp = tf.matmul(features, weights_hidden_layer) + biases_hidden_layer
tmp = tf.add(tf.matmul(features, weights_hidden_layer),biases_hidden_layer)
tmp_relu = tf.nn.relu(tmp)

# Hidden layer
#logits = tf.matmul(tmp_relu, weights) + biases
logits = tf.add(tf.matmul(tmp_relu, weights), biases)

prediction = tf.nn.softmax(logits)

#<<JC>> # Cross entropy
#<<JC>> cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)
#<<JC>> 
#<<JC>> # Training loss
#<<JC>> loss = tf.reduce_mean(cross_entropy)

loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)

# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

print('Accuracy function created.')



# In[11]:

# TODO: Find the best parameters for each configuration
epochs = 200
batch_size = 500
learning_rate = 0.03



### DON'T MODIFY ANYTHING BELOW ###
# Gradient Descent
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)    
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

# Create an operation that initializes all variables
init = tf.initialize_all_variables()

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()
plt.show()

print('Training accuracy at {}'.format(training_accuracy))
print('Validation accuracy at {}'.format(validation_accuracy))


# ## Test
# Set the epochs, batch_size, and learning_rate with the best learning parameters you discovered in problem 3.  You're going to test your model against your hold out dataset/testing data.  This will give you a good indicator of how well the model will do in the real world.  You should have a test accuracy of at least 80%.

# In[12]:

#<<JC>> # TODO: Set the epochs, batch_size, and learning_rate with the best parameters from problem 3
#<<JC>> epochs = 1
#<<JC>> batch_size = 100
#<<JC>> learning_rate = 0.1
#<<JC>> 
#<<JC>> 
#<<JC>> 
#<<JC>> ### DON'T MODIFY ANYTHING BELOW ###
#<<JC>> # The accuracy measured against the test set
#<<JC>> test_accuracy = 0.0
#<<JC>> 
#<<JC>> with tf.Session() as session:
#<<JC>>     
#<<JC>>     session.run(init)
#<<JC>>     batch_count = int(math.ceil(len(train_features)/batch_size))
#<<JC>> 
#<<JC>>     for epoch_i in range(epochs):
#<<JC>>         
#<<JC>>         # Progress bar
#<<JC>>         batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
#<<JC>>         
#<<JC>>         # The training cycle
#<<JC>>         for batch_i in batches_pbar:
#<<JC>>             # Get a batch of training features and labels
#<<JC>>             batch_start = batch_i*batch_size
#<<JC>>             batch_features = train_features[batch_start:batch_start + batch_size]
#<<JC>>             batch_labels = train_labels[batch_start:batch_start + batch_size]
#<<JC>> 
#<<JC>>             # Run optimizer
#<<JC>>             _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
#<<JC>> 
#<<JC>>         # Check accuracy against Test data
#<<JC>>         test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)
#<<JC>> 
#<<JC>> 
#<<JC>> assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
#<<JC>> print('Nice Job! Test Accuracy is {}'.format(test_accuracy))


# # Multiple layers
# Good job!  You built a one layer TensorFlow network!  However, you want to build more than one layer.  This is deep learning after all!  In the next section, you will start to satisfy your need for more layers.
