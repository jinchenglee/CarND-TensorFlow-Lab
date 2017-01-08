
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

# Learning rate
learning_rate = 0.03

# Gradient Descent
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)    
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

# Create an operation that initializes all variables
init = tf.initialize_all_variables()

# Saver of the network
save_file = 'model.ckpt'
saver = tf.train.Saver()


### DON'T MODIFY ANYTHING BELOW ###
# The accuracy measured against the test set
test_accuracy = 0.0

with tf.Session() as session:
    
    saver.restore(session, '/home/vitob/git_repositories/CarND-TensorFlow-Lab/model.ckpt')

    # Check accuracy against Test data
    test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)

assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
print('Nice Job! Test Accuracy is {}'.format(test_accuracy))

