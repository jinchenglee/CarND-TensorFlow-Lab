
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

# Drop out probability
keep_prob = tf.placeholder(tf.float32)
TRAIN_PROB = 0.5

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
train_feed_dict = {features: train_features, labels: train_labels, keep_prob: 1.0}
valid_feed_dict = {features: valid_features, labels: valid_labels, keep_prob: 1.0}
test_feed_dict = {features: test_features, labels: test_labels, keep_prob: 1.0}

# Linear Function WX + b
#tmp = tf.matmul(features, weights_hidden_layer) + biases_hidden_layer
tmp = tf.add(tf.matmul(features, weights_hidden_layer),biases_hidden_layer)
tmp_relu = tf.nn.relu(tmp)
tmp_dropout = tf.nn.dropout(tmp_relu, keep_prob)

# Hidden layer
logits = tf.add(tf.matmul(tmp_dropout, weights), biases)

prediction = tf.nn.softmax(logits)

# Using below code instead of tf.nn.softmax_cross_entropy_with_logits() will hit
# numerical instability issue. as prediction can be quite possible be exactly 0 after RELU, 
# thus tf.log(0) is infinity -- causing NaN in cross_entropy etc. 
#
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
epochs = 100
 
for batch_size in [128,256,512]:
    for learning_rate in [0.0007,0.001,0.003]: 
        
        ### DON'T MODIFY ANYTHING BELOW ###
        # Gradient Descent
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)    
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    
        
        # Create an operation that initializes all variables
        init = tf.initialize_all_variables()
        
        # Saver of the network
        save_file = 'model.ckpt'
        saver = tf.train.Saver()
        
        
        # The accuracy measured against the validation set
        validation_accuracy = 0.0
        
        # Measurements use for graphing loss and accuracy
        log_batch_step = 250
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
                        feed_dict={features: batch_features, labels: batch_labels, keep_prob:TRAIN_PROB})
        
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
        
                # Check accuracy against Test data
                test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)
          
            # Save the network
            saver.save(session, save_file)
        
        #loss_plot = plt.subplot(211)
        #loss_plot.set_title('Loss')
        #loss_plot.plot(batches, loss_batch, 'g')
        #loss_plot.set_xlim([batches[0], batches[-1]])
        #acc_plot = plt.subplot(212)
        #acc_plot.set_title('Accuracy')
        #acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
        #acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
        #acc_plot.set_ylim([0, 1.0])
        #acc_plot.set_xlim([batches[0], batches[-1]])
        #acc_plot.legend(loc=4)
        #plt.tight_layout()
        #plt.show()
         
        #print('Training accuracy at {}'.format(training_accuracy))
        #print('Validation accuracy at {}'.format(validation_accuracy))
        #assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
        #print('Nice Job! Test Accuracy is {}'.format(test_accuracy))

        print("batch_size %4d, lr %f, train_accu %4.2f, valid_accu %4.2f, test_accu %4.2f" % (batch_size, learning_rate, training_accuracy, validation_accuracy, test_accuracy))


