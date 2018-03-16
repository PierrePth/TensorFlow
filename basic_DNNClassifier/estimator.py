#!/usr/bin/env python3

# This file is licensed under the Apache license, Version 2.0.
# You may not use this file except in compliance with the license
# You can find a copy of this file at :
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# This file was modified from its original state that can be found in the
# TensorFlow tutorial : Getting started with TensorFlow. Further description
# may be found here :
# This tutorial can be found here :
#
#   https://www.tensorflow.org/get_started/premade_estimators 
#
# The code was modify in an effort to understand the phylosophy behind
# TensorFlow.
# See the license for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import argparse
import pandas as pd
""" pandas is a python data analysis toolkit that provides fast, flexible
    and expressive data structure to work with labeled data """
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

""" Parse all the input arguments """
args = parser.parse_args(sys.argv[1:])


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

""" the kera.utils.get_file method download a file from an URL if is not
    already in the cache. """
def maybe_download():
  train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
  test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

  return train_path, test_path

def load_data(y_name='Species'):
  """ Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
  train_path, test_path = maybe_download()

  """ Because the data set files are CSV files, we need to parse them """
  train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
  train_x, train_y = train, train.pop(y_name)

  test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
  test_x, test_y = test, test.pop(y_name)

  return (train_x, train_y), (test_x, test_y)

""" The two pairs returned are the training input / output and the tests
    input / output """
(train_x, train_y), (test_x, test_y) = load_data()

""" Defines some input training example and the expected result.
    features is a dictionnary which contains the following association :
    
    key : The name of the parameter
    Value : The possible values  of the parameter"""
def input_evaluation_set():
  features = {
    'SepalLength' : np.array([6.4, 5.0]),
    'SepalWidth'  : np.array([2.8, 2.3]),
    'PetalLength' : np.array([5.6, 3.3]),
    'PetalWidth'  : np.array([2.2, 1.0]),
  }
  """ labels is an array of labels for every example.
      In TensorFlow, a label is the result of an example. """
  labels = np.array([2,1])
  """ This means that having
    6.4 2.8 5.6 2.2 as paramater should activate the 1th output neurons
    (0 bases index) """
  return features, labels

def train_input_fn(features, labels, batch_size):
  """ Create a dataset from the features and labels defined beforehand.
      features being a dictionnary, we can transform this dictionnary of
      array to a dataset of dictionnaries. By adding labels as parameters
      to from_tensor_slices, we create a dataset of pair of dictionnaries
      To convince yourself, print the dataset. The dict method build a
      dictionnary from key / value pairs."""
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
  """ Repeat the examples / batch them.

      The shuffle method uses a fixed size buffer to shuffle items of the
      data set. We use a greater size than the actual number of example in
      order to ensure the data is well shuffled.
  
      The repeat method repeats the data set a count times. count being
      its parameter. No parameters means no limit.
      
      The batch methods stacks the number of examples (batch_size) and
      adds a dimension to their shape. The shape of a tensor is the list
      of its dimensions.
      
      For example :
        t = tf.constant([[[1,1,1], [2,2,2]], [[3,3,3], [4,4,4]]])
        tf.shape(t)   # [2,2,3]

      The data set has an unknown batch size because the last batch will
      less elements. Now, the data set contains a 1D vector of elements """
 
  return dataset.shuffle(1000).repeat().batch(batch_size)

""" Lets build a function for evaluation """
def eval_input_fn(features, labels, batch_size):
  features = dict(features)
  if labels is None:
    inputs = features
  else :
    inputs = (features, labels)

  dataset = tf.data.Dataset.from_tensor_slices(inputs)

  assert batch_size is not None, "Batch size must be valid"
  dataset = dataset.batch(batch_size)

  return dataset

""" How to use data input in our model ? Use features column and
    take the key of the input set """
my_feature_columns = []
""" Add a feature column for each parameter (input neuron) """
for k in train_x.keys():
  my_feature_columns.append(tf.feature_column.numeric_column(key=k))

""" Building up the DNNClassifier
    Parameters are :
      . Which feature column to use
      . Number of neurons per hidden layer given as array
      . number of output neurons (classes) """
classifier = tf.estimator.DNNClassifier (
  feature_columns = my_feature_columns,
  hidden_units = [10, 10],
  n_classes = 3
)

""" The model is now created, we can train it.
    To do so, we pass to our model an input function we define ourselves
    throught a lambda. Here our lamda. The batch_step argument fix a
    maximum number of training step."""
classifier.train (
  input_fn = lambda:train_input_fn(train_x, train_y, args.batch_size),
  steps = args.train_steps
)

""" We can evaluate the model """
eval_result  = classifier.evaluate (
  input_fn = lambda:eval_input_fn(test_x, test_y, args.batch_size)
)

print ('\nTest set accuracy : {accuracy: 0.3f}\n'.format(**eval_result))
