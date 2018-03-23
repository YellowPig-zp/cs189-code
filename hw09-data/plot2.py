import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from starter import *


def neural_network(X, Y, X_test, Y_test, num_neurons, activation):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_neurons: number of neurons in each layer
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    n_hidden = num_neurons
    lr = 0.0000001
    training_epochs = 30000

    x = tf.placeholder(tf.float32, [None, 7])
    y = tf.placeholder(tf.float32, [None, 2])

    W1 = tf.Variable(tf.random_normal([7, n_hidden]))
    W2 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
    W3 = tf.Variable(tf.random_normal([n_hidden, 2]))
    b1 = tf.Variable(tf.random_normal([n_hidden]))
    b2 = tf.Variable(tf.random_normal([n_hidden]))
    b3 = tf.Variable(tf.random_normal([2]))

    activations = {"ReLU": tf.nn.relu, "tanh":tf.nn.tanh}
    a = activations[activation]
    layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    pred = tf.matmul(layer2, W3) + b3
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum((pred - y) ** 2, axis=1)))

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={x: X, y: Y})
        mse = tf.reduce_mean(tf.sqrt(tf.reduce_sum((pred - y) ** 2, axis=1)))
        mse_value = mse.eval({x: X_test, y: Y_test})
    return mse_value


#############################################################################
#######################PLOT PART 2###########################################
#############################################################################
def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
    return generate_dataset(
        sensor_loc,
        num_sensors=k,
        spatial_dim=d,
        num_data=n,
        original_dist=original_dist,
        noise=noise)


np.random.seed(0)
n = 200
num_neuronss = np.arange(100, 550, 50)
mses = np.zeros((len(num_neuronss), 2))

# for s in range(replicates):

sensor_loc = generate_sensors()
X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
X_test, Y_test = generate_data(sensor_loc, n=1000)
for t, num_neurons in enumerate(num_neuronss):
    ### Neural Network:
    mse = neural_network(X, Y, X_test, Y_test, num_neurons, "ReLU")
    mses[t, 0] = mse

    mse = neural_network(X, Y, X_test, Y_test, num_neurons, "tanh")
    mses[t, 1] = mse

    print('Experiment with {} neurons done...'.format(num_neurons))

### Plot MSE for each model.
plt.figure()
activation_names = ['ReLU', 'Tanh']
for a in range(2):
    plt.plot(num_neuronss, mses[:, a], label=activation_names[a])

plt.title('Error on validation data verses number of neurons')
plt.xlabel('Number of neurons')
plt.ylabel('Average Error')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('num_neurons.png')
