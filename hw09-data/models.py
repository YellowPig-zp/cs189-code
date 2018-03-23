import numpy as np
import scipy.spatial
from starter import *
import tensorflow as tf
from sklearn import preprocessing

#####################################################################
## Models used for predictions.
#####################################################################
def optimize(x, y, pred, optimizer, training_epochs, X, Y, Xs_test, Ys_test):
    with tf.Session() as sess:
        sess.run(tf.global_variable_initializer())
        total_loss = 0
        for epoch in range(training_epochs):
            sess.run([optimizer], feed_dict={x: X, y: Y})
        mse = tf.reduce_mean(tf.sqrt(tf.reduce_sum((pred - y) ** 2, axis=1)))
        mses = [mse.eval({x: X_test, y: Y_test}) for X_test, Y_test in zip(Xs_test, Ys_test)]
    return mses

def compute_update(single_obj_loc, sensor_loc, single_distance):
    """
    Compute the gradient of the log-likelihood function for part a.

    Input:
    single_obj_loc: 1 * d numpy array.
    Location of the single object.

    sensor_loc: k * d numpy array.
    Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    grad: d-dimensional numpy array.

    """
    loc_difference = single_obj_loc - sensor_loc  # k * d.
    phi = np.linalg.norm(loc_difference, axis=1)  # k.
    grad = loc_difference / np.expand_dims(phi, 1)  # k * 2.
    update = np.linalg.solve(grad.T.dot(grad), grad.T.dot(single_distance - phi))

    return update


def get_object_location(sensor_loc, single_distance, num_iters=20, num_repeats=10):
    """
    Compute the gradient of the log-likelihood function for part a.

    Input:

    sensor_loc: k * d numpy array. Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    obj_loc: 1 * d numpy array. The mle for the location of the object.

    """
    obj_locs = np.zeros((num_repeats, 1, 2))
    distances = np.zeros(num_repeats)
    for i in range(num_repeats):
        obj_loc = np.random.randn(1, 2) * 100
        for t in range(num_iters):
            obj_loc += compute_update(obj_loc, sensor_loc, single_distance)

        distances[i] = np.sum((single_distance - np.linalg.norm(obj_loc - sensor_loc, axis=1))**2)
        obj_locs[i] = obj_loc

    obj_loc = obj_locs[np.argmin(distances)]

    return obj_loc[0]


def generative_model(X, Y, Xs_test, Ys_test):
    """
    This function implements the generative model.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    initial_sensor_loc = np.random.randn(7, 2) * 100
    estimated_sensor_loc = find_mle_by_grad_descent_part_e(
        initial_sensor_loc, Y, X, lr=0.001, num_iters=1000)

    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array(
            [get_object_location(estimated_sensor_loc, X_test_single) for X_test_single in X_test])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def oracle_model(X, Y, Xs_test, Ys_test, sensor_loc):
    """
    This function implements the generative model.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    sensor_loc: location of the sensors.
    Output:
    mse: Mean square error on test data.
    """
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array([
            get_object_location(sensor_loc, X_test_single)
            for X_test_single in X_test
        ])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def linear_regression(X, Y, Xs_test, Ys_test):
    """
    This function performs linear regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """

    ## YOUR CODE HERE
    #################
    poly = preprocessing.PolynomialFeatures(1)
    new_X = poly.fit_transform(X)
    w = np.linalg.inv(new_X.T @ new_X) @ new_X.T @ Y
    mses = [np.mean(np.sqrt(np.sum((poly.fit_transform(X_test) @ w - Y_test) ** 2, axis=1))) for X_test, Y_test in zip(Xs_test, Ys_test)]
    return mses


def poly_regression_second(X, Y, Xs_test, Ys_test):
    """
    This function performs second order polynomial regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    poly = preprocessing.PolynomialFeatures(2)
    new_X = poly.fit_transform(X)
    w = np.linalg.inv(new_X.T @ new_X) @ new_X.T @ Y
    mses = [np.mean(np.sqrt(np.sum((poly.fit_transform(X_test) @ w - Y_test) ** 2, axis=1))) for X_test, Y_test in zip(Xs_test, Ys_test)]
    return mses


def poly_regression_cubic(X, Y, Xs_test, Ys_test):
    """
    This function performs third order polynomial regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    poly = preprocessing.PolynomialFeatures(3)
    new_X = poly.fit_transform(X)
    w = np.linalg.inv(new_X.T @ new_X) @ new_X.T @ Y
    mses = [np.mean(np.sqrt(np.sum((poly.fit_transform(X_test) @ w - Y_test) ** 2, axis=1))) for X_test, Y_test in zip(Xs_test, Ys_test)]
    return mses


def neural_network(X, Y, Xs_test, Ys_test):
    """
    This function performs neural network prediction.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################

    n_hidden = 100
    lr = 0.1
    training_epochs = 100

    x = tf.placeholder(tf.float32, [None, 7])
    y = tf.placeholder(tf.float32, [None, 2])

    W1 = tf.Variable(tf.random_normal(7, n_hidden))
    W2 = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
    W3 = tf.Variable(tf.random_normal([n_hidden, 2]))
    b1 = tf.Variable(tf.random_normal([n_hidden]))
    b2 = tf.Variable(tf.random_normal([n_hidden]))
    b3 = tf.Variable(tf.random_normal([2]))

    layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    pred = tf.matmul(layer2, W3) + b3
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum((pred - y) ** 2, axis=1)))

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    return optimize(x, y, pred, optimizer, training_epochs, X, Y, Xs_test, Ys_test)


