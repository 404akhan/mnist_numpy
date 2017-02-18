import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data_dir = '/home/akhan/PycharmProjects/untitled6/input_data'

def predict(xs, theta):
    z = np.matmul(xs, theta.transpose())
    return np.argmax(z, axis=1)

def probability(l, xs, theta):
    a = np.exp(np.matmul(xs, theta.transpose()))
    return a[0][l] / np.sum(a[0])

def gradients(l, xs, ys, theta):
    sum = np.zeros(len(xs[0]))
    for i in range(len(xs)):
        sum += -1 * xs[i] * ((ys[i]==l) - probability(l, [xs[i]], theta))
    sum /= len(xs)
    return sum

def main(lr = 0.5):
    mnist = input_data.read_data_sets(data_dir, one_hot=False)

    theta = np.zeros([10, 784]) # todo 784 -> 785
    grads = np.zeros([10, 784])

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)

        for l in range(10):
            grads[l] = gradients(l, batch_xs, batch_ys, theta)
        theta = theta - lr * grads

    test_xs = mnist.test.images
    test_ys = mnist.test.labels

    predictions = predict(test_xs, theta)
    accuracy = np.sum(predictions == test_ys) * 1. / len(test_ys)

    print 'accuracy:', accuracy

main()
