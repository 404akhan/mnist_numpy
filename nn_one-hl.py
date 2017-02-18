import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data

data_dir = '/home/akhan/PycharmProjects/untitled6/input_data'

def predict(xs, w1, w2):
    z1 = np.matmul(w1, xs)
    a1 = 1 / (1 + np.exp(-z1))

    z2 = np.matmul(w2, a1)
    a2 = np.exp(z2)

    return np.argmax(a2, axis=0)

def gradients(xs, ys, w1, w2):
    z1 = np.matmul(w1, xs)
    a1 = 1 / (1 + np.exp(-z1))

    z2 = np.matmul(w2, a1)
    a2 = np.exp(z2)
    a2_sum = np.sum(a2, axis=0)

    sigma2 = np.zeros([10, len(ys)])
    for l in range(10):
        for i in range(len(ys)):
            sigma2[l][i] += -((ys[i] == l) - a2[l][i] / a2_sum[i])

    sigma1 = np.matmul(w2.transpose(), sigma2) * a1 * (1 - a1)

    grads2 = np.matmul(sigma2, a1.transpose())
    grads1 = np.matmul(sigma1, xs.transpose())

    return (grads1, grads2)

def loging(iteration):
    iteration += 1
    if iteration % 100 == 0:
        sys.stdout.write("\rIteration %i" % iteration)
        sys.stdout.flush()

def log_accuracy(iteration, mnist, w1, w2):
    iteration += 1
    if iteration % 5000 == 0:
        test_xs = mnist.test.images
        test_ys = mnist.test.labels

        predictions = predict(test_xs.transpose(), w1, w2)
        accuracy = np.sum(predictions == test_ys) * 1. / len(test_ys)

        print ', accuracy:', accuracy

def main(lr = 0.1, lam = 0, batch_size = 100, hl_size = 40):
    mnist = input_data.read_data_sets(data_dir, one_hot=False)

    w1 = 0.1 * np.random.randn(hl_size, 784)
    w2 = 0.1 * np.random.randn(10, hl_size)

    for iteration in range(25000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        grads1, grads2 = gradients(batch_xs.transpose(), batch_ys, w1, w2)

        w1 = w1 - lr * (1./batch_size * grads1 + lam/batch_size * w1)
        w2 = w2 - lr * (1./batch_size * grads2 + lam/batch_size * w2)

        loging(iteration)
        log_accuracy(iteration, mnist, w1, w2)

main()

# todo 1) random init 2) p ~ n think lambda 3) batch_size = all 4) hidden layer size vary