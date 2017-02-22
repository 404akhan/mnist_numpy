import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data

data_dir = '/home/akhan/PycharmProjects/untitled6/input_data'
mnist = input_data.read_data_sets(data_dir, one_hot=False)

# relu, bias, immute

def get_predictions(Xs, W1, b1, W2, b2):
    hl = np.maximum(0, np.matmul(W1, Xs) + b1)
    scores = np.matmul(W2, hl) + b2

    return np.argmax(scores, axis=0)

def get_accuracy(W1, b1, W2, b2):
    test_xs = mnist.test.images
    test_ys = mnist.test.labels

    predictions = get_predictions(test_xs.transpose(), W1, b1, W2, b2)
    accuracy = np.mean(predictions == test_ys)

    return accuracy

def log_accuracy(iteration, W1, b1, W2, b2):
    if iteration % 5000 == 0:
        accuracy = get_accuracy(W1, b1, W2, b2) * 100
        print ', accuracy %.2f%%' % accuracy

def log_iteration(iteration):
    if iteration % 100 == 0:
        sys.stdout.write('\rIteration %d' % iteration)
        sys.stdout.flush()

def main(step_size=0.1, reg=0., num_iterations = 25001, batch_size = 100):
    print '\nTraining for step_size %f, reg %f started!' % (step_size, reg)

    hl_size = 40
    W2 = 0.1 * np.random.randn(10, hl_size)
    b2 = np.zeros([10, 1])
    W1 = 0.1 * np.random.randn(hl_size, 784)
    b1 = np.zeros([hl_size, 1])

    for iteration in range(num_iterations):
        Xs, ys = mnist.train.next_batch(batch_size)
        Xs = Xs.transpose() # Xs | 784 * batch_size, ys | 784

        hl = np.maximum(0, np.matmul(W1, Xs) + b1) # hl | 40 * batch_size
        scores = np.matmul(W2, hl) + b2 # scores | 10 * batch_size

        probs = np.exp(scores) # probs | 10 * batch_size
        probs /= np.sum(probs, axis=0, keepdims=True)

        dscores = probs # dscores | 10 * batch_size
        dscores[ys, range(batch_size)] -= 1
        dscores /= batch_size

        dhl = np.matmul(W2.transpose(), dscores) # dhl | 40 * batch_size
        dhl[hl <= 0] = 0

        dW2 = np.matmul(dscores, hl.transpose()) # dW2 | 10 * 40
        dW1 = np.matmul(dhl, Xs.transpose()) # dW1 | 40 * 784
        db2 = np.sum(dscores, axis=1, keepdims=True) # db2 | 10 * 1
        db1 = np.sum(dhl, axis=1, keepdims=True) # db1 | 40 * 1
        dW2 += reg * W2
        dW1 += reg * W1

        W2 -= step_size * dW2
        W1 -= step_size * dW1
        b2 -= step_size * db2
        b1 -= step_size * db1

        log_iteration(iteration)
        log_accuracy(iteration, W1, b1, W2, b2)

main()