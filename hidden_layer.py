import matplotlib.pyplot as plt
import numpy as np
import struct


num_labels = 10
num_train_images = 1000
num_test_images = 1000
num_hidden_units = 20

def batch_gradient_descent(train_images, train_labels, alpha, method='max'):
    weights = np.zeros([num_labels, train_images.shape[1]])
    errors = []
    accuracy_list = []

    for i, d in enumerate(train_images):
        # keep track of predictions and accuracy for each epoch
        predictions = predict(weights, train_images, method)
        acc = accuracy(predictions, train_labels)
        accuracy_list.append(np.sum(acc))
        errors.append(np.sum(1-acc))

#        print "epoch: " + str(i)
        # gradient descent update rule
        weights -=  alpha * batch_gradient(weights, train_images, train_labels, method)
    
    return weights, errors, accuracy_list

def batch_gradient(weights, train_images, train_labels, method='max'):
    # create the one hot encoding vectors according to the label of each image
    y = np.zeros([num_train_images, num_labels])
    for i, l in enumerate(train_labels):
        y[i][l] = 1

    # t is our prediction and y is the true labels
    
    # use sigmoidal function for the logistic function
    if method == 'max':
        t = 1 / (1 + np.exp(-np.dot(weights, train_images.T)))
        return np.dot((t - y.T), train_images)

    # use the softmax activation function so that all the probabilities for
    # each column vector adds to 1
    elif method == 'softmax':
        t = np.exp(np.dot(weights, train_images.T))
        t = np.nan_to_num(t) # make nan values 0 
        t = t / t.sum(axis=0) # normalize each of the column vectores 
        return -((np.dot((y.T - t), train_images) / num_test_images) + weights)
        

def stochastic_gradient_descent(train_images, train_labels, alpha, method='max'):
    # keep track of predictions and accuracy for each epoch
    weights = np.zeros([num_labels, train_images.shape[1]])
    errors = []
    for i, d in enumerate(train_images):
        predictions = predict(weights, train_images, method)
        acc = accuracy(predictions, train_labels)
        errors.append(np.sum(1-acc))

#        print "epoch: " + str(i)
        # gradient descent update rule
        weights -= alpha * stochastic_gradient(weights, train_images[i], train_labels[i], method)

    return weights, errors

def stochastic_gradient(weights, train_image, train_label, method='max'):
    y = np.zeros(num_labels)
    y[train_label] = 1

    # t is our prediction and y is the true labels

    # use sigmoidal function for the logistic function
    if method == 'max':
        t = 1 / (1 + np.exp(-np.dot(weights, train_image))) 
        return np.dot((t-y).reshape(t.shape[0], 1), train_image.reshape(1, train_image.shape[0]))

    # use the softmax activation function so that all the probabilities for
    # each column vector adds to 1
    elif method == 'softmax':
        t = np.exp(np.dot(weights, train_image))
        t = np.nan_to_num(t) # make nan values 0 
        t = t / t.sum(axis=0) # normalize each of the column vectores 
        return -((np.dot((y-t).reshape(t.shape[0], 1), train_image.reshape(1, train_image.shape[0])) / num_train_images) + weights)

def predict(weights, test_images, method='max'):

    if method == 'max':
        t = 1 / (1 + np.exp(-np.dot(weights, test_images.T)))
        # the index of the maximum value gives us our predicted label
        return np.argmax(t, axis=0)

    elif method == 'softmax':
        t = np.exp(np.dot(weights, test_images.T))
        t = t / t.sum(axis=0)
        return np.argmax(t, axis=0)


def accuracy(predictions, true_labels):
    return np.count_nonzero(predictions == true_labels) / float(len(true_labels))

def create_plot(x, y, title, xlabel, ylabel, style):
    plt.plot(x, y, style)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def main():
    train_images_filename = 'train-images-idx3-ubyte'
    train_labels_filename = 'train-labels-idx1-ubyte'

    test_images_filename = 't10k-images-idx3-ubyte'
    test_labels_filename = 't10k-labels-idx1-ubyte'
    
    ############ Load Train train_images ################
    with open(train_images_filename, 'rb') as image_file:
        train_images_data = image_file.read()

    # The binarized data is encoded in big endian format
    # The first 4 values in the binary file are 4 byte ints
    magic, numTrainImages, rowDim, colDim = struct.unpack('>iiii', train_images_data[:16])
    # After the first 4 integers, the rest of the data
    train_images = np.array(struct.unpack('>' + 'B' * (784 * num_train_images), train_images_data[16:16+(num_train_images*784)])).reshape(num_train_images, rowDim * colDim)

    train_images = np.insert(train_images, 0, 1, axis=1)

    ############ Load Train Labels ################
    with open(train_labels_filename, 'rb') as image_file:
        train_labels_data = image_file.read()

    magic, numLables = struct.unpack('>ii', train_labels_data[:8])
    train_labels = np.array(struct.unpack('>' + 'B' * num_train_images, train_labels_data[8:8+num_train_images]))

    ############ Load Test Data ################
    with open(test_images_filename, 'rb') as image_file:
        test_images_data = image_file.read()

    magic, numtestImages, rowDim, colDim = struct.unpack('>iiii', test_images_data[:16])
    test_images = np.array(struct.unpack('>' + 'B' * (784 * num_test_images), test_images_data[16:16+(num_test_images*784)])).reshape(num_test_images, rowDim * colDim)

    test_images = np.insert(test_images, 0, 1, axis=1) 
    ############ Load Test Labels ################
    with open(test_labels_filename, 'rb') as image_file:
        test_labels_data = image_file.read()

    magic, numLables = struct.unpack('>ii', test_labels_data[:8])
    test_labels = np.array(struct.unpack('>' + 'B' * num_test_images, test_labels_data[8:8+num_test_images]))

    fig = plt.figure()
    ############## Sigmoid Max ##############
    # Run Batch Gradient Descent
    batch_weights, batch_errors, batch_accuracies = batch_gradient_descent(train_images, train_labels, 0.01, 'max')
    batch_predictions = predict(batch_weights, test_images, 'max') # Run prediction
    print "Batch Gradient Descent Sigmoid Accuracy: {}", accuracy(batch_predictions, test_labels) # Calculate accuracy
    plt.subplot(331)
    create_plot(range(len(batch_errors)), batch_errors, 'Sigmoid Batch Gradient Descent Error', 'Epoch', 'Error', 'bo-') # Plot errors for each epoch

    # Run Stochastic Gradient Descent
    stochastic_weights, stochastic_errors = stochastic_gradient_descent(train_images, train_labels, 0.01, 'max')
    stochastic_predictions = predict(stochastic_weights, test_images, 'max')
    print "Stochastic Gradient Descent Sigmoid Accuracy: {}", accuracy(stochastic_predictions, test_labels)

    ############## Softmax ################
    # Run Batch Gradient Descent
    batch_weights, softmax_batch_errors, softmax_batch_accuracies = batch_gradient_descent(train_images, train_labels, 0.00001, 'softmax')
    batch_predictions = predict(batch_weights, test_images, 'softmax') # Run prediction
    print "Batch Gradient (Softmax) Descent Accuracy: {}", accuracy(batch_predictions, test_labels) # Calculate accuracy
    plt.subplot(336)
    create_plot(range(len(softmax_batch_accuracies)), softmax_batch_accuracies, 'Softmax Stochastic Gradient Descent Accuracies', 'Epoch', 'Accuracy Ratio', 'bo-') # Plot errors for each epoch

    # Run Stochastic Gradient Descent
    stochastic_weights, softmax_stochastic_errors = stochastic_gradient_descent(train_images, train_labels, 0.00001, 'softmax')
    stochastic_predictions = predict(stochastic_weights, test_images, 'softmax')
    print "Stochastic Gradient Descent (Softmax) Accuracy: {}", accuracy(stochastic_predictions, test_labels)
    plt.subplot(337)
    create_plot(range(len(softmax_stochastic_errors)), softmax_stochastic_errors, 'Softmax Stochastic Gradient Descent Error', 'Epoch', 'Error Ratio', 'bo-') # Plot errors for each epoch

    plt.savefig('plots.png')
    plt.show()


if __name__ == '__main__':
    main()
