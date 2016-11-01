#! /usr/bin/python

import numpy
import theano
import theano.tensor as T

class softmax_regression(object):

    def __init__(self, nIn, nOut, l2_coef=0.01):
        # DECLARE SOME VARIABLES

        self.nIn = nIn
        self.nOut = nOut
        self.l2_coef = l2_coef
        self.W = 2 * numpy.random.random((self.nIn, self.nOut)) - 1 # numpy.zeros((self.nIn, self.nOut))
        self.b = 2 * numpy.random.random(self.nOut) - 1 # numpy.zeros(self.nOut)

    def build_logistic_regression_model(self):
        X = T.matrix('X')  #our points, one point per row
        Y = T.matrix('Y')  #store our labels as place codes (label 3 of 5 is vector [00100])

        W = T.matrix('W')  #the linear transform to apply to our input points
        b = T.vector('b')  #a vector of biases, which make our transform affine instead of linear

        learning_rate = T.scalar('learning_rate')  # a learning_rate for gradient descent

        # REGRESSION MODEL AND COSTS TO MINIMIZE
        prediction = T.nnet.softmax(T.dot(X, W) + b)
        cross_entropy = -1 * T.sum(Y * T.log(prediction), axis = 1)
        cost = self.l2_coef * T.sum((W ** 2).sum()) + cross_entropy.mean()

        # GET THE GRADIENTS NECESSARY TO FIT OUR PARAMETERS
        grad_w, grad_b = T.grad(cost, [W, b])

        #
        # GET THE GRADIENTS NECESSARY TO FIT OUR PARAMETERS
        update_fn = theano.function(
            inputs = [X, Y, learning_rate,
                theano.In(W, 
                    name = 'W', 
                    value = self.W,
                    update = W - learning_rate * grad_w,
                    mutable = True,
                    strict = True), 
                theano.In(b, 
                    name = 'b', 
                    value = self.b,
                    update = b - learning_rate * grad_b,
                    mutable = True,
                    strict = True)
            ],
            outputs = cost,
            mode = 'FAST_RUN')

        apply_fn = theano.function(
            inputs = [X, 
                theano.In(W, value = self.W), 
                theano.In(b, value = self.b)
            ], 
            outputs = prediction)

        return update_fn, apply_fn

def eval(golden, pred):
    nCorrect = 0
    sum = golden.shape[0]
    for i in range(sum):
        if (golden[i] == pred[i]).all() :
            nCorrect += 1

    print "    Total ", sum, "samples."
    print "    Precision ", nCorrect, "/", sum, "=", float(nCorrect)/float(sum)

    return nCorrect, sum, float(nCorrect)/float(sum)

#USUALLY THIS WOULD BE IN A DIFFERENT FUNCTION/CLASS
#FIT SOME DUMMY DATA: 100 points with 10 attributes and 3 potential labels

sr = softmax_regression(nIn = 10, nOut = 3, l2_coef = 0.01)
train, apply = sr.build_logistic_regression_model()

x_data = numpy.random.randn(100, 10)
y_data = numpy.random.randn(100, 3)
y_data = theano._asarray(numpy.exp(y_data - y_data.max(axis = 1, keepdims = True)), dtype = 'int64')
y_data = theano._asarray(y_data, dtype = theano.config.floatX)

print "Model Training ..."
for iteration in xrange(20000):
    cost = train(x_data, y_data, learning_rate = 0.01)
    if (iteration % 100 == 0):
        print "  iter", iteration, "cost", cost

print "Model Predictions"
y_pred = apply(x_data)
y_pred = theano._asarray(numpy.exp(y_pred - y_pred.max(axis = 1, keepdims = True)), dtype = 'int64')
eval(y_data, y_pred)