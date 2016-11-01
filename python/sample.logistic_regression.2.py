def build_logistic_regression_model(n_in, n_out, l2_coef=30.0)
    # DECLARE SOME VARIABLES
    import tensor as T

    x = T.matrix()  #our points, one point per row
    y = T.matrix()  #store our labels as place codes (label 3 of 5 is vector [00100])

    w = T.matrix()  #the linear transform to apply to our input points
    b = T.vector()  #a vector of biases, which make our transform affine instead of linear

    stepsize = T.scalar('stepsize')  # a stepsize for gradient descent

    # REGRESSION MODEL AND COSTS TO MINIMIZE
    prediction = T.softmax(T.dot(x, w) + b)
    cross_entropy = T.sum(y * T.log(prediction), axis=1)
    cost = T.sum(cross_entropy) + l2_coef * T.sum(T.sum(w * w))

    # GET THE GRADIENTS NECESSARY TO FIT OUR PARAMETERS
    grad_w, grad_b = T.grad(cost, [w, b])

    #
    # GET THE GRADIENTS NECESSARY TO FIT OUR PARAMETERS
    update_fn = theano.function(
        inputs = [x, y, stepsize,
            In(w, 
                name = 'w', 
                value = numpy.zeros((n_in, n_out)),
                update = w - stepsize * grad_w,
                mutable = True,
                strict = True)
            In(b, 
                name = 'b', 
                value = numpy.zeros(n_out),
                update = b - lr * grad_b,
                mutable = True,
                strict = True)
        ],
        outputs = cost,
        mode = 'EXPENSIVE_OPTIMIZATIONS')

    apply_fn = theano.function(
        inputs = [x, In(w, value = update_fn.storage[w]), In(b, value = update_fn.storage[b])],
        outputs = [prediction])

    return update_fn, apply_fn

#USUALLY THIS WOULD BE IN A DIFFERENT FUNCTION/CLASS
#FIT SOME DUMMY DATA: 100 points with 10 attributes and 3 potential labels

up_fn, app_fn = build_logistic_regression_model(n_in=10, n_out=3, l2_coef=30.0)

x_data = numpy.random.randn(100, 10)
y_data = numpy.random.randn(100, 3)
y_data = theano._asarray(y_data == numpy.max(y_data, axis=1), dtype='int64')

print "Model Training ..."
for iteration in xrange(1000):
    print "  iter", iteration, "cost", update_fn(x_data, y_data, stepsize=0.0001)

print "Model Predictions"
print apply_fn(x_data)