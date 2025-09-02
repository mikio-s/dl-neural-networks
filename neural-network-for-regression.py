%load_ext autoreload

%autoreload 2

import numpy as np
from numpy.matlib import repmat
from scipy.io import loadmat
import time

# Non-interactive matplotlib plots
%matplotlib inline
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/codio/workspace/.modules")
from helper import *

print("Python version:", sys.version.split(" ")[0])

X, y = generate_data()
print(f"The shape of X is {X.shape}")

plt.figure()
plt.plot(X[:, 0], y, "*")
plt.title("Generated Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_grad(x):
    return (x > 0).astype("float64")

plt.figure()
plt.plot(
    np.linspace(-2, 2, 1000),
    ReLU(np.linspace(-2, 2, 1000)),
    label=r"ReLU $\sigma(x)$",
    lw=2,
)
plt.plot(
    np.linspace(-2, 2, 1000),
    ReLU_grad(np.linspace(-2, 2, 1000)),
    "-.",
    label="ReLU_grad",
    lw=2,
)
plt.title("ReLU Activation Function and Its Gradient")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc="upper left")
plt.show()

x = np.array([2.7, -0.5, -3.2])
print("           x:", x)
print("     ReLU(x):", ReLU(x))
print("ReLU_grad(x):", ReLU_grad(x))

def initweights(specs):
    """
    Given a specification for a neural network, output a random weight array.

    Input:
        specs: Array of length m-1, where m = len(specs).
               specs[0] should be the dimension of the feature
               and spec[-1] should be the dimension of output.

    Output:
        W: Array of length m-1, each element is a matrix,
           where W[i].shape == (specs[i], specs[i+1])
    """
    W = []
    
    for i in range(len(specs) - 1):
        W.append(np.random.randn(specs[i], specs[i + 1]))
    
    return W

W = initweights([2, 3, 1])

def forward_pass(W, xTr):
    """
    Propagates the data matrix xTr forward in the network
    specified by the weight matrices in the array W.

    Input:
        W: List of L weight matrices, specifying the network
        xTr: nxd data matrix. Each row is an input vector

    OUTPUTS:
        A, Z
        A: List of L+1 matrices, each of which is the result of matrix
           multiplication of previous layer's outputs and weights.
           The first matrix in the list is xTr.
        Z: List of L+1 matrices, each of which is the result of
           transition functions on elements of A.
           The first matrix in the list is xTr.
    """
    A = [xTr]
    Z = [xTr]

    for j in range(len(W)):
        Aj = np.dot(Z[j], W[j])
        if j < len(W)-1:
            Zj = ReLU(Aj) 
        else: 
            Zj = Aj
        A.append(Aj)
        Z.append(Zj)

    return A, Z

def forward_test1():
    """
    Check that your forward_pass returns a tuple of length 2
    """
    X, _ = generate_data()      # Generate data
    W = initweights([2, 3, 1])  # Generate random weights
    out = forward_pass(W, X)    # Run your forward pass
    return type(out) == tuple and len(out) == 2


def forward_test2():
    """
    Check that your forward_pass output has the correct length
    """
    X, _ = generate_data()      # Generate data
    W = initweights([2, 3, 1])  # Generate random weights
    A, Z = forward_pass(W, X)   # Run your forward pass
    return len(A) == 3 and len(Z) == 3


def forward_test3():
    """
    Check that each layer produces output with correct shape
    """
    X, _ = generate_data()      # Generate data
    n, d = X.shape              # Get number of rows
    W = initweights([2, 3, 1])  # Generate random weights
    A, Z = forward_pass(W, X)   # Run your forward pass
    return (
        A[0].shape == (n, d)
        and Z[0].shape == (n, d)
        and A[1].shape == (n, 3)
        and Z[1].shape == (n, 3)
        and A[2].shape == (n, 1)
        and A[2].shape == (n, 1)
    )


def forward_test4():
    """
    Check that you did not apply the transition function to A[-1]
    """
    X = -1 * np.ones((1, 2))   # Generate a feature matrix of all negative ones
    W = [np.ones((2, 1))]      # Single-layer network with weights all ones
    A, Z = forward_pass(W, X)  # Run your forward pass
    return np.linalg.norm(Z[-1] - X @ W[0]) < 1e-7


def forward_test5():
    """
    Check that your forward pass generates the correct A and Z
    """
    X, _ = generate_data()                          # Generate data
    n, _ = X.shape                                  # Get number of rows
    W = initweights([2, 3, 1])                      # Generate random weights
    A, Z = forward_pass(W, X)                       # Run your forward pass
    A_grader, Z_grader = forward_pass_grader(W, X)  # Run our forward pass

    # Compute the difference between your solution and ours
    Adiff = 0
    Zdiff = 0
    for i in range(1, 3):
        Adiff += np.linalg.norm(A[i] - A_grader[i])
        Zdiff += np.linalg.norm(Z[i] - Z_grader[i])

    return Adiff < 1e-7 and Zdiff < 1e-7


runtest(forward_test1, "forward_test1")
runtest(forward_test2, "forward_test2")
runtest(forward_test3, "forward_test3")
runtest(forward_test4, "forward_test4")
runtest(forward_test5, "forward_test5")

def MSE(out, y):
    """
    Calculates the Mean Squared Error (MSE) for output and true labels y

    Input:
        out: Output of network (n-dimensional vector)
        y: True labels (n-dimensional vector)

    Output:
        loss: MSE loss (a scalar)
    """
    loss = None  # Initialize variable to be returned
    loss = np.mean((out - y)**2)

    return loss

def MSE_test1():
    """
    Check that your MSE loss is a scalar
    """
    X, y = generate_data()          # Generate data
    W = initweights([2, 3, 1])      # Generate random weights
    A, Z = forward_pass(W, X)       # Run your forward pass
    loss = MSE(Z[-1].flatten(), y)  # Compute your MSE loss
    return np.isscalar(loss)


def MSE_test2():
    """
    Check that your MSE loss is non-negative
    """
    X, y = generate_data()          # Generate data
    W = initweights([2, 3, 1])      # Generate random weights
    A, Z = forward_pass(W, X)       # Run your forward pass
    loss = MSE(Z[-1].flatten(), y)  # Compute your MSE loss
    return loss >= 0


def MSE_test3():
    """
    Check that your MSE loss is close to ours.
    If this test case fails, check whether your loss is divided by 1/n.
    """
    X, y = generate_data()                        # Generate data
    W = initweights([2, 3, 1])                    # Generate random weights
    A, Z = forward_pass(W, X)                     # Run your forward pass
    loss = MSE(Z[-1].flatten(), y)                # Compute your MSE loss
    loss_grader = MSE_grader(Z[-1].flatten(), y)  # Compute our MSE loss
    return np.absolute(loss - loss_grader) < 1e-7


runtest(MSE_test1, "MSE_test1")
runtest(MSE_test2, "MSE_test2")
runtest(MSE_test3, "MSE_test3")

def MSE_grad(out, y):
    """
    Calculates the gradient of MSE loss with respect to network output

    Input:
        out: Output of network (n-dimensional vector)
        y: True labels (n-dimensional vector)

    Output:
        grad: Gradient of the MSE loss with respect to out (n-dimensional vector)
    """
    n = len(y)
    grad = np.empty(n)  # Initialize variable to be returned
    grad = 2/n * (out - y)

    return grad

def MSE_grad_test1():
    """
    Check that your MSE gradient has the right shape
    """
    X, y = generate_data()               # Generate data
    n, _ = X.shape                       # Get number of rows
    W = initweights([2, 3, 1])           # Generate random weights
    A, Z = forward_pass(W, X)            # Run your forward pass
    grad = MSE_grad(Z[-1].flatten(), y)  # Compute your MSE gradient
    return grad.shape == (n,)


def MSE_grad_test2():
    """
    Check your gradient is close to the numerical gradient
    """
    out = np.array([1])
    y = np.array([1.2])

    # Calculate numerical MSE gradient using finite difference
    numerical_grad = (MSE(out + 1e-7, y) - MSE(out - 1e-7, y)) / 2e-7
    
    # Compute your MSE gradient
    grad = MSE_grad(out, y)

    return np.linalg.norm(numerical_grad - grad) < 1e-7


def MSE_grad_test3():
    """
    Check that your MSE gradient is close to ours
    """
    X, y = generate_data()
    n, _ = X.shape

    # Generate random weights to initialize a network
    W = initweights([2, 3, 1])

    # Run your forward pass to generate A and Z
    A, Z = forward_pass(W, X)

    # Compute your MSE gradient
    grad = MSE_grad(Z[-1].flatten(), y)

    # Compute our MSE gradient
    grad_grader = MSE_grad_grader(Z[-1].flatten(), y)

    return np.linalg.norm(grad_grader - grad) < 1e-7


runtest(MSE_grad_test1, "MSE_grad_test1")
runtest(MSE_grad_test2, "MSE_grad_test2")
runtest(MSE_grad_test3, "MSE_grad_test3")

def backprop(W, A, Z, y):
    """
    Implements the back-propagation algorithm for the network specified
    by weights in W and intermediate values A, Z for the data propagated
    forward (corresponding labels y).

    Input:
        W: List of L weight matrices
        A: List of L+1 matrices, each of which is the result of matrix
           multiplication of previous layer's outputs and weights.
           The first matrix in the list is the data.
        Z: List of L+1 matrices, each of which is the result of
           transition functions on elements of A.
           The first matrix in the list is the data.
        y: True labels (n-dimensional vector)

    Output:
        gradients: List of L matrices, each of which is the gradient
                   with respect to the corresponding entry of W
    """
    # Convert delta to a row vector to make things easier
    delta = (MSE_grad(Z[-1].flatten(), y) * 1).reshape(-1, 1)

    # Compute gradient with backprop
    gradients = []

    for j in range(len(W)-1, -1, -1):
        grad = np.dot(Z[j].T, delta)
        gradients.insert(0, grad)
        if j > 0:
            delta = ReLU_grad(A[j]) * np.dot(delta, W[j].T)

    return gradients

def backprop_test1():
    """
    Check that your gradients list has the correct length.
    You should return a list with the same length as W.
    """
    X, y = generate_data()
    n, _ = X.shape

    # Generate random weights to initialize a network
    W = initweights([2, 3, 1])

    # Run your forward pass to generate A and Z
    A, Z = forward_pass(W, X)

    # Use your backprop function to calculate gradients
    gradients = backprop(W, A, Z, y)

    return len(gradients) == len(W)


def backprop_test2():
    """
    Check that your gradients list elements have the correct shape
    """
    X, y = generate_data()
    n, _ = X.shape

    # Generate random weights to initialize a network
    W = initweights([2, 3, 1])
    
    # Run your forward pass to generate A and Z
    A, Z = forward_pass(W, X)

    # Use your backprop function to calculate gradients
    gradients = backprop(W, A, Z, y)

    # Your gradients[i] should match the shape of W[i]
    return np.all([gradients[i].shape == W[i].shape for i in range(len(W))])


def backprop_test3():
    """
    Check your gradient against the least squares gradient
    """
    X, y = generate_data()
    n, _ = X.shape

    # Generate random weights to initialize a one-layer
    # network that produces the least squares gradient.
    W = initweights([2, 1])

    # Run your forward pass to generate A and Z
    A, Z = forward_pass(W, X)

    # Use your backprop function to calculate gradients
    gradients = backprop(W, A, Z, y)

    # Calculate the least squares gradient
    least_square_gradient = 2 * ((X.T @ X) @ W[0] - X.T @ y.reshape(-1, 1)) / n

    # Your gradients[0] should be the least square gradient
    return np.linalg.norm(gradients[0] - least_square_gradient) < 1e-7


def backprop_test4():
    """
    Check whether your gradient matches our gradient
    """
    X, y = generate_data()
    n, _ = X.shape

    # Generate random weights to initialize a network
    W = initweights([2, 5, 5, 1])

    # Run your forward pass to generate A and Z
    A, Z = forward_pass(W, X)

    # Use your backprop function to calculate gradients
    gradients = backprop(W, A, Z, y)

    # Use our backprop function to calculate gradients
    gradients_grader = backprop_grader(W, A, Z, y)

    # Check whether your gradients list matches ours.
    # First, check to make sure the length matches.
    # Second, check that each component matches in shape and values.
    OK = [len(gradients_grader) == len(gradients)]
    for g, gg in zip(gradients_grader, gradients):  
        OK.append(gg.shape == g.shape and (np.linalg.norm(g - gg) < 1e-7))
    
    return all(OK)


def backprop_test5():
    """
    We reverse your gradients list and compare the reverse with ours.
    It shouldn't match. If your reverse gradients matches our gradients,
    this means you computed the gradients in reverse order.
    This is a common mistake due to looping backwards.
    """
    X, y = generate_data()
    n, _ = X.shape

    # Generate random weights to initialize a network
    W = initweights([2, 5, 5, 1])

    # Run your forward pass to generate A and Z
    A, Z = forward_pass(W, X)

    # Use your backprop function to calculate gradients
    gradients = backprop(W, A, Z, y)

    # Use our backprop function to calculate gradients
    gradients_grader = backprop_grader(W, A, Z, y)

    # Reverse your gradients list; from now on it should NOT match ours
    gradients.reverse()
    
    # Check whether your reversed gradients list matches ours (it should not)
    OK = []
    for g, gg in zip(gradients_grader, gradients):
        OK.append(gg.shape == g.shape and (np.linalg.norm(g - gg) < 1e-7))
    
    return not all(OK)


runtest(backprop_test1, "backprop_test1")
runtest(backprop_test2, "backprop_test2")
runtest(backprop_test3, "backprop_test3")
runtest(backprop_test4, "backprop_test4")
runtest(backprop_test5, "backprop_test5")

# Generate data
X, y = generate_data()

# Initialize a neural network with one hidden layer.
# Try varying the depth and width of the neural networks to see the effect.
W = initweights([2, 200, 1])
W_init = [w.copy() for w in W]

M = 6000  # Number of epochs (one epoch is one full pass through dataset)
lr = 0.001  # Learning rate for gradient descent
losses = np.zeros(M)  # Initialize vector to store losses

t0 = time.time()  # Time before training

# Start training
for i in range(M):

    # Do a forward pass
    A, Z = forward_pass(W, X)

    # Calculate the loss
    losses[i] = MSE(Z[-1].flatten(), y)

    # Calculate the loss using backprop
    gradients = backprop(W, A, Z, y)

    # Update the parameters
    for j in range(len(W)):
        W[j] -= lr * gradients[j]

t1 = time.time()  # Time after training
print("Elapsed time: %.2fs" % (t1 - t0))

plot_results(X[:, 0], y, Z, losses)

#Training a Model With TensorFlow

# Set environment variables and import TensorFlow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow messages
import tensorflow as tf

# Check the TensorFlow version and whether it is using a GPU
print("TensorFlow version: %s" % tf.__version__)
gpu_list = tf.config.list_physical_devices("GPU")
if len(gpu_list) > 0:
    print("TensorFlow is using a GPU.")
else:
    print("TensorFlow is only using a CPU.")

# TensorFlow by default uses float32 and expects data in that format.
# Since our data matrices X and y are in np.float64 format, we specify float64 here.
tf.keras.backend.set_floatx("float64")

# 1. Define the model structure
model_tf = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(
            units=200,
            activation="relu",
            use_bias=False,
            kernel_initializer="random_normal",
        ),
        tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=False,
            kernel_initializer="random_normal",
        ),
    ]
)

# 2. Define optimizer for the TensorFlow model
lr = 0.001  # Learning rate
optimizer_tf = tf.keras.optimizers.SGD(learning_rate=lr)
training_batch_size = len(y)

# 3. Define loss function for the TensorFlow model
loss_tf = tf.keras.losses.MeanSquaredError()

# Attach the optimizer and loss function to the model
model_tf.compile(optimizer=optimizer_tf, loss=loss_tf)

# 4. Fit the TensorFlow model on training data
t0 = time.time()

history = model_tf.fit(
    X,
    y,
    epochs=M,
    batch_size=training_batch_size,
    verbose=0,  # No progress bar
)

t1 = time.time()
print("Elapsed time: %.2fs" % (t1 - t0))

def tf_transitions(model, X):
    """
    Function that returns the result of each layer in the model

    INPUT:
        model: TensorFlow model
        X: nxd data matrix; each row is an input vector

    OUTPUTS:
        Z: List of matrices (of length L) that stores
           result of transition function at each layer
    """
    Z = [X]
    
    for layer in model.layers:
        X = layer(X).numpy()
        Z.append(X)

    return Z

losses = history.history["loss"]
Z = tf_transitions(model_tf, X)
plot_results(X[:, 0], y, Z, losses)

