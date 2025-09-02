import numpy as np
import pandas as pd
from numpy.matlib import repmat
from scipy.io import loadmat
import time

# Interactive plots
%matplotlib ipympl
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/codio/workspace/.modules")
from helper import *

print("Python version:", sys.version.split(" ")[0])

def spiraldata(N=300):
    """
    Generate spiral datasets for model training and testing
    """
    r = np.linspace(1, 2 * np.pi, N)  # Vector of "radius" values

    # Generate a curves that draw circles with increasing radius
    xTr1 = np.array([np.sin(2.0 * r) * r, np.cos(2 * r) * r]).T  
    xTr2 = np.array([np.sin(2.0 * r + np.pi) * r, np.cos(2 * r + np.pi) * r]).T

    # Create spiral data points with noise
    xTr = np.concatenate([xTr1, xTr2], axis=0)
    xTr = xTr + np.random.randn(xTr.shape[0], xTr.shape[1]) * 0.2
    
    # Create labels +1 (first half) and -1 (second half)
    yTr = np.concatenate([np.ones(N), -1 * np.ones(N)])

    # Sample alternating values to generate training and test sets
    xTr = xTr[1::2, :]
    yTr = yTr[1::2]
    xTe = xTr[::2, :]
    yTe = yTr[::2]

    return xTr, yTr, xTe, yTe


xTrSpiral, yTrSpiral, xTeSpiral, yTeSpiral = spiraldata(150)
print(f"Number of training points in spiral dataset: {xTrSpiral.shape[0]}")
print(f"Number of test points in spiral dataset:     {xTeSpiral.shape[0]}")
print(f"Number of features in spiral dataset:        {xTrSpiral.shape[1]}")

plt.figure()
plt.scatter(
    xTrSpiral[yTrSpiral == 1, 0],  # x-coordinate
    xTrSpiral[yTrSpiral == 1, 1],  # y-coordinate
    label="$+1$",
    c="blue",
    marker="o",
)
plt.scatter(
    xTrSpiral[yTrSpiral == -1, 0],  # x-coordinate
    xTrSpiral[yTrSpiral == -1, 1],  # y-coordinate
    label="$-1$",
    c="red",
    marker="x",
)
plt.title("Training Spiral Dataset")
plt.legend(loc="upper left")
plt.show()

# Load some binary test data (labels are -1 and +1)
data = loadmat("ion.mat")

# Load the training data
xTrIon = data["xTr"].T
yTrIon = data["yTr"].flatten()

# Load the test data
xTeIon = data["xTe"].T
yTeIon = data["yTe"].flatten()

print(f"Number of test points in the ION dataset: {xTeIon.shape[0]}")
print(f"Number of training points in ION dataset: {xTrIon.shape[0]}")
print(f"Number of features in the ION dataset:    {xTrIon.shape[1]}")
print(f"Shape of the ION training data matrix:    {xTrIon.shape}")

print()
print("The ION training data matrix looks like this:")
print()

TrIon_for_display = np.concatenate([yTrIon[:, None], xTrIon], axis=1)
TrIon_for_display = TrIon_for_display[TrIon_for_display[:, 0].argsort()]

display(
    pd.DataFrame(
        data=TrIon_for_display,
        columns=["$y$"]
        + [rf"$[\mathbf{{x}}]_{ {i+1} }$" for i in range(xTrIon.shape[1])],
    ).round(2)
)

def sqimpurity(yTr):
    """
    Computes the squared loss impurity (variance) of the labels.

    Input:
        yTr: n-dimensional vector of labels

    Output:
        impurity: Weighted variance/squared loss impurity of the labels
    """
    (N,) = yTr.shape
    assert N > 0     # Must have at least one sample
    impurity = None  # Initialize variable to be returned
    impurity = np.sum((yTr - 1/N * np.sum(yTr))**2)
    
    return impurity

def sqimpurity_test1():
    """
    Check that impurity is a scalar
    """
    yTr = np.random.randn(100)  # Generate random training labels
    impurity = sqimpurity(yTr)  # Compute impurity
    return np.isscalar(impurity)


def sqimpurity_test2():
    """
    Check that impurity is non-negative
    """
    yTr = np.random.randn(100)  # Generate random training labels
    impurity = sqimpurity(yTr)  # Compute impurity
    return impurity >= 0


def sqimpurity_test3():
    """
    Check that impurity is close to zero when labels are homogeneous
    """
    yTr = np.ones(100)          # Generate an all-one vector as labels
    impurity = sqimpurity(yTr)  # Compute impurity
    return np.isclose(impurity, 0)


def sqimpurity_test4():
    """
    Check that impurity is sum of squares when data has mean of zero
    """
    yTr = np.arange(-5, 6)      # Generate a vector with mean zero
    impurity = sqimpurity(yTr)  # Compute impurity
    sum_of_squares = np.sum(yTr**2)
    return np.isclose(impurity, sum_of_squares)


def sqimpurity_test5():
    """
    Check that your impurity is close to our impurity
    """
    np.random.seed(1)           # Set random seed for consistency
    yTr = np.random.randn(100)  # Generate random training labels
    your_impurity = sqimpurity(yTr)
    our_impurity = sqimpurity_grader(yTr)
    return np.isclose(your_impurity, our_impurity)


runtest(sqimpurity_test1, "sqimpurity_test1")
runtest(sqimpurity_test2, "sqimpurity_test2")
runtest(sqimpurity_test3, "sqimpurity_test3")
runtest(sqimpurity_test4, "sqimpurity_test4")
runtest(sqimpurity_test5, "sqimpurity_test5")

size = 10
y = np.ones(size)
fraction_pos, impurities = [], []

for i in range(size):
    fraction_pos.append(sum([y == 1]) / size)
    impurities.append(sqimpurity(y))
    y[i] = -1

fraction_pos.append(sum([y == 1]) / size)
impurities.append(sqimpurity(y))

display(pd.DataFrame(data={"fraction_pos": fraction_pos, "impurity": impurities}))

plt.figure()
plt.plot(fraction_pos, impurities)
plt.title("Squared Loss Impurity of Labels")
plt.xlabel("Fraction of Positive Labels ($1 -$ Fraction of Negative Labels)", fontsize=12)
plt.ylabel("Squared Loss", fontsize=12)
plt.grid()
plt.show()

def sqsplit(xTr, yTr):
    """
    Finds the best feature, cut value, and impurity for a split of (xTr, yTr) based on squared loss impurity.

    Input:
        xTr: n x d matrix of data points
        yTr: n-dimensional vector of labels

    Output:
        feature:  index of the best cut's feature (keep in mind this is 0-indexed)
        cut:      cut-value of the best cut
        bestloss: squared loss impurity of the best cut
    """
    n, d = xTr.shape
    assert d > 0  # Must have at least one dimension
    assert n > 1  # Must have at least two samples

    bestloss = np.inf
    feature = np.inf
    cut = np.inf

    for f in range(d):
        sort = xTr[:, f].argsort()
        sortIdX = xTr[sort, f]
        sortIdY = yTr[sort]

        for i in range(n):
            if i + 1 == n:
                continue
                
            if sortIdX[i + 1] > sortIdX[i]:
                t = (sortIdX[i] + sortIdX[i + 1]) / 2
                Sl = sortIdY[:i + 1]
                Sr = sortIdY[i + 1:]
                Is = sqimpurity(Sl) + sqimpurity(Sr)
                
                if Is < bestloss:
                    feature = f
                    cut = t
                    bestloss = Is

    #print(bestloss)
    
    return feature, cut, bestloss

t0 = time.time()
fid, cut, loss = sqsplit(xTrIon, yTrIon)
t1 = time.time()

print(f"Elapsed time: {t1-t0:.2f} seconds")
print("The best split is on feature 2 on value 0.304.")
print("Your tree split on feature %i on value %.3f.\n" % (fid, cut))


def sqsplit_test1():
    a = np.isclose(sqsplit(xor4, yor4)[2] / len(yor4), 0.25)
    b = np.isclose(sqsplit(xor3, yor3)[2] / len(yor3), 0.25)
    c = np.isclose(sqsplit(xor2, yor2)[2] / len(yor2), 0.25)
    return a and b and c


def sqsplit_test2():
    x = np.array(range(1000)).reshape(-1, 1)
    y = np.hstack([np.ones(500), -1 * np.ones(500)]).T
    _, cut, _ = sqsplit(x, y)
    return cut <= 500 or cut >= 499


def sqsplit_test3():
    fid, cut, loss = sqsplit(xor5, yor5)
    # cut should be 0.5 but 0 is also accepted
    return fid == 0 and (cut >= 0 or cut <= 1) and np.isclose(loss / len(yor5), 2 / 3)


runtest(sqsplit_test1, "sqsplit_test1")
runtest(sqsplit_test2, "sqsplit_test2")
runtest(sqsplit_test3, "sqsplit_test3")

class TreeNode:
    """
    Class for instantiating decision trees
    """

    def __init__(self, left, right, feature, cut, prediction):
        node_or_leaf_args = [left, right, feature, cut]
        all_args_none = all([arg == None for arg in node_or_leaf_args])
        no_args_none = all([arg != None for arg in node_or_leaf_args])

        # Check that all arguments are None or no arguments are None
        assert all_args_none or no_args_none

        # Check that prediction is not None when all arguments are None
        if all_args_none:
            assert prediction is not None
        
        # Check that prediction is None when no arguments are None
        if no_args_none:
            assert prediction is None

        self.left = left
        self.right = right
        self.feature = feature
        self.cut = cut
        self.prediction = prediction

  root = TreeNode(None, None, None, None, 0)

left_leaf = TreeNode(None, None, None, None, 1)
right_leaf = TreeNode(None, None, None, None, 2)
root2 = TreeNode(left_leaf, right_leaf, 0, 1, None)

def cart(xTr, yTr):
    """
    Builds a CART tree.

    Input:
        xTr: n x d matrix of data
        yTr: n-dimensional vector

    Output:
        tree: root of decision tree
    """
    n, d = xTr.shape
    node = None  # Intialize variable to be returned

    if np.all(yTr == yTr[0]):
        prediction = yTr[0]  
        return TreeNode(None, None, None, None, prediction)
    
    if np.all(np.isclose(xTr, xTr[0])):
        prediction = np.mean(yTr)
        return TreeNode(None, None, None, None, prediction)

    feature, cut, loss = sqsplit(xTr, yTr)

    left_idx = xTr[:, feature] <= cut
    right_idx = xTr[:, feature] > cut

    left_xTr, left_yTr = xTr[left_idx], yTr[left_idx]
    right_xTr, right_yTr = xTr[right_idx], yTr[right_idx]

    left_subtree = cart(left_xTr, left_yTr)
    right_subtree = cart(right_xTr, right_yTr)
    
    node = TreeNode(left_subtree, right_subtree, feature, cut, None)

    return node

def cart_test1():
    t = cart(xor4, yor4)
    return DFSxor(t)


def cart_test2():
    """
    Check that every label appears exactly once in the tree
    """
    y = np.random.rand(16)
    t = cart(xor4, y)
    yTe = DFSpreds(t)[:]
    y.sort()
    yTe.sort()
    return np.all(np.isclose(y, yTe))


def cart_test3():
    xRep = np.concatenate([xor2, xor2])
    yRep = np.concatenate([yor2, 1 - yor2])
    t = cart(xRep, yRep)
    return DFSxorUnsplittable(t)


def cart_test4():
    """
    Check child nodes, features, and predictioins
    """
    X = np.ones((5, 2))  # Create a dataset with identical examples
    y = np.ones(5)

    # On this dataset, your cart algorithm should return
    # a single leaf node with prediction equal to 1.
    tree = cart(X, y)

    # Check that the tree has no children
    children_check = (tree.left is None) and (tree.right is None)

    # Make sure tree does not cut any feature and at any value
    feature_check = (tree.feature is None) and (tree.cut is None)

    # Check tree's prediction
    prediction_check = np.isclose(tree.prediction, 1)
    
    return children_check and feature_check and prediction_check


def cart_test5():
    """
    Check whether you set tree.feature and tree.cut to something
    """
    X = np.arange(4).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    tree = cart(X, y)  # Your cart algorithm should generate one split
    feature_check = tree.feature is not None
    cut_check = tree.cut is not None
    return feature_check and cut_check


runtest(cart_test1, "cart_test1")
runtest(cart_test2, "cart_test2")
runtest(cart_test3, "cart_test3")
runtest(cart_test4, "cart_test4")
runtest(cart_test5, "cart_test5")

def evaltree(tree, xTe):
    """
    Evaluates testing points in xTe using decision tree root.

    Input:
        tree: TreeNode decision tree
        xTe: mxd matrix of data points

    Output:
        preds: m-dimensional vector of predictions
    """
    m = xTe.shape[0]
    preds = np.empty(m)  # Initialize variable to be returned

    for t in range(m):
        tr = tree
        while(True):
            if tr.left is None or tr.right is None:
                preds[t] = tr.prediction
                break
            elif xTe[t, tr.feature] <= tr.cut:
                tr = tr.left
            else:
                tr = tr.right

    return preds

# Fit a tree on the ION training data
t0 = time.time()
root = cart(xTrIon, yTrIon)
t1 = time.time()

# Calculate training error and test error
tr_err = np.mean((evaltree(root, xTrIon) - yTrIon) ** 2)
te_err = np.mean((evaltree(root, xTeIon) - yTeIon) ** 2)

print(f"Elapsed time: {t1 - t0:.4f} seconds")
print(f"Training MSE: {tr_err:.4f}")
print(f"Test MSE:     {te_err:.4f}\n")


def evaltree_test1():
    """
    Check that shuffling and expanding the data doesn't affect the predictions
    """
    t = cart(xor4, yor4)
    xor4te = xor4 + (np.sign(xor4 - 0.5) * 0.1)
    inds = np.arange(16)
    np.random.shuffle(inds)
    preds = evaltree(t, xor4te[inds, :])
    truth = yor4[inds]
    return np.all(np.isclose(preds, truth))


def evaltree_test2():
    """
    Check that a custom tree evaluates correctly
    """
    a = TreeNode(None, None, None, None, 1)
    b = TreeNode(None, None, None, None, -1)
    c = TreeNode(None, None, None, None, 0)
    d = TreeNode(None, None, None, None, -1)
    e = TreeNode(None, None, None, None, -1)
    x = TreeNode(a, b, 0, 10, None)
    y = TreeNode(x, c, 0, 20, None)
    z = TreeNode(d, e, 0, 40, None)
    t = TreeNode(y, z, 0, 30, None)
    preds = evaltree(t, np.array([[45, 35, 25, 15, 5]]).T)
    truth = np.array([-1, -1, 0, -1, 1])
    return np.all(np.isclose(preds, truth))


runtest(evaltree_test1, "evaltree_test1")
runtest(evaltree_test2, "evaltree_test2")

t0 = time.time()
root = cart(xTrIon, yTrIon)
t1 = time.time()

tr_err = np.mean((evaltree(root, xTrIon) - yTrIon) ** 2)
te_err = np.mean((evaltree(root, xTeIon) - yTeIon) ** 2)

print(f"Elapsed time: {t1 - t0:.4f} seconds")
print(f"Training MSE: {tr_err:.4f}")
print(f"Test MSE:     {te_err:.4f}")

def visclassifier(fun, xTr, yTr, w=None, b=0):
    """
    Visualize a classifier and its decision boundary.
    Define the symbols and colors we'll use in the plots later.
    """
    yTr = np.array(yTr).flatten()

    # Get the unique values from labels array
    classvals = np.unique(yTr)
    
    # Return 300 evenly spaced numbers over this interval
    res = 300
    xrange = np.linspace(min(xTr[:, 0]) - 0.5, max(xTr[:, 0]) + 0.5, res)
    yrange = np.linspace(min(xTr[:, 1]) - 0.5, max(xTr[:, 1]) + 0.5, res)

    # Repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    # Test all of these points on the grid
    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T
    testpreds = fun(xTe)

    # Reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    Z[0,0] = 1  # Optional: scale the colors correctly

    plt.figure()

    symbols = ["x", "o"]
    labels = ["$-1$", "$+1$"]
    mycolors = [[1, 0.5, 0.5], [0.5, 0.5, 1]]

    # Fill in the contours for these predictions
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    # Create x's and o's for training set
    for idx, c in enumerate(classvals):
        plt.scatter(
            xTr[yTr == c, 0],     # x-coordinate
            xTr[yTr == c, 1],     # y-coordinate
            label=labels[idx],    # Label (-1 or +1)
            marker=symbols[idx],  # Marker symbol ("x" or "o")
            color="black",
        )

    if w is not None:
        w = np.array(w).flatten()
        alpha = -1 * b / (w**2).sum()
        plt.quiver(w[0] * alpha, w[1] * alpha, w[0], w[1], linewidth=2, color=[0, 1, 0])

    plt.axis("tight")
    plt.legend(loc="upper left")
    # Show figure and blocks
    plt.show()

# Fit regression tree on spiral training data
tree = cart(xTrSpiral, yTrSpiral)

# Calculate training and test error
train_error = np.mean(np.sign(evaltree(tree, xTrSpiral)) != yTrSpiral)
test_error = np.mean(np.sign(evaltree(tree, xTeSpiral)) != yTeSpiral)
print(f"Training Error: {100 * train_error:.2f}%")
print(f"Test Error:     {100 * test_error:.2f}%")

# Visualize the classifier and decision boundary
visclassifier(lambda X: evaltree(tree, X), xTrSpiral, yTrSpiral)

# Initialize empty array of width 2 to store training points
xTrain = np.empty((0, 2))

# Initialize empty array of width 1 to store training labels
yTrain = np.empty((0, 1))


def onclick_cart(event):
    """
    Visualize boosted classifier by adding new points
    """
    global xTrain, yTrain

    # Shift+click to add a negative point.
    # Click to add a positive point.
    if event.key == "shift":
        label = -1
    else:
        label = 1

    # Create position vector for new point
    pos = np.array([[event.xdata, event.ydata]])

    # Add new point to training data
    xTrain = np.concatenate((xTrain, pos), axis=0)
    yTrain = np.append(yTrain, label)

    # Get the class values from training labels
    classvals = np.unique(yTrain)
    
    # Return 300 evenly spaced numbers over interval [0, 1]
    res = 300
    xrange = np.linspace(0, 1, res)
    yrange = np.linspace(0, 1, res)

    # Repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    # Get decision tree
    tree = cart(xTrain, np.array(yTrain).flatten())
    fun = lambda X: evaltree(tree, X)

    # Test all of these points on the grid
    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T
    testpreds = fun(xTe)

    # Reshape test predictions to match the grid size
    Z = testpreds.reshape(res, res)
    Z[0, 0] = 1  # Optional: scale the colors correctly

    # Plot configuration
    plt.cla()                                  # Clear axes
    plt.xlim((0, 1))                           # x-axis limits
    plt.ylim((0, 1))                           # y-axis limits
    labels = ["$-1$", "$+1$"]                  # Labels for training points
    symbols = ["x", "o"]                       # "x" for -1 and "o" for +1 (training)
    mycolors = [[1, 0.5, 0.5], [0.5, 0.5, 1]]  # Red for -1 and blue for +1 (predictions)

    if len(classvals) == 1 and 1 in classvals:
        # Plot blue prediction contours because only +1 points have been added
        plt.contourf(pixelX, pixelY, np.sign(Z), colors=[[0.5, 0.5, 1], [0.5, 0.5, 1]])
        
        # Plot points with "o" markers because only +1 points have been added
        for idx, c in enumerate(classvals):
            plt.scatter(
                xTrain[yTrain == c, 0],  # x-coordinate of training point
                xTrain[yTrain == c, 1],  # y-coordinate of training point
                label="$+1$",            # Label of training point
                marker="o",              # Marker of training point
                color="black",           # Color of training point
            )
    else:
        # Plot prediction contours: red for -1 and blue for +1
        plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

        # Plot training data: "x" for -1 and "o" for +1
        for idx, c in enumerate(classvals):
            plt.scatter(
                xTrain[yTrain == c, 0],  # x-coordinate of training point
                xTrain[yTrain == c, 1],  # y-coordinate of training point
                label=labels[idx],       # Label of training point (-1 or +1)
                marker=symbols[idx],     # Marker of training point ("x" or "o")
                color="black",           # Color of training point
            )

    plt.title("Click: Positive Point, Shift+Click: Negative Point")
    plt.legend(loc="upper left")
    plt.show()


# Interactive demo
print("Please keep in mind:")
print("1. You must run (or rerun) this cell right before interacting with the plot.")
print("2. Start the interactive demo by clicking the grid to add a positive point.")
print("3. Click to add a positive point or shift+click to add a negative point.")
print("4. You may notice a slight delay when adding points to the visualization.")
fig = plt.figure()
plt.title("Start by Clicking the Grid to Add a Positive Point")
plt.xlim(0, 1)
plt.ylim(0, 1)
cid = fig.canvas.mpl_connect("button_press_event", onclick_cart)

#Scikit-Learn implementation

from sklearn.tree import DecisionTreeRegressor, plot_tree

t0 = time.time()
tree = DecisionTreeRegressor(
    criterion="squared_error",  # Impurity function = Mean Squared Error (squared loss)
    splitter="best",            # Take the best split
    max_depth=None,             # Expand the tree to the maximum depth possible
)
tree.fit(xTrSpiral, yTrSpiral)
t1 = time.time()

tr_err = np.mean((tree.predict(xTrSpiral) - yTrSpiral) ** 2)
te_err = np.mean((tree.predict(xTeSpiral) - yTeSpiral) ** 2)

print(f"Elapsed time: {t1 - t0:.4f} seconds")
print(f"Training MSE: {tr_err:.4f}")
print(f"Test MSE:     {te_err:.4f}")

fig, ax = plt.subplots(figsize=(20, 20))
_ = plot_tree(
    tree,
    ax=ax,
    precision=2,
    feature_names=[rf"$[\mathbf{{x}}]_{i+1}$" for i in range(2)],
    filled=True,
)
