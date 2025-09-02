%load_ext autoreload

%%capture
from tqdm.notebook import tqdm
tqdm().pandas()

%autoreload 2

import math
import numpy as np

# Non-interactive matplotlib plots
%matplotlib inline
import matplotlib.pyplot as plt

# PyTorch and submodules
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision

# TensorFlow (suppress messages and allocate GPU memory)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"              # Suppress messages
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Allocate GPU memory
import tensorflow as tf

import sys
sys.path.append("/home/codio/workspace/.modules")
from helper import *

print("Python version:", sys.version.split(" ")[0])
print("PyTorch+CUDA version:", torch.__version__)
print("TensorFlow version:", tf.__version__)
print()

gpu_available_torch = torch.cuda.is_available()
if gpu_available_torch:
    print("PyTorch is using the GPU.")
    torch.cuda.manual_seed(1)  # Seed the random number generator
else:
    print("PyTorch is only using the CPU.")
    torch.manual_seed(1)  # Seed the random number generator

gpu_list = tf.config.list_physical_devices("GPU")
if len(gpu_list) > 0:
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is only using the CPU.")

# Setup the training set and test set
trainset = datasets.MNIST(
    root=".", train=True, download=True, transform=transforms.ToTensor()
)
testset = datasets.MNIST(
    root=".", train=False, download=True, transform=transforms.ToTensor()
)

# Setup dataloader that stacks small batches of images (512 per batch)
# into PyTorch Tensors for easier training and evaluation.
trainloader = torch.utils.data.DataLoader(
    trainset, shuffle=True, drop_last=True, batch_size=512, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, shuffle=True, drop_last=False, batch_size=512, num_workers=2
)

# The following line gets us small batch of data;
# X is a tensor of size (512, 1, 28, 28) that contains a batch of images,
# and y is a tensor of size (512) that contains the labels in X.
X, y = next(iter(trainloader))


def visualize_data(X):
    img_grid = torchvision.utils.make_grid(X[:5], padding=10)
    img_grid = img_grid.numpy()
    plt.figure()
    plt.imshow(np.transpose(img_grid, (1, 2, 0)), interpolation="nearest")
    return

visualize_data(X)

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        """
        Initializes a block consisting of two layers:
            1. Convolutional layer
            2. Batch normalization layer
        """
        super().__init__()

        # Define block parameters
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        # Define convolutional layer and batch normalization layer
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channel)
        
        return

    def forward(self, x):
        """
        Passes the input image through a convolutional layer,
        followed by a batch normalization layer and ReLU transition.
        """
        out = F.relu(self.bn(self.conv(x)))
        return out

Block1 = Block(1, 10, 3, 1)
out = Block1(X)
print(out.shape)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First block takes in the image of channel 1.
        # Filter specification:
        # Num filters=16, kernel size 3, stride 1
        self.block1 = Block(1, 16, 3, 1)

        # Filter specification:
        # num_filters=32, kernel_size=3, stride=1
        self.block2 = Block(16, 32, 3, 1)

        # Filter specification:
        # num_filters=64, kernel_size=3, stride=1
        self.block3 = Block(32, 64, 3, 1)

        # Filter specification:
        # num_filters=128, kernel_size=3, stride=1
        self.block4 = Block(64, 128, 3, 1)

        # This is the average pooling layer.
        # This is applied to the output of the last convolutional layer.
        # Essentially, it averages feature maps spatially to a single number.
        # For instance, if the output of the last conv layer is of size (128, 15, 4, 4),
        # then the following layer will average the 4x4 array into a single number,
        # so the output of applying this layer would have size (128, 15, 1, 1).
        # This operation vectorize the feature maps so that we can have a vector
        # that can be passed into a simple linear layer for classification.
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Create a linear layer:
        # the dataset has 10 classes, and
        # the model should output 10 belief values.
        self.fc = nn.Linear(128, 10)

        return

    def forward(self, x):
        batch_size = x.size(0)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        # Use `.squeeze()` method to remove unnecessary dimensions of size 1.
        # For example, if `X` is of shape (128, 128, 1, 1),
        # then the shape of `X.squeeze()` would be (128, 128).
        out = self.avgpool(out).squeeze()
        out = self.fc(out)
        
        return out

# Create a CNN model
model = ConvNet()

# Equip model with CUDA code if GPU is available
if gpu_available_torch:
    model = model.cuda()

def model_test1():
    """
    Check that block2, block3, block4 are instances of Block
    """
    model = ConvNet()

    # Create CUDA model when GPU is available
    if gpu_available_torch:
        model = model.cuda()
    
    isblock2 = isinstance(model.block2, Block)
    isblock3 = isinstance(model.block3, Block)
    isblock4 = isinstance(model.block3, Block)
    
    return isblock2 and isblock3 and isblock4


def model_test2():
    """
    Check specifications for block2
    """
    model = ConvNet()

    # Create CUDA model when GPU is available
    if gpu_available_torch:
        model = model.cuda()

    # Check input channel
    in_channel_check = model.block2.in_channel == 16

    # Check output channel
    out_channel_check = model.block2.out_channel == 32

    # Check kernel size
    kernel_size_check = model.block2.kernel_size == 3

    # Check stride size
    stride_check = model.block2.stride == 1
    
    return in_channel_check and out_channel_check and kernel_size_check and stride_check


def model_test3():
    """
    Check specifications for block3
    """
    model = ConvNet()

    # Create CUDA model when GPU is available
    if gpu_available_torch:
        model = model.cuda()

    # Check input channel
    in_channel_check = model.block3.in_channel == 32

    # Check output channel
    out_channel_check = model.block3.out_channel == 64

    # Check kernel size
    kernel_size_check = model.block3.kernel_size == 3

    # Check stride size
    stride_check = model.block3.stride == 1
    
    return in_channel_check and out_channel_check and kernel_size_check and stride_check


def model_test4():
    """
    Check specifications for block4
    """
    model = ConvNet()

    # Create CUDA model when GPU is available
    if gpu_available_torch:
        model = model.cuda()

    # Check input channel
    in_channel_check = model.block4.in_channel == 64

    # Check the output channel
    out_channel_check = model.block4.out_channel == 128

    # Check the kernel size
    kernel_size_check = model.block4.kernel_size == 3

    # Check the stride size
    stride_check = model.block4.stride == 1
    
    return in_channel_check and out_channel_check and kernel_size_check and stride_check


runtest(model_test1, "model_test1")
runtest(model_test2, "model_test2")
runtest(model_test3, "model_test3")
runtest(model_test4, "model_test4")

loss_fn = nn.CrossEntropyLoss()

def loss_fn_test1():
    """
    Check that the loss is a scalar
    """
    num_classes = 10                                  # Number of classes
    num_examples = 5                                  # Number of examples
    logits = torch.ones((num_examples, num_classes))  # Simulate model belief
    y = torch.zeros(num_examples).long()              # Tensor of zeros
    loss = loss_fn(logits, y)                         # Calculate the loss
    return loss.size() == torch.Size([])


def loss_fn_test2():
    """
    Check that if the model has equal belief for each class (i.e., P(y|x) is uniform),
    then the negative log-likelihood should be -log(1 /num_classes) = log(num_classes).
    """
    num_classes = 10  # Number of classes
    num_examples = 1  # Number of examples

    # Simulate model belief that each class is equally likely,
    # and then calculate your loss.
    logits = torch.ones((num_examples, num_classes))
    y = torch.zeros(num_examples).long()
    loss = loss_fn(logits, y)

    return loss.item() == torch.log(torch.Tensor([num_classes])).item()


def loss_fn_test3():
    """
    Check that your loss is close to our loss
    """
    num_classes = 10  # Number of classes
    num_examples = 5  # Number of examples

    # Simulate model belief that each class is equally likely,
    # and then calculate your loss and our loss.
    logits = torch.rand((num_examples, num_classes))
    y = torch.zeros(num_examples).long()
    loss = loss_fn(logits, y)
    loss_grader = loss_fn_grader(logits, torch.zeros(num_examples).long())

    return (torch.abs(loss - loss_grader)).item() < 1e-5


runtest(loss_fn_test1, "loss_fn_test1")
runtest(loss_fn_test2, "loss_fn_test2")
runtest(loss_fn_test3, "loss_fn_test3")

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def optimizer_test1():
    """
    Check that you implemented the correct type of optimizer
    """
    return isinstance(optimizer, torch.optim.SGD)

runtest(optimizer_test1, "optimizer_test1")

def train(model, optimizer, loss_fn, trainloader):
    """
    Trains a model using the optimizer, loss function, and training data in trainloader.

    Input:
        model: ConvNet model
        optimizer: Optimizer for the model
        loss_fn: Loss function
        trainloader: Dataloader with training data

    Output:
        running_loss: Average loss of the network
    """
    model.train()       # Put the model in training mode
    running_loss = 0.0  # Keep track of the running loss

    # Iterate through trainloader; for each interation,
    # you will get a batch of images X and labels y.
    for i, (X, y) in enumerate(trainloader):

        # Move data to CUDA GPU to accelerate training
        if gpu_available_torch:
            X, y = X.cuda(), y.cuda()

        # Zero the parameter gradient
        optimizer.zero_grad()
        #1: Do a forward pass to get the logits
        logits = model(X)
        #2: Evaluate the loss
        loss = loss_fn(logits, y)
        #3: Do a backward pass using .backward()
        loss.backward()

        optimizer.step()             # Update the parameters
        running_loss += loss.item()  # Update the loss
    
    return running_loss / len(trainloader)

def train_test1():
    """
    Check that your losses are close to our losses
    """
    from copy import deepcopy

    model = ConvNet()
    if gpu_available_torch:
        model = model.cuda()
    model2 = deepcopy(model)

    optimizer = create_optimizer(model)
    optimizer2 = create_optimizer(model2)

    # Sample one batch of training examples
    X, y = next(iter(trainloader))

    # Create a dataset using the small batch
    small_set = torch.utils.data.TensorDataset(X, y)

    # Create a dataloader for the small_set
    loader = torch.utils.data.DataLoader(small_set, batch_size=128)

    # Run your train function twice (i.e., update your model twice)
    # essentially updates the model twice.
    loss = train(model, optimizer, loss_fn, loader)
    loss2 = train(model, optimizer, loss_fn, loader)

    # Run our train function twice (i.e., update our model twice)
    loss_grader = train_grader(model2, optimizer2, loss_fn, loader)
    loss_grader_2 = train_grader(model2, optimizer2, loss_fn, loader)

    return ((loss - loss_grader) ** 2 + (loss2 - loss_grader_2) ** 2) < 1e-5


runtest(train_test1, "train_test1")

num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    running_loss = train(model, optimizer, loss_fn, trainloader)
    print(f"Running Loss (Epoch {epoch + 1}/{num_epochs}): {running_loss:.4f}")

def pred(logits):
    """
    Calculates the predictions of the ConvNet using the logits.

    Input:
        logits: nxC output matrix of the network, where n is the
                number of data points and C is the number of labels

    Output:
        prediction: n-dimensional vector of predictions
    """
  
    prediction = torch.argmax(logits, dim=1)
  
    return prediction

def pred_test1():
    """
    Check that your predictions match the expected output
    """
    logits = torch.Tensor([[0, 1], [2, -1]])    # Generate some beliefs
    prediction = pred(logits)                   # Make your predictions
    expected_output = torch.LongTensor([1, 0])  # Expected output: [1, 0]
    return torch.equal(prediction, expected_output)


runtest(pred_test1, "pred_test1")

total = 0.0
correct = 0.0
model.eval()
with torch.no_grad():
    for X, y in testloader:
        if gpu_available_torch:
            X, y = X.cuda(), y.cuda()
        logits = model(X)

        prediction = pred(logits)
        total += X.size(0)
        correct += (prediction == y).sum().item()

print(f"PyTorch Accuracy: {100 * (correct / total):.2f}%")

test_iterator = iter(testloader)
X_test, y_test = next(test_iterator)
X_test, y_test = X_test[:5], y_test[:5]
visualize_data(X_test)

model.eval()
with torch.no_grad():
    if gpu_available_torch:
        X_test = X_test.cuda()
    logits = model(X_test)
    prediction = pred(logits)

print("PyTorch predictions for the first five test images:", prediction.tolist())

#TensorFlow Approach

def plot_imgs(images, labels=None):
    subplots_x = int(math.ceil(len(images) / 5))
    plt.figure(figsize=(10, 2 * subplots_x))
    for i in range(min(len(images), subplots_x * 5)):
        plt.subplot(subplots_x, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        if labels is not None:
            plt.xlabel(labels[i])
    plt.show()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
plot_imgs(x_train[:5], y_train[:5])

# TensorFlow downloads the images raw, so the
# pixel values are normalized between 0 and 1.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert datasets of shape (num_points, 28, 28) to (num_points, 28, 28, 1)
x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

def block_tf(filters, kernel_size, stride):
    return [
        tf.keras.layers.Conv2D(
            filters, kernel_size, strides=(stride, stride), activation=None
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ]


# Define the network
model_tf = tf.keras.models.Sequential(
    [tf.keras.Input((28, 28, 1))]
    + block_tf(16, 3, 1)
    + block_tf(32, 3, 1)
    + block_tf(64, 3, 1)
    + block_tf(128, 3, 1)
    + [
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation=None),
    ]
)

model_tf.summary()

# Define the loss function and optimizer
loss_fn_tf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer_tf = tf.keras.optimizers.SGD(learning_rate=0.1)

# Compile and fit the TensorFlow model
model_tf.compile(optimizer=optimizer_tf, loss=loss_fn_tf, metrics=["accuracy"])
model_tf.fit(x_train, y_train, epochs=5, batch_size=32);

logits_tf = model_tf(x_test)
predictions_tf = tf.argmax(logits_tf, axis=1)
accuracy_tf = tf.reduce_mean(tf.cast(predictions_tf == y_test, tf.float64)).numpy()
print(f"TensorFlow Accuracy: {100 * accuracy_tf:.2f}%")
plot_imgs(np.squeeze(x_test * 255, axis=-1)[:5], predictions_tf.numpy()[:5])

img_idx = 17
img_array = x_test[img_idx][np.newaxis, ...]

logits_tf2 = model_tf(img_array)
predictions_tf2 = tf.argmax(logits_tf2, axis=1)

last_conv_layer_name = [
    layer.name for layer in model_tf.layers if isinstance(layer, tf.keras.layers.Conv2D)
][-1]
heatmap = make_gradcam_heatmap(img_array, model_tf, last_conv_layer_name)
display_gradcam(
    np.uint(255 * img_array)[0], heatmap, prediction=int(predictions_tf2.numpy()[0])
)
  
