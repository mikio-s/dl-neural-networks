%load_ext autoreload

%autoreload 2

import math
import numpy as np

# Non-interactive matplotlib plots
%matplotlib inline
import matplotlib.pyplot as plt

# PyTorch and submodules
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torchvision import datasets, transforms

# TensorFlow (with messages suppressed)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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

#PyTorch Approach

# Load data
review_train, review_test, vocab = load_data()

# Label 0: negative review
# Label 1: positive review
label_meaning = ["Negative", "Positive"]

print("Number of Training Reviews:", review_train.shape[0])
print("Number of Test Reviews:", review_test.shape[0])
print("Number of Words in the Vocabulary:", len(vocab))

print("A Positive Training Review:", review_train.iloc[0]["review"])
print("A Negative Training Review:", review_train.iloc[-1]["review"])

# Create a simple vocabulary
simple_vocab = {"learn": 0, "machine": 1, "learning": 2, "teach": 3}

# Create a simple sentence that will be converted into bag-of-words features
simple_sentence = " I learn machine learning to teach machine how to learn."

# Create a featurizer by passing in the vocabulary
simple_featurizer = generate_featurizer(simple_vocab)

# Transform the sentence to its bag-of-words features
simple_featurizer.transform([simple_sentence]).toarray()

bow_featurizer = generate_featurizer(vocab)

X_train = torch.Tensor(
    bow_featurizer.transform(review_train["review"].values).toarray()
)
y_train = torch.LongTensor(review_train["label"].values.flatten())

X_test = torch.Tensor(bow_featurizer.transform(review_test["review"].values).toarray())
y_test = torch.LongTensor(review_test["label"].values.flatten())

# Generate PyTorch datasets
trainset = torch.utils.data.TensorDataset(X_train, y_train)
testset = torch.utils.data.TensorDataset(X_test, y_test)

# Generate PyTorch dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, drop_last=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=True, drop_last=False
)

class DAN(nn.Module):
    def __init__(self, vocab_size, embedding_size=32):
        super().__init__()

        # Create a word embedding of dimension embedding_size.
        # The variable self.embeds is now the matrix E,
        # where each column corresponds to the embedding of a word.
        self.embeds = torch.nn.Parameter(torch.randn(vocab_size, embedding_size))
        self.embeds.requires_grad_(True)
        
        # Add a final linear layer that computes the
        # 2D output from the averaged word embedding
        self.fc = nn.Linear(embedding_size, 2)

    def average(self, x):
        """
        This function takes in multiple inputs, stored in one tensor x.
        Each input is a bag-of-words representation of reviews.
        For each review, it retrieves the word embedding of each word in the
        review and averages them (weighted by the corresponding entry in x).

        Input:
            x: nxv torch Tensor such that each row corresponds
               to bag-of-words representation of a review

        Output:
            emb: nxd torch Tensor for the averaged review
        """
        emb = None

        # YOUR CODE HERE
        emb = torch.matmul(x, self.embeds)
        emb /= x.sum(axis=1, keepdims=True)
        # END OF YOUR CODE

        return emb

    def forward(self, x):
        """
        This function takes in a bag-of-words representation of reviews.
        It calls the self.average to get the averaged review and pass
        it through the linear layer to produce the model's belief.

        Input:
            x: nxv torch Tensor, where each row corresponds
               to bag-of-words representation of reviews

        Output:
            out: nx2 torch Tensor that corresponds to model belief of the input.
                 For instance, output[i][0] is is the model belief that the i-th
                 review is negative and output[i][1] is the model belief that
                 the i-th review is positive.
        """
        review_averaged = self.average(x)
        out = None
        out = self.fc(review_averaged)
        
        return out

  def average_test1():
    """
    Check that your average function output has the correct shape 
    """
    n = 10                                   # Number of reviews
    vocab_size = 5                           # Size of vocabulary
    embedding_size = 32                      # Size of word embedding
    model = DAN(vocab_size, embedding_size)  # Create the DAN
    X = torch.rand(n, vocab_size)
    output_size = model.average(X).shape
    return output_size[0] == n and output_size[1] == embedding_size


def average_test2():
    """
    First check that your average function output has the correct values
    """
    n = 10                                   # Number of reviews
    vocab_size = 3                           # Size of vocabulary
    embedding_size = 5                       # Size of word embedding
    model = DAN(vocab_size, embedding_size)  # Create the DAN

    # Generate a simple input with four rows
    X = torch.FloatTensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ]
    )

    # Given that the input has four rows, we know that the
    # first three rows correspond to the first three words.
    # The last row should be the average of the three words,
    # so the difference between the last row and the average
    # of the first three rows should be small.

    # Get the averaged reviews
    averaged_reviews = model.average(X)

    # Get the last row
    last_row = averaged_reviews[3]

    # Compute the average of the first three rows
    avg_first_rows = torch.mean(averaged_reviews[:3], dim=0)

    # Calculate the squared difference
    diff = torch.sum((avg_first_rows - last_row) ** 2).item()
    
    return diff < 1e-5


def average_test3():
    """
    Second check that your average function output has the correct values
    """
    n = 10                                   # Number of reviews
    vocab_size = 3                           # Size of vocabulary
    embedding_size = 5                       # Size of word embedding
    model = DAN(vocab_size, embedding_size)  # Create the DAN

    # Generate a simple input
    X = torch.FloatTensor([[1, 1, 1], [2, 2, 2]])

    # Since the 2nd review is a multiple of the first,
    # the two averaged review should be the same.

    # Get the averaged reviews
    averaged_reviews = model.average(X)

    # Compute the sum of squared differences
    diff = torch.sum((averaged_reviews[0] - averaged_reviews[1]) ** 2).item()

    return diff < 1e-5


def forward_test1():
    """
    Check that your forward function output has the correct shape
    """
    n = 10                                   # Number of reviews
    vocab_size = 5                           # Size of vocabulary
    embedding_size = 32                      # Size of word embedding
    model = DAN(vocab_size, embedding_size)  # Create the DAN
    X = torch.rand(n, vocab_size)            # Generate data

    # Call your forward function and get output size
    output_size = model(X).shape

    return output_size[0] == n and output_size[1] == 2


def forward_test2():
    """
    Check that your forward function output has the correct values
    """
    n = 10                                   # Number of reviews
    vocab_size = 5                           # Size of vocabulary
    embedding_size = 32                      # Size of embedding
    model = DAN(vocab_size, embedding_size)  # Create the DAN
    X = torch.rand(n, vocab_size)            # Generate data

    # Get the output of your forward function (forward pass)
    logits = model(X)

    # Get the intermediate averaged reviews
    averaged_reviews = model.average(X)

    # Get model beliefs using your intermediate average reviews
    logits2 = model.fc(averaged_reviews)

    return torch.sum((logits - logits2) ** 2).item() < 1e-5


runtest(average_test1, "average_test1")
runtest(average_test2, "average_test2")
runtest(average_test3, "average_test3")
runtest(forward_test1, "forward_test1")
runtest(forward_test2, "forward_test2")

model = DAN(len(vocab), embedding_size=32)
if gpu_available_torch:
    model = model.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5)

# Number of epochs
num_epochs = 400

# Start training
model.train()
for epoch in range(num_epochs):

    # Define variables to track running losses and accuracies
    running_loss = 0.0
    running_acc = 0.0
    count = 0

    for i, (X, y) in enumerate(trainloader):

        # Use GPU if available
        if gpu_available_torch:
            X = X.cuda()
            y = y.cuda()

        # Clear the gradient buffer
        optimizer.zero_grad()

        # Do forward propagation to get the model beliefs
        logits = model(X)

        # Compute the loss
        loss = loss_fn(logits, y)

        # Run a backward propagation to get the gradient
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Get the predictions
        pred = torch.argmax(logits, dim=1)

        # Update the running statistics
        running_acc += torch.sum((pred == y).float()).item()
        running_loss += loss.item()
        count += X.size(0)

    # Print running statistics after training for 100 epochs
    if (epoch + 1) % 100 == 0:
        acc = running_acc / count
        loss = running_loss / len(trainloader)
        print(f"Epoch: {epoch+1}/{num_epochs}, Accuracy: {acc:.4f}, Loss: {loss:.4f}")

  # Evaluate the model
model.eval()

# Define variables to keep track of the running accuracy
running_acc = 0.0
count = 0.0

for X, y in testloader:
    # Use gpu if available
    if gpu_available_torch:
        X = X.cuda()
        y = y.cuda()

    # Do a forward pass and tell PyTorch that no gradient is necessary to save memory
    with torch.no_grad():
        logits = model(X)

    # Calculate the prediction
    pred = torch.argmax(logits, dim=1)

    # Update the running stats
    running_acc += torch.sum((pred == y).float()).item()
    count += X.size(0)

print(f"PyTorch Test Accuracy: {100 * (running_acc / count):.2f}%")

target = torch.randint(high=len(testset), size=(1,)).item()

review_target, label_target = review_test.iloc[target]
if gpu_available_torch:
    bog_target = testset[target][0].unsqueeze(0).cuda()
else:
    bog_target = testset[target][0].unsqueeze(0)


model.eval()
with torch.no_grad():
    logits_target = model(bog_target)

pred = torch.argmax(logits_target, dim=1)
probability = torch.exp(logits_target.squeeze()) / torch.sum(
    torch.exp(logits_target.squeeze())
)

print("Review:", review_target)
print("Ground Truth:", label_meaning[int(label_target)])
cer = 100.0 * probability[pred.item()]
print(f"Prediction: {label_meaning[pred.item()]} ({cer:.2f}% certainty)")

#TensorFlow Approach

X_train = tf.convert_to_tensor(
    bow_featurizer.transform(review_train["review"].values).toarray()
)
y_train = tf.convert_to_tensor(review_train["label"].values.flatten(), dtype=tf.int64)

X_test = tf.convert_to_tensor(
    bow_featurizer.transform(review_test["review"].values).toarray()
)
y_test = tf.convert_to_tensor(review_test["label"].values.flatten(), dtype=tf.int64)

class TFAveragingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size=32):
        super().__init__()
        self.embeds = self.add_weight(shape=(vocab_size, embedding_size))

    def call(self, x):
        return tf.matmul(x, self.embeds) / tf.reduce_sum(x, axis=1, keepdims=True)


model_tf = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(len(vocab),)),
        TFAveragingLayer(len(vocab), embedding_size=32),
        tf.keras.layers.Dense(2, activation=None),
    ]
)

model_tf.summary()

# Create optimizer and loss function
loss_fn_tf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer_tf = tf.keras.optimizers.SGD(learning_rate=5.0)

model_tf.compile(optimizer=optimizer_tf, loss=loss_fn_tf, metrics=["accuracy"])
num_epochs = 100
history = model_tf.fit(
    X_train,
    y_train,
    epochs=num_epochs,
    verbose=0,
    callbacks=[ProgBarLoggerNEpochs(num_epochs, every_n=25)],
)

plt.figure(figsize=(9, 6))
plt.plot(np.arange(num_epochs) + 1, history.history["loss"])
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training Loss for the TensorFlow Model")
plt.show()

## Evaluate the network
logits_tf = model_tf(X_test)

# Calculate the prediction
pred_tf = tf.argmax(logits_tf, axis=1)

# Update the running stats
acc = tf.reduce_mean(tf.cast(pred_tf == y_test, tf.float64)).numpy()
print(f"TensorFlow Test Accuracy: {100 * acc:.2f}%")

bog_target = tf.expand_dims(X_test[target], axis=0)
logits_target = model_tf(bog_target)

pred_tf2 = tf.argmax(logits_target, axis=1)
prob_num = tf.exp(tf.squeeze(logits_target))
prob_den = tf.reduce_sum(tf.exp(tf.squeeze(logits_target)))
probability = prob_num / prob_den

print("Review:", review_target)

real = label_meaning[int(label_target)]
pred = label_meaning[int(pred_tf2)]
cert = 100.0 * probability[int(pred_tf2)]
print("Ground Truth:", real)
print(f"Prediction: {pred} ({cert:.2f}% certainty)")

