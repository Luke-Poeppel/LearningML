## The five key steps

See [here](https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/) for a good tutorial.

The ex1 model comes from [here](https://ashleyy-czumak.medium.com/mnist-digit-classification-in-pytorch-302476b34e4f). 

1. Prepare the data
2. Define the model
3. Train the model
4. Evaluate the model
5. Make predictions

The standard installer for MNIST was broken. Suggests using wget instead. First, install with homebrew:
```
$ brew install wget
$ wget www.di.ens.fr/~lelarge/MNIST.tar.gz
$ tar -zxvf MNIST.tar.gz
```
Then just follow the code used for downloading in `ex1.py`

### Data
- Shape of training data: `torch.Size([60000, 28, 28])` -> 60,000 images of size 28x28.
- It's not enough to just prepare the dataset objects. You also should have a `DataLoader`. This object, "reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval."

### NN
- Input layer 
- Hidden layer
- Output layer

In `Net.forward`, we apply ReLU (rectified linear unit) after each linear transformation (except the last one...). 

Activation functions are used in the hidden layers _and_ in the output layer.

I think the layers are called forward-propogation, but not totally sure.

NOTE: `lr`=learning rate

### Random
`Tensor.view` is the same as `np.reshape`. Note that: "If there is any situation that you don't know how many rows you want but are sure of the number of columns, then you can specify this with a -1. (Note that you can extend this to tensors with more dimensions. Only one of the axis value can be -1)."
https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch 
