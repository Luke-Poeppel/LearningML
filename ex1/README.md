## The five key steps

See [here](https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/) for a good tutorial.

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
