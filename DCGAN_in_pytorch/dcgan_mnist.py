# USAGE
# python dcgan_mnist.py --output output


# import the necessary packages
from pyimagesearch.dcgan import Generator, Discriminator

# dataset
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms

from sklearn.utils import shuffle
from imutils import build_montages

from torch.optim import Adam
from torch.nn import BCELoss
from torch import nn
import numpy as np
import argparse
import torch
import cv2
import os

# custom weights initialization called on Generator and Discriminator
def weights_init(model):
    """ initialize custom weights depending on the layer encountered . 
    Later, during the inference step, we’ll see that this has improved our training loss values."""
    # get the class name
    classname = model.__class__.__name__

    # Check if the classname contains 'Conv'
    if classname.find('Conv') != -1:
        #initialize the weights from the normal distribution
        # model.weight.data.normal_(0.0, 0.02)
        nn.init.normal_(model.weight.data, mean=0.0, std=0.02)

    # otherwise, check if the classname contains 'BatchNorm'
    elif classname.find('BatchNorm') != -1:
        # initialize the weights from normal distribution and set the biases to 0
        # model.weight.data.normal_(1.0, 0.02)
        # model.bias.data.fill_(0)
        nn.init.normal_(model.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(model.bias.data, 0)

    
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=20, help="# epochs to train for")
ap.add_argument("-b", "--batch-size", type=int, default=128, help="batch size for training")
ap.add_argument("-d", "--debug", type=bool, default=False, help="help debugging")
args = vars(ap.parse_args())

# store the epochs and batch size in convenience variables
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]
DEBUG = args["debug"]

# Set the device to use for training
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define data transformation
data_transforms = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

# load the MNIST dataset and stack the training and testing data into a single dataset -> to have additional training data
print("[INFO] loading MNIST dataset...")
trainData = MNIST(root='./data', train=True, transform=data_transforms, download=True)
testData = MNIST(root='./data', train=False, transform=data_transforms, download=True)
data = torch.utils.data.ConcatDataset([trainData, testData])

# initialize our dataLoader
dataloader =DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)

# calculate the total number of batches per training epoch (steps per epoch ?)
stepsPerEpoch  = len(dataloader.dataset) // BATCH_SIZE

# build the generator, initialize the weights, and flash it to the GPU/CPU
print("[INFO] building generator...")
gen = Generator(inputDim= 100, outputDim = 512, outputChannels= 1, debug = DEBUG)
gen.apply(weights_init)
gen.to(DEVICE)

# build the discriminator, initialize the weights, and flash it to the GPU/CPU
print("[INFO] building discriminator...")
disc = Discriminator(depth= 1, debug = DEBUG)
disc.apply(weights_init)
disc.to(DEVICE)

# initialize the optimizers for both generator and discriminator
genOpt = Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002 / NUM_EPOCHS)
discOpt = Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002 / NUM_EPOCHS)

# initialize the BCE loss function
criterion = BCELoss()

# randomely generate some benchmark noise so we can consistently see the generator's performance
print("[INFO] strating training...")
benchmarkNoise = torch.randn(256, 100, 1, 1, device=DEVICE)

# define real and fake labels
realLabel, fakeLabel = 1, 0

# loop over the epochs
for epoch in range(NUM_EPOCHS):
    # show epoch info and compute the number of batches / steps per epoch
    print("[INFO] starting epoch {} of {}...".format(epoch + 1, NUM_EPOCHS))

    # initialize current epoch lose for generator and discriminator
    epochLossG, epochLossD = 0, 0

    for x in dataloader: 
        # zero the gradient
        disc.zero_grad()

        # grab the images and send them to the DEVICE
        images = x[0]
        images = images.to(DEVICE)

        # get the batch size and create a label tensor
        bs = images.size(0)
        labels = torch.full((bs,), realLabel, dtype= torch.float ,device=DEVICE)

        # forward pass through the discriminator
        if DEBUG: print("[INFO]  forward pass through the discriminator ...")
        output = disc(images).view(-1)

        # calculate the loss on all-real batch 
        errorReal = criterion(output, labels)

        # calculate the gradients by performing backpropagation
        errorReal.backward()

        # randomly generate noise fro the generator to preiddct on 
        noise = torch.randn(bs, 100, 1, 1, device=DEVICE)

        # generate fake images batch using the generator
        fake = gen(noise)
        labels.fill_(fakeLabel)

        # perform a forward pass through the discriminator using fake batch data
        if DEBUG: print("[INFO]  perform a forward pass through the discriminator using fake batch data ...")
        output = disc(fake.detach()).view(-1)
        errorFake = criterion(output, labels)

        # calculate the gradients by performing backpropagation
        errorFake.backward()

        # compute the total error for the discriminator and update it
        errorD = errorReal + errorFake
        discOpt.step()

        # set all generator gradients to zero
        gen.zero_grad()
        # update the labels as fake labels are real for the generator
        # and perform a forward pass  of fake data batch through the discriminator
        labels.fill_(realLabel)
        output = disc(fake).view(-1)
        # calculate generator's loss based on output from
        # discriminator and calculate gradients for generator
        errorG = criterion(output, labels)
        errorG.backward()
        # update the generator
        genOpt.step()
        # add the current iteration loss of discriminator and generator
        epochLossD += errorD
        epochLossG += errorG

    # display training information to disk
    print("[INFO] Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format( 
                        epochLossG / stepsPerEpoch, epochLossD / stepsPerEpoch))

    # check to see if we should visualize the output of the generator model on our benchmark data
    if (epoch + 1) % 2 == 0:
        # set the generator in evaluation phase, make predictions on the benchmark noise, 
        # scale it back to the range [0, 255], and generate the montage
        gen.eval()
        images = gen(benchmarkNoise)
        images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
        images = ((images * 127.5) + 127.5).astype("uint8")
        images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (28, 28), (16, 16))[0]
        # build the output path and write the visualization to disk
        p = os.path.join(args["output"], "epoch_{}.png".format(
            str(epoch + 1).zfill(4)))
        cv2.imwrite(p, vis)
        # set the generator to training mode
        gen.train()