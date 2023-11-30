# Settings of NN models in the ablation study
## CNN

The input signal is segmented into 200 points per segment, as we discover that smaller segments can yield better results in CNN. 
After passing through the CNN, all the outputs are concatenated to form the denoised signal. 

The convolutional neural network consists of 4 convolutional layers and 2 fully connected layers in the first half and second half, respectively. Except for the last fully connected layer, which is a linear layer, all other layers use the ReLU activation function and batch normalization. The parameters for the convolutional layers are shown below. 
| Layer type and number   | Input channel | Filter number | Filter size | Stride |
|-------------------------|---------------|---------------|-------------|--------|
| Convolutional layer 1   | 1             | 16            | 8           | 2      |
| Convolutional layer 2   | 16            | 32            | 8           | 2      |
| Convolutional layer 3   | 32            | 64            | 8           | 2      |
| Convolutional layer 4   | 64            | 128           | 8           | 2      |

In addition, the dimensions of the two fully connected layers are 400 and 200, respectively. Due to a large number of parameters in the first fully connected layer, dropout regularization with a dropout rate of 50% is applied during training.





## FCN

## U-Net

## U-Net+Trans
