# Architecture of other NN models in the ablation study

The following NN models are used for the ablation study of TrustEMG-Net, including CNN, FCN, U-Net, and U-Net+Trans. 
The kernels of the convolutional layers of these models are all 1-dimensional.

## CNN

The input signal is segmented into 200 points per segment, as we discover that smaller segments can yield better results on CNN. 
After passing through the CNN, all the outputs are concatenated to form the denoised signal. 

The CNN consists of 4 convolutional layers and 2 fully connected layers. Except for the last fully connected layer, which is a linear layer, all other layers use the ReLU activation function and batch normalization. The parameters for the convolutional layers are shown below. 

| Layer type and number   | Input channel | Filter number | Filter size | Stride |
|-------------------------|---------------|---------------|-------------|--------|
| Convolutional layer 1   | 1             | 16            | 8           | 2      |
| Convolutional layer 2   | 16            | 32            | 8           | 2      |
| Convolutional layer 3   | 32            | 64            | 8           | 2      |
| Convolutional layer 4   | 64            | 128           | 8           | 2      |

The dimensions of the two fully connected layers are 400 and 200, respectively. Due to a large number of parameters in the first fully connected layer, dropout regularization with a dropout rate of 50% is applied during training.

## FCN

The FCN's encoder and decoder each consist of 5 convolutional layers and 5 transposed convolutional layers, with detailed parameters shown as below. Except for the last transposed convolutional layer, each layer uses the ReLU activation function and batch normalization

| Layer type and number     | Input channel | Filter number | Filter size | Stride |
|---------------------------|---------------|---------------|-------------|--------|
| Convolutional layer 1     | 1             | 64            | 8           | 2      |
| Convolutional layer 2     | 64            | 128           | 8           | 2      |
| Convolutional layer 3     | 128           | 256           | 8           | 2      |
| Convolutional layer 4     | 256           | 512           | 8           | 2      |
| Convolutional layer 5     | 512           | 1024          | 8           | 2      |
| ConvTranspose layer 1     | 1024          | 512           | 8           | 2      |
| ConvTranspose layer 2     | 512           | 256           | 8           | 2      |
| ConvTranspose layer 3     | 256           | 128           | 8           | 2      |
| ConvTranspose layer 4     | 128           | 64            | 8           | 2      |
| ConvTranspose layer 5     | 64            | 1             | 8           | 2      |


## U-Net
The architecture of the U-Net is the same as the U-Net part in TrustEMG-Net. The only difference is that the bottleneck here does not utilize a Transformer encoder.

## U-Net+Trans
The parameter setting of the U-Net+Trans is the same as TrustEMG-Net. The only difference is that the Transformer encoder here does not utilize a masking approach.
