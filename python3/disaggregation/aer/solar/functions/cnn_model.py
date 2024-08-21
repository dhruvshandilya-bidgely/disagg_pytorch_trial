"""
Author - Paras Tehria
Date - 12/11/19
This module converts the data to suitable format for solar detection
"""

# import python packages

import torch.nn as nn


class Net(nn.Module):
    """
    Convolutional Net architechture

    """

    def __init__(self, n_channel, width, height, batch_size, output_dim, solar_config):
        """
        This function initializes convolutional net

        Parameters:
            n_channel                  (int)              :      number of channels in convolutional net
            width                      (int)              :      number of columns in input instance
            height                     (int)              :      number of rows in input instance
            batch_size                 (int)              :      batch size in cnn
            output_dim                 (int)              :      number of classes

        Return:

        """
        super(Net, self).__init__()

        self.n_channel = n_channel
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.output_dim = output_dim

        # getting CNN parameters of cov1 from config
        conv1_out_channels = solar_config.get('cnn_model').get('conv1_out_channels')
        conv1_kernel = solar_config.get('cnn_model').get('conv1_kernel')
        conv1_stride = solar_config.get('cnn_model').get('conv1_stride')
        conv1_pad = solar_config.get('cnn_model').get('conv1_pad')

        # CNN architecture:
        # Conv 1 -> Batch Normalization -> ReLU -> Dropout  ->
        # -> Conv 2 -> Batch Normalization 2 -> ReLU -> fully connected 1 -> fully connected 2 -> output

        self.conv1 = nn.Conv2d(in_channels=self.n_channel, out_channels=conv1_out_channels, kernel_size=conv1_kernel,
                               stride=conv1_stride, padding=conv1_pad)

        # getting CNN parameters of bn1 from config
        bn1_feat = solar_config.get('cnn_model').get('bn1_feat')
        self.bn1 = nn.BatchNorm2d(num_features=bn1_feat)
        self.relu1 = nn.ReLU()

        # getting parameters of pool1 from config
        pool1_kernel = solar_config.get('cnn_model').get('pool1_kernel')
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_kernel)

        # getting parameters of dropout from config
        drop_prob = solar_config.get('cnn_model').get('drop_prob')
        self.dropout = nn.Dropout(p=drop_prob)

        # getting parameters of conv2 from config
        conv2_in_channels = solar_config.get('cnn_model').get('conv2_in_channels')
        conv2_out_channels = solar_config.get('cnn_model').get('conv2_out_channels')
        conv2_kernel = solar_config.get('cnn_model').get('conv2_kernel')
        conv2_stride = solar_config.get('cnn_model').get('conv2_stride')
        conv2_pad = solar_config.get('cnn_model').get('conv2_pad')

        # conv2
        self.conv2 = nn.Conv2d(in_channels=conv2_in_channels, out_channels=conv2_out_channels, kernel_size=conv2_kernel,
                               stride=conv2_stride, padding=conv2_pad)

        # getting parameters of batch normalisation 2 from config
        bn2_feat = solar_config.get('cnn_model').get('bn2_feat')
        self.bn2 = nn.BatchNorm2d(num_features=bn2_feat)
        self.relu2 = nn.ReLU()

        # getting parameters of pool2 from config
        pool2_kernel = solar_config.get('cnn_model').get('pool2_kernel')
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_kernel)

        # getting parameters of fc1 from config
        fc1_in_feat = solar_config.get('cnn_model').get('fc1_in_feat')
        fc1_out_feat = solar_config.get('cnn_model').get('fc1_out_feat')
        self.fc1 = nn.Linear(in_features=fc1_in_feat, out_features=fc1_out_feat)

        # getting parameters of fc2 from config
        fc2_out_feat = solar_config.get('cnn_model').get('fc2_out_feat')
        self.fc2 = nn.Linear(in_features=fc1_out_feat, out_features=fc2_out_feat)

        self.out = nn.Linear(in_features=fc2_out_feat, out_features=self.output_dim)

    # Defining the forward pass
    def forward(self, x):
        """
        Forward propagation cnn

        Parameters:
            x           (np.ndarray)      :       input data to CNN model

        Return:
            output      (torch.Tensor)    :       CNN output class probability
        """

        # CNN layer 1: convolution -> batch normalisation -> relu -> max pooling
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        r1 = self.relu1(b1)
        p1 = self.pool1(r1)

        p1 = self.dropout(p1)

        # CNN layer 2: convolution -> batch normalisation -> relu -> max pooling
        c2 = self.conv2(p1)
        b2 = self.bn2(c2)
        r2 = self.relu2(b2)
        p2 = self.pool2(r2)

        p2 = p2.flatten(1)

        # Fully connected layer
        fc1 = self.fc1(p2)
        fc2 = self.fc2(fc1)

        # output layer
        output = self.out(fc2)

        return output
