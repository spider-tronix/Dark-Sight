#OM NAMO NARAYANA
import numpy as np
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

# Ignore warnings
import warnings

class sidUnet(nn.Module):
    def __init__(self):
        super(sidUnet, self).__init__()
        self.conv1_1 = nn.Conv2d(5, 32, 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.pool = nn.MaxPool2d(2)
        self.up_6 = nn.ConvTranspose2d(512, 256, 2, stride=(2, 2))
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=(1, 1))
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.up_7 = nn.ConvTranspose2d(256, 128, 2, stride=(2, 2))
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=(1, 1))
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.up_8 = nn.ConvTranspose2d(128, 64, 2, stride=(2, 2))
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=(1, 1))
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.up_9 = nn.ConvTranspose2d(64, 32, 2, stride=(2, 2))
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=(1, 1))
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.conv10 = nn.Conv2d(32, 12, 3, padding=(1, 1))


    def forward(self, x):
        conv1 = self.pool(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
        conv2 = self.pool(F.relu(self.conv2_2(F.relu(self.conv2_1(conv1)))))
        
        conv3 = self.pool(F.relu(self.conv3_2(F.relu(self.conv3_1(conv2)))))
        conv4 = self.pool(F.relu(self.conv4_2(F.relu(self.conv4_1(conv3)))))
        conv5 = self.pool(F.relu(self.conv5_2(F.relu(self.conv5_1(conv4)))))
        
        # print('conv1.shape: ', conv1.shape, 'conv2.shape: ', conv2.shape, 'conv3.shape: ', conv3.shape, 'conv4.shape: ', conv4.shape, 'conv5.shape: ', conv5.shape, )
        # print('before upsampling:', conv5.shape, 'after upsampling:',  self.up_6(conv5).shape)
        # print('layer shape:', torch.cat((self.up_6(conv5),self.up_6(conv5)), dim = 1).shape)
        up6 = torch.cat((self.up_6(conv5),conv4), dim = 1)
        conv6 = F.relu(self.conv6_2(F.relu(self.conv6_1(up6))))
        # print('before upsampling:', conv6.shape, 'after upsampling:',  self.up_7(conv6).shape)
        up7 = torch.cat((self.up_7(conv6),conv3), dim = 1)
        conv7 = F.relu(self.conv7_2(F.relu(self.conv7_1(up7))))
        up8 = torch.cat((self.up_8(conv7),conv2), dim = 1)
        conv8 = F.relu(self.conv8_2(F.relu(self.conv8_1(up8))))
        up9 = torch.cat((self.up_9(conv8),conv1), dim = 1)
        conv9 = F.relu(self.conv9_2(F.relu(self.conv9_1(up9))))

        conv10 = self.conv10(conv9)
        out = F.pixel_shuffle(conv10, 2)
        print(conv10.shape, out.shape)
        return out



# def lrelu(x):
#     return tf.maximum(x * 0.2, x)


# def upsample_and_concat(x1, x2, output_channels, in_channels):
#     pool_size = 2
#     deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
#     deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

#     deconv_output = tf.concat([deconv, x2], 3)
#     deconv_output.set_shape([None, None, None, output_channels * 2])

#     return deconv_output


# def network(input):
#     conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
#     conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
#     pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

#     conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
#     conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
#     pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

#     conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
#     conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
#     pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

#     conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
#     conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
#     pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

#     conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
#     conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

#     up6 = upsample_and_concat(conv5, conv4, 256, 512)
#     conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
#     conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

#     up7 = upsample_and_concat(conv6, conv3, 128, 256)
#     conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
#     conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

#     up8 = upsample_and_concat(conv7, conv2, 64, 128)
#     conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
#     conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

#     up9 = upsample_and_concat(conv8, conv1, 32, 64)
#     conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
#     conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

#     conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
#     out = tf.depth_to_space(conv10, 2)
#     return out


# def network(input):
#     output = input
#     return output


'''trial run'''
model = sidUnet()
tin = torch.rand(1, 5, 512, 512)
model.forward(tin)