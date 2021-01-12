# OM NAMO NARAYANA

# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
# from __future__ import division
# import os, time, scipy.io
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# import numpy as np
# import rawpy
# import glob

# input_dir = './dataset/dataset/'
# gt_dir = './dataset/Sony/long/'
# checkpoint_dir = './dataset/augmentation_test/result/'


# result_dir = './dataset/augmentation_test/result/'

# # get train IDs
# train_fns = glob.glob(gt_dir + '0*.ARW')
# train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

# ps = 512  # patch size for training
# save_freq = 500


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

# def pack_raw(raw):
#     # pack Bayer image to 4 channels
#     im = raw.raw_image_visible.astype(np.float32)
#     im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

#     im = np.expand_dims(im, axis=2)
#     img_shape = im.shape
#     H = img_shape[0]
#     W = img_shape[1]

#     out = np.concatenate((im[0:H:2, 0:W:2, :],
#                           im[0:H:2, 1:W:2, :],
#                           im[1:H:2, 1:W:2, :],
#                           im[1:H:2, 0:W:2, :]), axis=2)
#     return out


# sess = tf.Session()
# in_image = tf.placeholder(tf.float32, [None, None, None, 4])
# gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
# out_image = network(in_image)

# G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))

# t_vars = tf.trainable_variables()
# lr = tf.placeholder(tf.float32)
# G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

# saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
# ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
# if ckpt:
#     print('loaded ' + ckpt.model_checkpoint_path)
#     saver.restore(sess, ckpt.model_checkpoint_path)

# # Raw data takes long time to load. Keep them in memory after loaded.
# gt_images = [None] * 6000
# input_images = {}
# input_images['300'] = [None] * len(train_ids)
# input_images['250'] = [None] * len(train_ids)
# input_images['100'] = [None] * len(train_ids)

# g_loss = np.zeros((5000, 1))

# allfolders = glob.glob(result_dir + '*0')
# lastepoch = 0
# for folder in allfolders:
#     lastepoch = np.maximum(lastepoch, int(folder[-4:]))

# learning_rate = 1e-4
# for epoch in range(lastepoch, 4001):
#     if os.path.isdir(result_dir + '%04d' % epoch):
#         continue
#     cnt = 0
#     if epoch > 2000:
#         learning_rate = 1e-5

#     for ind in np.random.permutation(len(train_ids)):
#         # get the path from image id
#         train_id = train_ids[ind]
#         in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
#         in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
#         in_fn = os.path.basename(in_path)

#         gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
#         gt_path = gt_files[0]
#         gt_fn = os.path.basename(gt_path)
#         in_exposure = float(in_fn[9:-5])
#         gt_exposure = float(gt_fn[9:-5])
#         ratio = min(gt_exposure / in_exposure, 300)

#         st = time.time()
#         cnt += 1

#         if input_images[str(ratio)[0:3]][ind] is None:
#             raw = rawpy.imread(in_path)
#             input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

#             gt_raw = rawpy.imread(gt_path)
#             im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
#             gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

#         # crop
#         H = input_images[str(ratio)[0:3]][ind].shape[1]
#         W = input_images[str(ratio)[0:3]][ind].shape[2]

#         xx = np.random.randint(0, W - ps)
#         yy = np.random.randint(0, H - ps)
#         input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
#         gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

#         if np.random.randint(2, size=1)[0] == 1:  # random flip
#             input_patch = np.flip(input_patch, axis=1)
#             gt_patch = np.flip(gt_patch, axis=1)
#         if np.random.randint(2, size=1)[0] == 1:
#             input_patch = np.flip(input_patch, axis=2)
#             gt_patch = np.flip(gt_patch, axis=2)
#         if np.random.randint(2, size=1)[0] == 1:  # random transpose
#             input_patch = np.transpose(input_patch, (0, 2, 1, 3))
#             gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

#         input_patch = np.minimum(input_patch, 1.0)

#         _, G_current, output = sess.run([G_opt, G_loss, out_image],
#                                         feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
#         output = np.minimum(np.maximum(output, 0), 1)
#         g_loss[ind] = G_current

#         print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

#         if epoch % save_freq == 0:
#             if not os.path.isdir(result_dir + '%04d' % epoch):
#                 os.makedirs(result_dir + '%04d' % epoch)

#             temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
#             scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
#                 result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

#     saver.save(sess, checkpoint_dir + 'model.ckpt')


import glob
import rawpy
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from torch.nn import ReplicationPad2d
import tensorflow as tf

from DarkNet.models.sid_unet import sidUnet

# Ignore warnings
import warnings

from torchvision.transforms.functional import pil_to_tensor

warnings.filterwarnings("ignore")

long_shots_dir = "./dataset/dataset/"  # Arvinth
# long_shots_dir = '../dataset/'  # Harshit

long_shot_cam = glob.glob(long_shots_dir + "**/*long*.CR3", recursive=True)
short_shot_cam = glob.glob(long_shots_dir + "**/*short*.CR3", recursive=True)
therm = glob.glob(long_shots_dir + "**/*temp.jpg", recursive=True)


def pack_raw(raw, blevel=512):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    # subtract the black level
    im = np.maximum(im - blevel, 0) / (16383 - blevel)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    return out


class DarkSightDataset(Dataset):
    """DarkSight dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the datapoints.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print(root_dir)
        self.root_dir = root_dir
        self.long_exp_list = glob.glob(root_dir + "**/*long*.CR3", recursive=True)
        self.short_exp_list = glob.glob(root_dir + "**/*short*.CR3", recursive=True)
        self.therm_list = glob.glob(root_dir + "**/temp.jpg", recursive=True)
        self.transform = transform
        print(len(self.long_exp_list))
        assert len(self.long_exp_list) == len(self.short_exp_list) and len(
            self.long_exp_list
        ) == len(self.therm_list)

    def __len__(self):
        print(len(self.long_exp_list))
        return len(self.long_exp_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        long_exp = rawpy.imread(self.long_exp_list[idx])
        short_exp = rawpy.imread(self.short_exp_list[idx])
        therm = Image.open(self.therm_list[idx])
        therm = ImageOps.grayscale(therm)
        sample = {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "thermal_response": therm,
        }

        if self.transform:
            sample = self.transform(sample)

        return [
            torch.tensor(sample["input_sample"].copy()),
            torch.tensor(sample["output_sample"].copy()),
        ]


class PreprocessRaw(object):
    def __call__(self, sample):
        long_exp, short_exp, therm = (
            sample["long_exposure"],
            sample["short_exposure"],
            sample["thermal_response"],
        )
        long_exp = long_exp.postprocess(
            use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
        )
        long_exp = np.float32(long_exp / 65535.0)
        short_exp = pack_raw(short_exp)
        return {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "thermal_response": therm,
        }


class MatchSize(object):
    def __init__(self, cam_shape, therm_shape):
        self.ratio = min(cam_shape[0] / therm_shape[1], cam_shape[1] / therm_shape[0])
        self.therm_shape = (
            int(therm_shape[0] * self.ratio),
            int(therm_shape[1] * self.ratio),
        )
        lpad = int((cam_shape[0] - self.therm_shape[1]) / 2)
        rpad = cam_shape[0] - lpad - self.therm_shape[1]
        upad = int((cam_shape[1] - self.therm_shape[0]) / 2)
        dpad = cam_shape[1] - upad - self.therm_shape[0]
        # self.padding = (upad, lpad, dpad, rpad) #for ImageOps.expand
        self.padding = (upad, dpad, lpad, rpad)  # for RepllicationPad2d
        self.shape = cam_shape

    def __call__(self, sample):
        long_exp, short_exp, therm = (
            sample["long_exposure"],
            sample["short_exposure"],
            sample["thermal_response"],
        )
        therm = therm.resize(self.therm_shape)
        # therm = ImageOps.expand(therm, self.padding)
        print(self.padding)
        m = ReplicationPad2d(self.padding)
        therm_tensor = transforms.ToTensor()(therm).unsqueeze_(0)
        therm_tensor = m(therm_tensor)
        therm = transforms.ToPILImage()(therm_tensor.squeeze_(0))
        return {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "thermal_response": therm,
        }


class RandomCrop(object):
    def __init__(self, ps=512, hbuf=550, wbuf=550):
        self.ps = ps  # patch size
        self.hbuf = hbuf
        self.wbuf = wbuf

    def __call__(self, sample, color_percent=50):
        iter_times = 0
        while True:
            iter_times += 1
            long_exp, short_exp, therm = (
                sample["long_exposure"],
                sample["short_exposure"],
                sample["thermal_response"],
            )
            W = short_exp.shape[1]
            H = short_exp.shape[0]
            ps = self.ps
            xx = np.random.randint(self.wbuf, W - ps - self.wbuf)
            yy = np.random.randint(self.hbuf, H - ps - self.hbuf)
            short_exp = short_exp[yy : yy + ps, xx : xx + ps, :]
            therm = therm.crop((xx, yy, xx + ps, yy + ps))
            long_exp = long_exp[yy * 2 : yy * 2 + ps * 2, xx * 2 : xx * 2 + ps * 2, :]
            cnt = 0
            # #commented for the sake of testing
            # for i in range(0, 512):
            #     for j in range(0, 512):
            #         r = long_exp[i][j][0] * 255
            #         g = long_exp[i][j][1] * 255
            #         b = long_exp[i][j][2] * 255
            #         if r < 40 and g < 40 and b < 40:
            #             cnt += 1
            # print(cnt)
            # color_precent added
            break  # for fast testing remove this
            if cnt < ((512 * 512) * color_percent / 100) or iter_times > 10:
                print("resampling..iteration:{}".format(iter_times))
                break
        return {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "thermal_response": therm,
        }


class RandomFlip(object):
    def __call__(self, sample):
        long_exp, short_exp, therm = (
            sample["long_exposure"],
            sample["short_exposure"],
            sample["thermal_response"],
        )
        therm = np.array(therm)
        if np.random.randint(2, size=1)[0] == 1:  # random flip
            short_exp = np.flip(short_exp / 255.0, axis=0)
            therm = np.flip(therm, axis=0)
            long_exp = np.flip(long_exp, axis=0)
        if np.random.randint(2, size=1)[0] == 1:
            short_exp = np.flip(short_exp, axis=1)
            therm = np.flip(therm, axis=1)
            long_exp = np.flip(long_exp, axis=1)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            short_exp = np.transpose(short_exp, (1, 0, 2))
            therm = np.transpose(therm, (1, 0))
            long_exp = np.transpose(long_exp, (1, 0, 2))
        return {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "thermal_response": therm,
        }


class ConcatTherm(object):
    def __call__(self, sample):
        long_exp, short_exp, therm = (
            sample["long_exposure"],
            sample["short_exposure"],
            sample["thermal_response"],
        )
        # print('short_exp.shape: ', short_exp.shape, 'therm.shape: ', therm.shape)
        input_sample = np.transpose(short_exp, (2, 0, 1))
        input_sample = np.append(input_sample, np.expand_dims(therm, axis=0), axis=0)
        print("input_sample.shape: ", input_sample.shape)
        return {"input_sample": input_sample, "output_sample": long_exp}


def my_transform(train=True, cam_shape=(2010, 3012), therm_shape=(32, 24)):
    transform = [PreprocessRaw()]
    transform.append(MatchSize(cam_shape, therm_shape))
    transform.append(RandomCrop())
    transform.append(RandomFlip())
    transform.append(ConcatTherm())
    transform = transforms.Compose(transform)
    return transform


if __name__ == "__main__":
    # drive code
    dataset_dir = "./dataset/dataset/"  # ARVINTH
    # dataset_dir = './dataset/' HARSHITH
    transformed_dataset = DarkSightDataset(dataset_dir, transform=my_transform(True))
    # data = [transformed_dataset[0], transformed_dataset[1]]
    data = list(transformed_dataset)

    # print(data[0])
    # debugging

    """dictionary format

    print('dataset1: ', data[0]['short_exposure'].shape,
        data[0]['thermal_response'].shape)
    # print('dataset2: ', data[1]['short_exposure'].shape,
    #       data[1]['thermal_response'].shape)
    # data[0]['thermal_response'].show()
    print(np.max(data[0]['long_exposure']))
    plt.figure()
    plt.imshow(data[0]['long_exposure'])
    plt.figure()
    plt.imshow(data[0]['thermal_response'])
    plt.figure()
    plt.imshow(data[0]['short_exposure'][:, :, :3])
    print(data[0]['short_exposure'][:, :, 1:4].shape)
    plt.show()
    print(data[0]['thermal_response'][0][20:30])

    """

    """tensor format"""
    model = sidUnet()
    # data[0] = torch.tensor(data[0])
    # print(data[0])
    # out = model.forward(data[0][0])
    # print(out.shape)

    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    dataiter = iter(dataloader)
    inputs, outputs = dataiter.next()
    print(inputs, outputs)
    prediction = model(inputs)
    print("predictions: ", prediction)


class DarkSighDataLoader:
    def __init__(self):
        # drive code
        self.dataset_dir = "./dataset/dataset/"  # ARVINTH
        # dataset_dir = './dataset/' HARSHITH
        self.transformed_dataset = DarkSightDataset(
            self.dataset_dir, transform=my_transform(True)
        )
        # data = [transformed_dataset[0], transformed_dataset[1]]
        self.data = list(self.transformed_dataset)

    def load(self, batch_size=1, shuffle=True):
        dataloader = DataLoader(self.data, batch_size, shuffle=shuffle)
        return dataloader
