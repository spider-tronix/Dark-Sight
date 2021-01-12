import os

import numpy as np
import rawpy


def pack_raw(raw, black_level=512, nbits=14):
    """
    Packs Bayer Img to 4 Channels. The spatial resolution is reduced
    by a factor of 2 on each dimension. The black level is adjusted
    :param raw: Input RAW image
    :param black_level: Level of brightness at the darkest (black) part of a visual image
    :param nbits: Size of image
    :return: Processed image
    """
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (
        np.power(2, nbits) - 1 - black_level
    )  # Subtracting the black level and normalizing the image

    im = np.expand_dims(im, axis=2)  # Bring-up for depth-wise concatenation
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    # Reduction of spatial dimension using a step size of '2'
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


def get_paths_fns(type="raw", dataset_root="/home/syzygianinfern0/sambashare/"):
    dataset = []

    if type == "raw":
        file = "myFile_raw.txt"
    elif type == "small":
        file = "myFile_jpg.txt"
    elif type == "large":
        file = "myFile_raw_jpg.txt"
    else:
        raise FileNotFoundError

    with open(dataset_root + file) as handler:  # Also try myFile_raw_jpg and myFile_jpg
        data = handler.read()
        for line in data.split("\n"):
            files = line.split("\t")
            dataset.append(files) if len(files) == 3 else None

    return dataset


def get_processed(type="small", dataset_root="/home/syzygianinfern0/sambashare/"):
    dataset = get_paths_fns(type=type, dataset_root=dataset_root)
    # TODO: Path to be made absolute

    # Exposure ratio between input and ground truth
    ratio = 2.0 / (1 / 40)

    # Reading input
    input_img_list = [
        rawpy.imread(os.path.join(dataset_root, in_path))
        for in_path in list(zip(*dataset))[0]
    ]
    input_img_list = [
        pack_raw(input_img) * ratio for input_img in input_img_list
    ]  # (H,W,C)  C = 4
    input_thermal_list = [
        open(therm).read().strip().split("\n") for _, _, therm in zip(*dataset)
    ]
    temps = []
    for row in input_thermal_list:
        a = [unit for unit in row.strip().split("\t")]
        temps.append(a)
    temps = np.array(temps).astype(np.float)
    temps = np.fliplr(np.flipud(temps))

    # reading ground truth
    gt_img = rawpy.imread(gt_path)
    gt_img = gt_img.postprocess(
        use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
    )
    gt_img = np.float32(gt_img / 65535.0)

    return input_img_list, gt_img


get_processed()
