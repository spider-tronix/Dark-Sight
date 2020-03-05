from skimage.measure import compare_ssim, compare_psnr


def get_psnr(im1, im2):
    return compare_psnr(im1, im2, data_range=255)


def get_ssim(im1, im2):
    return compare_ssim(im1, im2, data_range=255, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
