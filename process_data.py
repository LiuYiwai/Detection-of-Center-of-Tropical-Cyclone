import argparse
import collections
import json
import os

import matplotlib.image as mpimg
import numpy as np
from scipy import signal
from tqdm import tqdm

from utils import encode


def get_H():
    a = 10
    b = 20
    c = 4
    d = 4
    kernel = 9
    mat_index = np.arange(0, kernel * kernel).reshape(1, kernel, kernel)
    x_index, y_index = divmod(mat_index, kernel)
    r = ((c - x_index) ** 2 + (d - y_index) ** 2) ** (1 / 2)
    H = (a ** (-2) * np.exp((-r ** 2) / (a ** 2)) - b ** (-2) * np.exp((-r ** 2) / (2 * b ** 2)))
    return H


def get_img(directory_name, filenames):
    img = []
    for filename in filenames:
        file_path = os.path.join(directory_name, filename)
        img_item = mpimg.imread(file_path)  # 读取和代码处于同一目录下的 lena.
        img_item = np.require(img_item, dtype=np.float, requirements=['O', 'W'])
        img.append(img_item)
    return np.array(img)


def get_variance(img, img_size, var_kernel):
    # since = time.time()
    # 计算方差
    var_offset = (var_kernel - 1) // 2
    expand_img = np.pad(img, ((0, 0), (var_offset, var_offset), (var_offset, var_offset)))
    expand_img = np.expand_dims(expand_img, axis=3)
    expand_img = np.repeat(expand_img, var_kernel * var_kernel, axis=3)
    expand_img = expand_img.reshape(expand_img.shape[0], expand_img.shape[1], expand_img.shape[2],
                                    var_kernel, var_kernel)

    ret = np.zeros_like(expand_img[:, var_offset:-var_offset, var_offset:-var_offset])
    for x_offset in range(-var_offset, var_offset + 1):
        for y_offset in range(-var_offset, var_offset + 1):
            mat_x_offset = var_offset + x_offset
            mat_y_offset = var_offset + y_offset
            ret[:, :, :, mat_x_offset, mat_y_offset] = expand_img[
                                                       :,
                                                       mat_x_offset:mat_x_offset + img_size,
                                                       mat_y_offset:mat_y_offset + img_size,
                                                       var_offset, var_offset]

    ret = ret.reshape(ret.shape[0], ret.shape[1], ret.shape[2], -1)
    variance = np.abs(img - (ret.sum(axis=3) / (var_kernel * var_kernel)))

    # time_elapsed = time.time() - since
    # print('completed variance in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))
    return variance


def get_gradient(variance, gradient_kernel):
    # since = time.time()
    # gradient
    gradient_offset = gradient_kernel
    expand_variance = np.pad(variance, ((0, 0),
                                        (gradient_offset, gradient_offset),
                                        (gradient_offset, gradient_offset)))

    plus_k = expand_variance[:, gradient_kernel + gradient_offset:, gradient_offset:-gradient_offset]
    minus_k = expand_variance[:, :-(gradient_kernel + gradient_offset), gradient_offset:-gradient_offset]
    gradient_x = (plus_k - minus_k) / (2 * gradient_kernel)

    plus_k = expand_variance[:, gradient_offset:-gradient_offset, gradient_kernel + gradient_offset:]
    minus_k = expand_variance[:, gradient_offset:-gradient_offset, :-(gradient_kernel + gradient_offset)]
    gradient_y = (plus_k - minus_k) / (2 * gradient_kernel)

    # bias
    gradient_x[gradient_x == 0] = 1e-16
    slope = gradient_y / gradient_x
    angle = np.arctan(slope)
    # angle[np.isnan(angle)] = 0

    # time_elapsed = time.time() - since
    # print('completed angle in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))
    return slope, angle


def get_density(slope, img, img_size):
    # since = time.time()

    mat_index = np.arange(0, img_size * img_size).reshape(1, img_size, img_size)
    mat_index = np.repeat(mat_index, img.shape[0], axis=0)

    x_index, y_index = divmod(mat_index, img_size)
    intercept = y_index - slope * x_index

    slope = slope.reshape(img.shape[0], -1)
    intercept = intercept.reshape(img.shape[0], -1)

    density_matrix = np.zeros_like(img)

    for x_index in range(img_size):
        y = intercept + x_index * slope
        y = np.round(y).astype(np.int)
        mask = (y[:] > 0) & (y[:] < img_size)

        for batch_index in range(img.shape[0]):
            val_y = y[batch_index, mask[batch_index]]
            count = collections.Counter(val_y)
            count_dict = dict(count)
            for key, value in count_dict.items():
                density_matrix[batch_index, x_index, key] += value

    # time_elapsed = time.time() - since
    # print('completed density in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))
    return density_matrix


def get_smooth(density_matrix, smooth_kernel, img_size):
    # since = time.time()
    smooth_offset = (smooth_kernel - 1) // 2
    # smooth
    expand_density = np.pad(density_matrix, ((0, 0), (smooth_offset, smooth_offset), (smooth_offset, smooth_offset)))

    expand_density = np.expand_dims(expand_density, axis=3)
    expand_density = np.repeat(expand_density, smooth_kernel * smooth_kernel, axis=3)
    expand_density = expand_density.reshape(expand_density.shape[0], expand_density.shape[1], expand_density.shape[2],
                                            smooth_kernel, smooth_kernel)

    smooth_density = np.zeros_like(expand_density[:, smooth_offset:-smooth_offset, smooth_offset:-smooth_offset])
    for x_offset in range(-smooth_offset, smooth_offset + 1):
        for y_offset in range(-smooth_offset, smooth_offset + 1):
            mat_x_offset = smooth_offset + x_offset
            mat_y_offset = smooth_offset + y_offset
            smooth_density[:, :, :, mat_x_offset, mat_y_offset] = expand_density[
                                                                  :,
                                                                  mat_x_offset:mat_x_offset + img_size,
                                                                  mat_y_offset:mat_y_offset + img_size,
                                                                  smooth_offset, smooth_offset]

    smooth_density = smooth_density.reshape(smooth_density.shape[0], smooth_density.shape[1],
                                            smooth_density.shape[2], -1).sum(axis=3) / (smooth_kernel * smooth_kernel)

    # time_elapsed = time.time() - since
    # print('completed smooth in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))
    return smooth_density


def get_included_angle(max_position, angle, img_size):
    max_x, max_y = divmod(max_position, img_size)
    mat_index = np.arange(0, img_size * img_size).reshape(1, img_size, img_size).astype(np.float)
    mat_index = np.repeat(mat_index, len(max_position), axis=0)
    x_index, y_index = divmod(mat_index, img_size)

    for idx in range(len(max_position)):
        x_index[idx] -= max_x[idx]
        y_index[idx] -= max_y[idx]

    x_index[x_index == 0] = 1e-16
    slope = y_index / x_index
    normal_vector = np.arctan(slope) + np.pi / 2
    included_angle = np.abs(normal_vector - angle)
    included_angle[included_angle > (np.pi / 2)] = np.pi / 2 - included_angle[included_angle > (np.pi / 2)]

    return included_angle


def split_list(filename_list, batch=32):
    filename_split = []
    filename_num = len(filename_list)
    batch_num = len(filename_list) // batch
    start, end = None, None
    for i in range(batch_num):
        start = i * batch
        end = min(filename_num, (i + 1) * batch)
        filename_split.append(filename_list[start:end])
    if end != filename_num:
        filename_split.append(filename_list[end:filename_num])

    return filename_split


def save_var_encode(filenames, output_dir, result_variance):
    # TODO save some other things like speed in filename
    for filename, result in zip(filenames, result_variance):
        img_name = filename.split('.')[0]
        save_path = os.path.join(output_dir, img_name + '.json')

        variance_encode = list(map(encode, result))
        with open(save_path, 'w') as f:
            json.dump(variance_encode, f)


def save_max_point(filenames, output_dir, max_position, img_size):
    max_x, max_y = divmod(max_position, img_size)
    for filename, (x, y) in zip(filenames, zip(max_x, max_y)):
        img_name = filename.split('.')[0]
        save_path = os.path.join(output_dir, img_name + '.txt')

        with open(save_path, 'w') as f:
            f.write('x:' + str(x) + '\n')
            f.write('y:' + str(y) + '\n')


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    img_size = args.img_size
    var_kernel = args.var_kernel
    gradient_kernel = args.gradient_kernel
    smooth_kernel = args.smooth_kernel
    batch = args.batch

    filename_list = os.listdir(r"./" + input_dir)
    filename_list = split_list(filename_list, batch)
    filename_tqdm = tqdm(filename_list)

    for filenames in filename_tqdm:
        img = get_img(input_dir, filenames)

        variance = get_variance(img, img_size=img_size, var_kernel=var_kernel)

        slope, angle = get_gradient(variance, gradient_kernel=gradient_kernel)

        density_matrix = get_density(slope, img, img_size)

        smooth_density = get_smooth(density_matrix, smooth_kernel, img_size)

        H = get_H()
        H = signal.convolve(smooth_density, H, mode="same")
        max_position = np.array([np.argmax(i) for i in H])

        save_max_point(filenames, output_dir, max_position, img_size)

        included_angle = get_included_angle(max_position, angle, img_size)

        result_variance = get_variance(included_angle, img_size=img_size, var_kernel=var_kernel)

        save_var_encode(filenames, output_dir, result_variance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data', help='input dir path')
    parser.add_argument('--output_dir', type=str, default='./out', help='output dir path')
    parser.add_argument('--batch', type=int, default=32, help='batch')
    parser.add_argument('--img_size', type=int, default=256, help='img_size (int)')
    parser.add_argument('--var_kernel', type=int, default=3, help='var_kernel')
    parser.add_argument('--gradient_kernel', type=int, default=5, help='gradient_kernel')
    parser.add_argument('--smooth_kernel', type=int, default=3, help='smooth_kernel')

    args = parser.parse_args()
    main(args)
