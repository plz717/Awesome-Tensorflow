import tensorflow as tf
import skimage.io
import matplotlib.pylab as plt
import numpy as np
FLAGS = tf.flags.FLAGS



def TSNCrop(raw_frames, img_H, img_W, out_h, out_w, max_distort=0, more_crop=False, flip=True):
  raw_frames = tf.reshape(raw_frames, [1,img_H,img_W,3])
  crop_ratio = [1.0,0.875,0.75,0.66]
  pairs = []
  for i, h in enumerate(crop_ratio):
    for j, w in enumerate(crop_ratio):
      if abs(i - j) <= max_distort:
        pairs.append((w, h))

  pair_num = len(pairs)
  #print('There are {} scale options.'.format(pair_num))
  #print(pairs)
  crop_pair = tf.convert_to_tensor(pairs, dtype=tf.float32)

  if more_crop:
    loc_num = 13
  else:
    loc_num = 5
  #print('There are {} location options.'.format(loc_num))
  #print('For one frame, there are {} cropped patch.'.format(loc_num*pair_num))

  short_edge = tf.reshape(tf.minimum(img_W, img_H), [1])
  img_hw = tf.concat([short_edge, short_edge], axis=0)
  img_hw = tf.cast(img_hw, dtype=tf.float32)
  # return img_H

  def do_crop(crop_pair_size):
    crop_pair = crop_pair_size
    crop_hw = tf.multiply(img_hw, crop_pair)
    crop_hw = tf.cast(crop_hw, dtype=tf.int32)
    crop_h = crop_hw[0]
    crop_w = crop_hw[1]
    crop_size = [-1, crop_h, crop_w, -1]

    w_step = tf.cast(img_W - crop_w, tf.float32)
    w_step = w_step / 4.0
    w_step = tf.squeeze(tf.cast(w_step, tf.int32))
    h_step = tf.cast(img_H - crop_h, tf.float32)
    h_step = h_step / 4.0
    h_step = tf.squeeze(tf.cast(h_step, tf.int32))

    crop_start_all = tf.convert_to_tensor([[0, 0, 0, 0],                                # top left
                                           [0, 0, 4 * w_step, 0],                         # top right
                                           [0, 4 * h_step, 0, 0],                         # bottom left
                                           [0, 4 * h_step, 4 * w_step, 0],                  # bottom right
                                           [0, 2 * h_step, 2 * w_step, 0]])                 # center

    if more_crop:
      crop_start_add = tf.convert_to_tensor([[0, 2 * h_step, 0, 0],  # center left
                                             [0, 2 * h_step, 4 * w_step, 0],  # center right
                                             [0, 4 * h_step, 2 * w_step, 0],  # lower center
                                             [0, 0 * h_step, 2 * w_step, 0],  # upper center
                                             [0, 1 * h_step, 1 * w_step, 0],  # upper left quarter
                                             [0, 1 * h_step, 3 * w_step, 0],  # upper right quarter
                                             [0, 3 * h_step, 1 * w_step, 0],  # lower left quarter
                                             [0, 3 * h_step, 3 * w_step, 0]]) # lower righ quarter

      crop_start_all = tf.concat([crop_start_all, crop_start_add], axis=0)

    cropped_img_batch = None
    for i in range(loc_num):
      crop_start = crop_start_all[i]
      cropped_img_temp = tf.slice(raw_frames, begin=crop_start, size=crop_size)
      if i == 0:
        cropped_img_batch = cropped_img_temp
      else:
        cropped_img_batch = tf.concat([cropped_img_batch, cropped_img_temp], axis=0)
    cropped_img_batch = tf.reshape(cropped_img_batch, [loc_num, crop_h, crop_w, 3])
    cropped_img_batch = tf.image.resize_images(cropped_img_batch, [out_h, out_w])

    return cropped_img_batch

  cropped_frames = None
  for i in range(pair_num):
    cropped_frames_temp = do_crop(crop_pair[i])
    cropped_frames_temp = tf.expand_dims(cropped_frames_temp, axis=0)
    if i == 0:
      cropped_frames = cropped_frames_temp
    else:
      cropped_frames = tf.concat([cropped_frames, cropped_frames_temp], axis=0)

  cropped_frames = tf.reshape(cropped_frames, [pair_num, loc_num, out_h, out_w, 3])

  if flip:
    flipped_frames = tf.reshape(cropped_frames, [pair_num*loc_num* out_h, out_w, 3])
    flipped_frames = tf.image.flip_left_right(flipped_frames)
    flipped_frames = tf.reshape(flipped_frames, [pair_num, loc_num, out_h, out_w, 3])
    flipped_frames = tf.concat([cropped_frames, flipped_frames], axis=0)
    output_frames = tf.reshape(flipped_frames, [2, pair_num, loc_num, out_h, out_w, 3])
  else:
    output_frames = tf.expand_dims(cropped_frames, axis=1)

  return  output_frames


