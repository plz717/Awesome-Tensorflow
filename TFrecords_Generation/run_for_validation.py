# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import glob
import array
import time
import numpy as np
from extract_feature import extract_feature
from nets import nets_factory
import time
import os

tf.flags.DEFINE_integer('start', 0, '')
tf.flags.DEFINE_integer('end', 1, '')
tf.flags.DEFINE_string('output', '', '')
tf.flags.DEFINE_string('label_file', '.', '')
tf.flags.DEFINE_string('split', 'train', '')
tf.flags.DEFINE_integer('record_id', 0, '')
FLAGS = tf.flags.FLAGS

#tf.app.flags.DEFINE_string('train_directory', '/home/sensetime/plz_workspace/Misc/WebVisionStat/webvision/ResizedImages/',
#                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/home/sensetime/plz_workspace/Misc/WebVisionStat/webvision/ResizedImages/val_images_256/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/home/sensetime/plz_workspace/Misc/WebVisionStat/webvision/tfrecord/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('frame_size', 224,
                            'Number of threads to preprocess the images.')
name_to_scope_mapping = {'inception_v4': 'InceptionV4'}
# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   n01440764
#   n01443537
#   n01484850
# where each line corresponds to a label expressed as a synset. We map
# each synset contained in the file to an integer (based on the alphabetical
# ordering). See below for details.
tf.app.flags.DEFINE_string('labels_file',
                           '/home/sensetime/plz_workspace/Misc/WebVisionStat/webvision/info/info/val_filelist.txt',
                           'Labels file')
tf.app.flags.DEFINE_string('ckpt_dir',
                           '/home/sensetime/plz_workspace/Misc/WebVisionStat/ckpt_files/',
                           'Inception model file')
tf.app.flags.DEFINE_string('feature_extractor_name',
                           'inception_v4',
                           'model name')
tf.app.flags.DEFINE_integer('index_range',1,'')
tf.app.flags.DEFINE_string('dataset_name','val','google or flickr or val')
tf.app.flags.DEFINE_string('subset_name','validation','train or validation')

def load_checkpoint(ckpt_dir, sess=None):
    sess = sess or tf.get_default_session()
    model_checkpoint_path = os.path.join(ckpt_dir, FLAGS.feature_extractor_name + '.ckpt')
    if model_checkpoint_path:

        global_step = model_checkpoint_path.split('/')[-1].split('-')[-1]
        if not global_step.isdigit():
            global_step = 0
        else:
            global_step = int(global_step)

        reader = pywrap_tensorflow.NewCheckpointReader(model_checkpoint_path)

        var_list = tf.global_variables()
        if isinstance(var_list, dict):
            var_dict = var_list
        else:
            var_dict = {var.op.name: var for var in var_list}
        available_vars = {}
        for var in var_dict:
            if 'global_step' in var:
                available_vars[var] = var_dict[var]
                continue
            var_check_name = var[var.index('/') + 1:]
            if reader.has_tensor(var_check_name):
                available_vars[var_check_name] = var_dict[var]
            elif reader.has_tensor(var):
                available_vars[var] = var_dict[var]
            else:
                tf.logging.warning(
                    'Variable %s missing in checkpoint %s', var_check_name, model_checkpoint_path)
        var_list = available_vars
        if var_list:
            saver = tf.train.Saver(var_list)
            saver.restore(sess, model_checkpoint_path)
            print('Successfully loaded model from %s.' % model_checkpoint_path)
        return global_step
    else:
        raise RuntimeError('No checkpoint file found.')


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#### add by 111qqz starts ################################


def read_labeled_image_list(image_list_file,image_raw_data_dir):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line, 2-D list
    Returns:
       List with all filenames in file image_list_file
    """

    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    video_name_list=[]
    for line in f:
        filename, label = line.split(',',1)
        
        video_name,_=filename.split('-',1)
        video_name_list.append(video_name)
        #print("filename: %s dirname:%s"%(filename,dirname))
        #print ("label:%s"%label)
        filenames.append(image_raw_data_dir+'/'+video_name+'/'+filename)
        labels.append([str(label.strip('\n'))])
    return video_name_list,filenames, labels



def _find_image_files(data_dir, labels_file):
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of images.
      labels_file: string to the file

    Returns:
      filenames: list of strings; each string is a path to an image file.
      labels: list of integer; each integer identifies the ground truth.
    """
    print('Reading label file %s.' % labels_file)
    webvision_train_file = [l.strip() for l in
                            tf.gfile.FastGFile(labels_file, 'r').readlines()]

    labels = []
    filenames = []
    uids = []

    for idx, row in enumerate(webvision_train_file):
        relative_path, cls = row.split(' ')
        if relative_path.startswith(FLAGS.dataset_name):
            #x = range(FLAGS.index_range[0],FLAGS.index_range[1])
            #index_r = ['%.4d'%(i) for i in x]
            index_r = '%.2d'%(FLAGS.index_range)
            if relative_path[4:6] in index_r and int(relative_path[5:9])%1000<1000:
                path = os.path.join(data_dir, relative_path)
                print("path is:{}",path)
                filenames.append(path)
                uids.append(idx)
                labels.append(int(cls))

    print('Found %d JPEG files inside %s.' %
          (len(filenames), data_dir))
    return uids, filenames, labels



def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """

    filename = input_queue[0]
    label = input_queue[1]
    uid = input_queue[2]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
    print("example_shape:", tf.shape(example))
    return filename, uid, example, label, file_contents


def _convert_to_example_1(uid, filename, image_buffer, feature_vector, label):
    """Build an Example proto for an example.

    Args:
      uid: position in label_file
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      human: string, human-readable label, use for extra infos
    Returns:
      Example proto
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(''),
        'image/filename': _bytes_feature(filename),
        'image/encoded': _bytes_feature(image_buffer),
        'image/feature': _float_feature(feature_vector),
        'image/uid': _int64_feature(uid)}))
    return example


def crop_image(image, height, width, crop_method=0):
    # crop_method 0: center
    # other methods: 4 corners
    # to speed up, don't use list
    shape = tf.shape(image)
    if crop_method == 0:
        y0 = (shape[0] - height) // 2
        x0 = (shape[1] - width) // 2
    elif crop_method == 1:
        y0 = 0
        x0 = 0
    elif crop_method == 2:
        y0 = (shape[0] - height)
        x0 = 0
    elif crop_method == 3:
        y0 = (shape[0] - height)
        x0 = (shape[1] - width)
    elif crop_method == 4:
        y0 = 0
        x0 = (shape[1] - width)
    else:
        raise ValueError('Invalid crop_method, must be 0~4, currently %d' % crop_method)
    return tf.image.crop_to_bounding_box(image, y0, x0, height, width)


def distort_image(image, height, width):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
      image: 3-D float Tensor of image
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      thread_id: integer indicating the preprocessing thread.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    # with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
    # with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
    with tf.name_scope('distort_image'):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        float_image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Note that CropMethod contains 5 enumerated resizing methods, (Center + 4
        # Corners).
        # crop_method = thread_id % 5
        crop_method = 0
        distorted_image = crop_image(float_image, height, width, crop_method)

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors.
        # distorted_image = distort_color(distorted_image, thread_id)

        # Normalize to -1.0, 1.0
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)

        return distorted_image



#### add by 111qqz ends ################################
def pipeline():
    with tf.Graph().as_default():
        image_raw_data_dir = FLAGS.validation_directory
        num_classes = 1000
        network_fn = nets_factory.get_network_fn(
            'inception_v4',
            num_classes=num_classes,
            weight_decay=0.001,
            is_training=False,
            data_format='NHWC')

        uids, filenames, labels = _find_image_files(image_raw_data_dir, FLAGS.labels_file)
        uids = tf.convert_to_tensor(uids, dtype=tf.int32)
        image_names = tf.convert_to_tensor(filenames, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        input_queue = tf.train.slice_input_producer([image_names, labels, uids],
                                                    num_epochs=1,
                                                    shuffle=False)

        filename, uid, image, label, image_content = read_images_from_disk(input_queue)

        distorted_image = distort_image(image, FLAGS.frame_size, FLAGS.frame_size)
        print(distorted_image.shape)
        distorted_image = tf.reshape(distorted_image, [1, 224, 224, 3])
        fea = extract_feature(distorted_image, network_fn)
        # print(fea)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)

            load_checkpoint(FLAGS.ckpt_dir, sess=sess)
            output_file = os.path.join(FLAGS.output_directory, FLAGS.subset_name + '_' + str(FLAGS.index_range))
            writer = tf.python_io.TFRecordWriter(output_file)
            coord = tf.train.Coordinator()
            cnt = 0
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                while(not coord.should_stop()):
                    #_image,_feas,_label,_content = sess.run([image,fea,label,image_content])

                    _filename, _uid, _feas, _label, _content = sess.run([filename, uid, fea, label, image_content])

                    image_buffer = _content
                    int64_label = int(_label)
                    feature_vector = list(_feas.reshape(-1).astype(np.float))

                    # example = _convert_to_example_1(image_buffer, feature_vector, int64_label)
                    example = _convert_to_example_1(_uid, _filename, image_buffer, feature_vector, int64_label)
                    if cnt % 30 == 0:
                        print("example is:", example)

                    writer.write(example.SerializeToString())
                    print("successfully write to tfrecords")
                    cnt = cnt + 1

            except Exception as e:
                coord.request_stop(e)
                print('write %d files in %s' % (cnt, output_file))

            writer.close()
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':
    pipeline()
  
