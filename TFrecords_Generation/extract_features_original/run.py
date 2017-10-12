# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import glob
import array
import time
import numpy as np
from crop import TSNCrop
from extract_feature import extract_feature
from nets import nets_factory
import time

tf.flags.DEFINE_integer('start', 0, '')
tf.flags.DEFINE_integer('end', 1, '')
tf.flags.DEFINE_string('output', '', '')
tf.flags.DEFINE_string('label_file', '.', '')
tf.flags.DEFINE_string('split', 'train', '')
tf.flags.DEFINE_integer('record_id', 0, '')
FLAGS = tf.flags.FLAGS

def load_checkpoint(sess, ckpt_dir):
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #if os.path.isabs(ckpt.model_checkpoint_path):
      # Restores from checkpoint with absolute path.
    model_checkpoint_path = ckpt.model_checkpoint_path
    #else:
      # Restores from checkpoint with relative path.
      #model_checkpoint_path = os.path.join(ckpt_dir, ckpt.model_checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
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
      var_check_name = 'v/' + var
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
      print('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)
    return global_step
  else:
    raise RuntimeError('No checkpoint file found.')


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#### add by 111qqz starts ################################

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder
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
def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    video_id=input_queue[2]
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_image(file_contents, channels=3)
    print("example_shape:",tf.shape(example))
    return example, label,video_id
#### add by 111qqz ends ################################
with tf.Graph().as_default():
  ckp_dir='/mnt/lustre/DATAshare/model-zoo/zhouyao/lsvc_v4_ckpoint'
  output_dir = FLAGS.output 
  split_name = FLAGS.split
  record_id = FLAGS.record_id
  image_label_file_dir=FLAGS.label_file + '/' + str(record_id) + '.txt'
  image_raw_data_dir='/mnt/lustre/DATAshare/lsvc_rawdata/lsvc_1fps'
  num_classes = 500
  network_fn = nets_factory.get_network_fn(
                    'inception_v4',
                    num_classes=num_classes,
                    weight_decay=0.001,
                    is_training=False,
                    data_format='NHWC')
  video_id_list,image_list, label_list = read_labeled_image_list(image_label_file_dir,image_raw_data_dir)
  images = tf.convert_to_tensor(image_list, dtype=tf.string)
  labels = tf.convert_to_tensor(label_list, dtype=tf.string)
  video_ids = tf.convert_to_tensor(video_id_list, dtype=tf.string)
  input_queue = tf.train.slice_input_producer([images, labels,video_ids],
                                            num_epochs=1,
                                            shuffle=False)
  image, label ,video_id = read_images_from_disk(input_queue)
  print("label before:",label)
  crop_frames = TSNCrop(image,img_H=256,img_W=256,out_h=299,out_w=299,max_distort=1, more_crop=True)     
  crop_frames = tf.reshape(crop_frames,[260,299,299,3])
  fea = extract_feature(crop_frames, network_fn)
  print(fea) 
  init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    global_step_val = load_checkpoint(sess, ckp_dir)
    
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
  
      cnt=0
      lst_video_id=' '
      feas_all_list  =[]
      writer = tf.python_io.TFRecordWriter(output_dir+'/'+split_name+'/'+split_name+'-%05d.tfrecord'%record_id)
      while(not coord.should_stop()):
        begin=time.time()
        _feas,_label,_video_id = sess.run([fea,label,video_id])
        end = time.time()
        print("sess time: %.6f"%(end-begin))


        if lst_video_id==' ':
          lst_video_id=_video_id
          lst_label = [int(l) for l in _label[0].split(',')]
        if _video_id==lst_video_id:
          for crop_id in range(260):
            feas_all_list.append(array.array('f', list(_feas[crop_id])).tostring())
          print(">> %s len_feas_list: %d"%(lst_video_id,len(feas_all_list)))
        else:          
          print(len(feas_all_list))
          new_example = tf.train.SequenceExample(
                context=tf.train.Features(
                    feature={
                      'labels': _int64_feature(list(lst_label)),
                      'video_id': _bytes_feature(tf.compat.as_bytes(lst_video_id)),
                }),
                feature_lists=tf.train.FeatureLists(
                    feature_list={
                    "rgb":tf.train.FeatureList(feature=[_bytes_feature(tf.compat.as_bytes(i)) for i in feas_all_list])
                    }
                )
            )
          writer.write(new_example.SerializeToString())
          #feas_all_np = np.array(feas_all_list)
          #print("***************feas_list:",feas_all_list)
          #print("***************feas_all_np:",feas_all_np)
          #saves = np.array([lst_video_id, lst_label, feas_all_np])
          lst_video_id=_video_id
          lst_label = [int(l) for l in _label[0].split(',')]
          feas_all_list =[]
          cnt=cnt+1
          print('write_files')

    except Exception as e:
      coord.request_stop(e)
 
    writer.close() 
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  
