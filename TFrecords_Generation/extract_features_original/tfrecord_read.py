import tensorflow as tf
#import pylab

video_lvl_record = '/mnt/lustre/DATAshare/LSVC2017_features/inceptionv4/train/train-00000.tfrecord'
for example in tf.python_io.tf_record_iterator(video_lvl_record):
    feature_names = ['rgb']
    contexts, features = tf.parse_single_sequence_example(
        example,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
                          "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in feature_names
        })

    video_id = contexts["video_id"]
    labels = contexts["labels"].values
    #width = contexts["width"]
    #height = contexts["height"]
    data = features["rgb"]
    decoded_features = tf.reshape(
            tf.cast(tf.decode_raw(data, tf.float32), tf.float32),
            [-1, 1536])

    with tf.Session() as sess:
        #print(counter)
        vid,labels,data = sess.run([video_id,labels,decoded_features])
        print(vid,labels,len(data),len(data[0]))
        print(data)
        print('\n\n')
        #for frame in data:
        #    image = tf.image.decode_jpeg(frame, channels=3)
        #    image,label = sess.run([image,labels])
        #    print(counter)
        #    pylab.imshow(image)
        #    pylab.show()
