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
