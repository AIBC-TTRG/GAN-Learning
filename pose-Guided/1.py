from tensorflow.python import pywrap_tensorflow
import os, pdb
import StringIO
import scipy.misc
import numpy as np
import glob

model_dir = '/home/aibc/Documents/pose/Pose-Guided-Person-Image-Generation/test_result'
checkpoint_path = os.path.join(model_dir, "model.ckpt-0")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key) 
    # print(reader.get_tensor(key)) # Remove this is you want to print only variable names