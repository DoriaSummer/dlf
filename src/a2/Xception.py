# Author: Wuli Zuo, a1785343
# Date: 2021-09-08


import numpy as np
from keras.applications.xception import Xception

base_model = Xception(include_top=False,input_shape=(32,32,3))
base_model.summary()