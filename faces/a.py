import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
a=[[1,2,3,4],[5,6,7,8],[1,2,3,4]]
a[0:1,1:3]=[[0,0,0],[0,0,0]]
print(tf.get_default_graph())
a=2
