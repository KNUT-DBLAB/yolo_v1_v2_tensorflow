import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# inputs = tf.reverse(inputs, [-1]) - tf.constant([103.939, 116.779, 123.68])
# inputs /= 255.0

para = np.load("/home/dblab/maeng_space/d_05_vgg_and_imgs/vgg_para/vgg16.npy", encoding="latin1", allow_pickle=True).item()

print("conv1_1", para["conv1_1"][0].shape)
print("conv1_1", para["conv1_1"][1].shape)

print("conv1_2", para["conv1_2"][0].shape)
print("conv1_2", para["conv1_2"][1].shape)

print("conv2_1", para["conv2_1"][0].shape)
print("conv2_1", para["conv2_1"][1].shape)

print("conv2_2", para["conv2_2"][0].shape)
print("conv2_2", para["conv2_2"][1].shape)


# inputs = relu(conv(inputs, para["conv1_1"][0], para["conv1_1"][1]))
# inputs = relu(conv(inputs, para["conv1_2"][0], para["conv1_2"][1]))
# inputs = max_pooling(inputs)
# inputs = relu(conv(inputs, para["conv2_1"][0], para["conv2_1"][1]))
# inputs = relu(conv(inputs, para["conv2_2"][0], para["conv2_2"][1]))
# inputs = max_pooling(inputs)
# inputs = relu(conv(inputs, para["conv3_1"][0], para["conv3_1"][1], True))
# inputs = relu(conv(inputs, para["conv3_2"][0], para["conv3_2"][1], True))
# inputs = relu(conv(inputs, para["conv3_3"][0], para["conv3_3"][1], True))
# inputs = max_pooling(inputs)
# inputs = relu(conv(inputs, para["conv4_1"][0], para["conv4_1"][1], True))
# inputs = relu(conv(inputs, para["conv4_2"][0], para["conv4_2"][1], True))
# inputs = relu(conv(inputs, para["conv4_3"][0], para["conv4_3"][1], True))
# inputs = max_pooling(inputs)
# inputs = relu(conv(inputs, para["conv5_1"][0], para["conv5_1"][1], True))
# inputs = relu(conv(inputs, para["conv5_2"][0], para["conv5_2"][1], True))
# inputs = relu(conv(inputs, para["conv5_3"][0], para["conv5_3"][1], True))
# inputs = max_pooling(inputs)
# inputs = tf.layers.flatten(inputs)
# inputs = fc("fc", inputs, 512)
# inputs = fc("logits", inputs, 7*7*30)
# inputs = tf.reshape(inputs, [-1, 7, 7, 30])
# return tf.sigmoid(inputs)

