__author__ = "Jing Wang"

from datetime import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from util import *
from model import SimpleGAN

batch_size = 128
z_dim = 100
tensorboard = True  # Write results to Tensorboard or save to files
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

simple_gan = SimpleGAN(z_dim=z_dim)
X, Z, D_solver, G_solver, loss_merged_summary, G_sample_summary, G_sample = simple_gan.build_model()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

sample_verify = sample_z(16, z_dim)
summary_writer = tf.summary.FileWriter('./log/train/' + datetime.now().strftime("%m-%d-%H-%M"), sess.graph)


for i in range(1, 1000000):
    X_batch, _ = mnist.train.next_batch(batch_size)
    sess.run(D_solver, feed_dict={X: X_batch, Z: sample_z(batch_size, z_dim)})
    sess.run(G_solver, feed_dict={Z: sample_z(batch_size, z_dim)})

    if i % 1000 == 0:
        if tensorboard:
            loss_summary = sess.run(loss_merged_summary, feed_dict={X: X_batch, Z: sample_z(batch_size, z_dim)})
            # sample_summary = sess.run(G_sample_summary, feed_dict={Z: sample_z(16, z_dim)})
            sample_summary = sess.run(G_sample_summary, feed_dict={Z: sample_verify})
            summary_writer.add_summary(loss_summary, global_step=i)
            summary_writer.add_summary(sample_summary, global_step=i)
        else:
            samples = sess.run(G_sample, feed_dict={Z: sample_verify})
            save_visualization(samples.reshape(-1, 28, 28, 1), (4, 4), save_path='./out/{}.png'.format(str(i)))

