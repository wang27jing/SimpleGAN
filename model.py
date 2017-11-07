"""Simple GAN to generate MNIST images."""
__author__ = "Jing Wang"

import tensorflow as tf
from util import *


class SimpleGAN():
    def __init__(self, z_dim = 100, x_dim = 28 * 28):
        # Real image
        self.X = tf.placeholder(tf.float32, shape=[None, x_dim])

        # Random noise
        self.Z = tf.placeholder(tf.float32, shape=[None, z_dim])

        self.G_W1 = tf.Variable(xavier_init([z_dim, 128]), dtype=tf.float32)
        self.G_b1 = tf.Variable(tf.zeros(shape=[128]), dtype=tf.float32)
        self.G_W2 = tf.Variable(xavier_init([128, 784]), dtype=tf.float32)
        self.G_b2 = tf.Variable(tf.zeros(shape=[784]), dtype=tf.float32)
        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        self.D_W1 = tf.Variable(xavier_init([784, 128]), dtype=tf.float32)
        self.D_b1 = tf.Variable(tf.zeros(shape=[128]), dtype=tf.float32)
        self.D_W2 = tf.Variable(xavier_init([128, 1]), dtype=tf.float32)
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]), dtype=tf.float32)
        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

    def generate(self, z):
        # Two fully connected layers
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_h2 = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_image = tf.nn.sigmoid(G_h2)
        return G_image

    def discriminate(self, x):
        # Two fully connected layers
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    def build_model(self):
        G_sample = self.generate(self.Z)
        D_real = self.discriminate(self.X)
        D_fake = self.discriminate(G_sample)

        D_real = tf.clip_by_value(D_real, 1e-7, 1. - 1e-7)
        D_fake = tf.clip_by_value(D_fake, 1e-7, 1. - 1e-7)

        # Loss for discriminator
        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))

        # Loss for generator
        G_loss = -tf.reduce_mean(tf.log(D_fake))

        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=self.theta_G)

        # For visualization in Tensorboard
        D_loss_summary = tf.summary.scalar('D_loss', D_loss)
        G_loss_summary = tf.summary.scalar('G_loss', G_loss)
        G_sample_summary = tf.summary.image('Generated_image', tf.reshape(G_sample, [-1, 28, 28, 1]), max_outputs=16)
        loss_merged_summary = tf.summary.merge([D_loss_summary, G_loss_summary])

        return self.X, self.Z, D_solver, G_solver, loss_merged_summary, G_sample_summary, G_sample