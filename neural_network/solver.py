import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import glob, datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
sns.set_theme(style="darkgrid")
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_dir(path: str):
    """
    Create a directory of it does not exist
    :param path: Path to directory
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def new_tri_plot():
    # lims_young = [0, 12]
    # lims_poi = [-0.5, 0.5]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[12, 4], constrained_layout=True)

    # ax1.plot(lims_young, lims_young, 'gray')
    # ax1.set_xlim(lims_young)
    # ax1.set_ylim(lims_young)
    ax1.set_aspect(1)
    ax1.set_xlabel("True values")
    ax1.set_ylabel("Predictions")
    ax1.set_title("Rho")

    # ax2.plot(lims_poi, lims_poi, 'gray')
    # ax2.set_xlim(lims_poi)
    # ax2.set_ylim(lims_poi)
    ax2.set_aspect(1)
    ax2.set_xlabel("True values")
    ax2.set_ylabel("Predictions")
    ax2.set_title("Young's modulus")

    ax3.set_aspect(1)
    ax3.set_xlabel("True values")
    ax3.set_ylabel("Predictions")
    ax3.set_title("Conductivity")

    return ax1, ax2, ax3


def tri_plot(y, pred, ax1, ax2, ax3):

    x1 = y[:, 0]
    x2 = y[:, 1]
    x3 = y[:, 2]
    y1 = pred[:, 0]
    y2 = pred[:, 1]
    y3 = pred[:, 2]

    ax1.scatter(x1, y1, c='C0', alpha=0.2)
    ax2.scatter(x2, y2, c='C1', alpha=0.2)
    ax3.scatter(x3, y3, c='C2', alpha=0.2)


def pad_dim(x, n=1):
    x = tf.concat((x[:, :, :, -n:, :], x, x[:, :, :, :n, :]), axis=-2)
    x = tf.concat((x[:, :, -n:, :, :], x, x[:, :, :n, :, :]), axis=-3)
    x = tf.concat((x[:, -n:, :, :, :], x, x[:, :n, :, :, :]), axis=-4)
    return x


class Solver(keras.Model):

    def __init__(self):
        super(Solver, self).__init__()

    def build(self):

        # unit 1, [64,64,64,1] => [32,32,32,16]
        self.conv1a = layers.Conv3D(16, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv1b = layers.Conv3D(16, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max1 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 2, => [16,16,16,32]
        self.conv2a = layers.Conv3D(32, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv2b = layers.Conv3D(32, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max2 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 3, => [8,8,8,64]
        self.conv3a = layers.Conv3D(64, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv3b = layers.Conv3D(64, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max3 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 4, => [4,4,4,128]
        self.conv4a = layers.Conv3D(128, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv4b = layers.Conv3D(128, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max4 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 5, => [2,2,2,256]
        self.conv5a = layers.Conv3D(256, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv5b = layers.Conv3D(256, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max5 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 6, => [1,1,1,512]
        self.conv6a = layers.Conv3D(512, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.conv6b = layers.Conv3D(512, kernel_size=3, padding='valid', activation=tf.nn.relu)
        self.max6 = layers.MaxPool3D(pool_size=2, strides=2, padding='valid')

        # unit 7, => [2]
        self.fc1 = layers.Dense(256, activation=tf.nn.relu)
        self.fc2 = layers.Dense(128, activation=tf.nn.relu)
        self.fc3 = layers.Dense(3, activation=None)

    def call(self, x):
        # inputs_noise: (b, 64), inputs_condition: (b, 3)
        x = pad_dim(x)
        x = self.conv1a(x)
        x = pad_dim(x)
        x = self.conv1b(x)
        x = self.max1(x)

        x = pad_dim(x)
        x = self.conv2a(x)
        x = pad_dim(x)
        x = self.conv2b(x)
        x = self.max2(x)

        x = pad_dim(x)
        x = self.conv3a(x)
        x = pad_dim(x)
        x = self.conv3b(x)
        x = self.max3(x)

        x = pad_dim(x)
        x = self.conv4a(x)
        x = pad_dim(x)
        x = self.conv4b(x)
        x = self.max4(x)

        x = pad_dim(x)
        x = self.conv5a(x)
        x = pad_dim(x)
        x = self.conv5b(x)
        x = self.max5(x)

        # x = pad_dim(x)
        # x = self.conv6a(x)
        # x = pad_dim(x)
        # x = self.conv6b(x)
        # x = self.max6(x)

        x = tf.keras.layers.Flatten()(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    # x = tf.expand_dims(x, -1)
    y = tf.cast(y, dtype=tf.float32)
    return x, y


def main():

    # tf.random.set_seed(2345)

    dataset_x = np.load("dataset_5000x32x32x32.npy")
    dataset_x = np.expand_dims(dataset_x, axis=-1)
    path_label = "phi_e_cond_5000x32x32x32.txt"
    tag_csv = pd.read_csv(path_label, delimiter="\t", header=None)
    label = tag_csv.values[:, 1:]

    x_test = dataset_x[int(0.8 * len(dataset_x)):-1]
    y_test = label[int(0.8 * len(label)):-1]
    x = dataset_x[0:int(0.8 * len(dataset_x))]
    y = label[0:int(0.8 * len(label))]

    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(10000).map(preprocess).batch(32)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess).batch(32)

    create_dir("./results/train")
    create_dir("./results/test")
    create_dir("./results/ckpt")
    with open('./results/loss.txt', 'w') as f:
        f.write("loss" + " " + "mse" + "\n")

    solver = Solver()
    solver.build(input_shape=[None, 32, 32, 32, 1])
    optimizer = optimizers.Adam(learning_rate=1e-4, beta_1=0.5)  ## lr = 2e-4, beta_1 = 0.9

    for epoch in range(200):

        # plot train
        ax1, ax2, ax3 = new_tri_plot()
        epoch += 1
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                logits = solver(x)
                loss = tf.losses.MSE(y, logits)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, solver.trainable_variables)
            optimizer.apply_gradients(zip(grads, solver.trainable_variables))

            if step < 30:
                tri_plot(y, logits, ax1, ax2, ax3)

        plt.savefig('results/train/%d_train.png' % epoch)
        plt.close()
        print(epoch, "loss: ", loss.numpy())

        # plot test
        ax1, ax2, ax3 = new_tri_plot()
        total_sum = 0
        total_error = 0
        rho_real = []
        rho_pred = []
        e_real = []
        e_pred = []
        cond_real = []
        cond_pred = []
        for x, y in test_db:

            pred = solver(x)
            loss_test = tf.losses.MSE(y, pred)
            loss_test = tf.reduce_mean(loss_test)

            total_sum += 1
            total_error += loss_test

            tri_plot(y, pred, ax1, ax2, ax3)

            if epoch == 200:
                rho_real = np.append(rho_real, y[:,0].numpy())
                rho_pred = np.append(rho_pred, pred[:,0].numpy())
                e_real = np.append(e_real, y[:,1].numpy())
                e_pred = np.append(e_pred, pred[:,1].numpy())
                cond_real = np.append(cond_real, y[:,2].numpy())
                cond_pred = np.append(cond_pred, pred[:,2].numpy())

        mse = total_error / total_sum
        print(epoch, "mse: ", mse.numpy())
        plt.savefig('results/test/%d_test.png' % epoch)
        plt.close()
        with open('./results/loss.txt', 'a') as f:
            f.write(str(loss.numpy()) + " " + str(mse.numpy()) + "\n")

        solver.save_weights('results/ckpt/solver_%d.weights.h5' % epoch)

        if epoch == 200:
            g = sns.jointplot(x=rho_real, y=rho_pred,
                              xlim=[0, 0.8], ylim=[0, 0.8],
                              color="C0", height=3.5)
            g.set_axis_labels("Input relative density", "Output relative density")
            plt.tight_layout()
            plt.savefig('results/results_rho.svg')
            plt.close()

            g = sns.jointplot(x=e_real, y=e_pred,
                              xlim=[0, 0.6], ylim=[0, 0.6],
                              color="C1", height=3.5)
            g.set_axis_labels("Input relative Young's modulus", "Output relative Young's modulus")
            plt.tight_layout()
            plt.savefig('results/results_e.svg')
            plt.close()

            g = sns.jointplot(x=cond_real, y=cond_pred,
                              xlim=[0, 0.7], ylim=[0, 0.7],
                              color="C2", height=3.5)
            g.set_axis_labels("Input relative conductivity", "Output relative conductivity")
            plt.tight_layout()
            plt.savefig('results/results_cond.svg')
            plt.close()


if __name__ == "__main__":
    main()
