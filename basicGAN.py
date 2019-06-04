import os, time, itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

############################################################################
# training parameters
batch_size = 300
lr = 0.0008
train_epoch = 200
noize_size = 256
sample_num = 25
############################################################################
# load MNIST
xqd = np.load('input/data/automobile.npy')
xqd = xqd[:60000, :]
xqd = xqd / 255
xqd[xqd > 0] = 1
# yqd = np.load('quickdrawData/y.npy')
print(np.shape(xqd))
# print(xqd[0,:])
yqd = np.zeros([np.shape(xqd)[0], 10])


# xqd = (xqd - 0.5) / 0.5  # normalization; range: -1 ~ 1

# 取数据
def next_draw_batch(batchsize, x, y):
    batchsize = np.int32(batchsize)
    rows = np.int32(np.random.uniform(0, np.shape(x)[0], [batchsize]))
    batch = x[rows, :]
    #     batch = batch/255
    label = y[rows, :]
    label = label
    return batch, label


# G(z) 256 512 1024 728
def generator(x):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('G_w0', [x.get_shape()[1], 512], initializer=w_init)
    b0 = tf.get_variable('G_b0', [512], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # 2nd hidden layer
    w1 = tf.get_variable('G_w1', [h0.get_shape()[1], 1024], initializer=w_init)
    b1 = tf.get_variable('G_b1', [1024], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    # # 3rd hidden layer
    # w2 = tf.get_variable('G_w2', [h1.get_shape()[1], 1024], initializer=w_init)
    # b2 = tf.get_variable('G_b2', [1024], initializer=b_init)
    # h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # output hidden layer
    w3 = tf.get_variable('G_w3', [h1.get_shape()[1], 784], initializer=w_init)
    b3 = tf.get_variable('G_b3', [784], initializer=b_init)
    o = tf.nn.tanh(tf.matmul(h1, w3) + b3)

    return o


# D(x)
def discriminator(x, drop_out):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('D_w0', [x.get_shape()[1], 512], initializer=w_init)
    b0 = tf.get_variable('D_b0', [512], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    h0 = tf.nn.dropout(h0, drop_out)

    # 2nd hidden layer
    w1 = tf.get_variable('D_w1', [h0.get_shape()[1], 256], initializer=w_init)
    b1 = tf.get_variable('D_b1', [256], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
    h1 = tf.nn.dropout(h1, drop_out)

    # 3rd hidden layer
    # w2 = tf.get_variable('D_w2', [h1.get_shape()[1], 256], initializer=w_init)
    # b2 = tf.get_variable('D_b2', [256], initializer=b_init)
    # h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    # h2 = tf.nn.dropout(h2, drop_out)

    # output layer
    w3 = tf.get_variable('D_w3', [h1.get_shape()[1], 1], initializer=w_init)
    b3 = tf.get_variable('D_b3', [1], initializer=b_init)
    o = tf.sigmoid(tf.matmul(h1, w3) + b3)

    return o


fixed_z_ = np.random.normal(0, 1, (sample_num, noize_size))


def show_result(num_epoch, show=False, save=False, path='result.png', isFix=False):
    z_ = np.random.normal(0, 1, (sample_num, noize_size))

    if isFix:
        test_images = sess.run(G_z, {z: fixed_z_, drop_out: 0.0})
    else:
        test_images = sess.run(G_z, {z: z_, drop_out: 0.0})
    # if num_epoch == 1:
    #     np.save('testimage.npy',test_images)
    #     np.save('testimage2.npy',(test_images+0.5)*2)
    # test_images = (test_images+0.5)*2
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# networks : generator
with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, shape=(None, noize_size))
    G_z = generator(z)

# networks : discriminator
with tf.variable_scope('D') as scope:
    drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
    x = tf.placeholder(tf.float32, shape=(None, 784))
    D_real = discriminator(x, drop_out)
    scope.reuse_variables()
    D_fake = discriminator(G_z, drop_out)

# loss for each network
eps = 1e-2
D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

# trainable variables for each network
t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'D_' in var.name]
G_vars = [var for var in t_vars if 'G_' in var.name]

# optimizer for each network
D_optim = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
G_optim = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# results save folder
if not os.path.isdir('input/MNIST_GAN_results'):
    os.mkdir('input/MNIST_GAN_results')
if not os.path.isdir('input/MNIST_GAN_results/Random_results'):
    os.mkdir('input/MNIST_GAN_results/Random_results')
if not os.path.isdir('input/MNIST_GAN_results/Fixed_results'):
    os.mkdir('input/MNIST_GAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
start_time = time.time()
for epoch in range(train_epoch + 1):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(xqd.shape[0] // batch_size):
        # update discriminator
        x_, nonono = next_draw_batch(batch_size, xqd, yqd)
        #         print(x_[1,:])
        z_ = np.random.normal(0, 1, (batch_size, noize_size))
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: 0.3})

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, noize_size))
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: 0.3})
        if loss_d_ > loss_g_ + 0.3:
            x_, nonono = next_draw_batch(batch_size, xqd, yqd)
            z_ = np.random.normal(0, 1, (batch_size, noize_size))
            loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: 0.3})
        if loss_g_ > loss_d_ - 0.1:
            z_ = np.random.normal(0, 1, (batch_size, noize_size))
            loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: 0.3})
        D_losses.append(loss_d_)
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % (
    (epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))

    if (epoch % 10 == 0):
        p = 'input/MNIST_GAN_results/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        fixed_p = 'input/MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        show_result((epoch + 1), save=True, path=p, isFix=False)
        show_result((epoch + 1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
show_train_hist(train_hist, save=True, path='input/MNIST_GAN_results/MNIST_GAN_train_hist.png')
print("Training finish!... save training results")

