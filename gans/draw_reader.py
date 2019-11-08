import matplotlib.pyplot  as plt
import numpy as np
import os
from scipy import misc
import itertools


def labeling(name):
    temp = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    namelist = [ f+".npy"  for f in temp]
    return namelist.index(name)

def next_draw_batch(batchsize):
    batchsize = np.int32(batchsize)
    x = np.load('quickdrawData/x.npy')
    y = np.load('quickdrawData/y.npy')
    rows = np.int32(np.random.uniform(0, np.shape(x)[0], [batchsize]))
    batch = x[rows, :]
    label = y[rows, :]
    return batch,label

################
# x,y = next_draw_batch(30)
# print(np.shape(x),':::::',np.shape(y))
# print(y)
# misc.imsave('tempmem/outfile1.jpg', x[0,:].reshape([28,28]))
# misc.imsave('tempmem/outfile2.jpg', x[29,:].reshape([28,28]))
# print(np.shape(x))
# np.save('quickdrawData/x.npy',x)
# np.save('quickdrawData/y.npy',y)
# x = np.load('quickdrawData/x.npy')
# for i in range(200):
#     tt  = np.random.randint(0,60000)
#     misc.imsave('out/q%d.png'%tt,x[tt,:].reshape(28,28))
clist = []
qlist = []
for i in range(18):
    a = misc.imread('zhong/%d.png'%i)
    clist.append(a)
    b = misc.imread('zhong/q%d.png'%i)
    qlist.append(b)

fig, ax = plt.subplots(6, 6, figsize=(5, 5))
for i, j in itertools.product(range(6), range(6)):
    ax[i, j].get_xaxis().set_visible(False)
    ax[i, j].get_yaxis().set_visible(False)
m=0
n=0
for k in range(6 * 6):
    i = k // 6
    j = k % 6
    ax[i, j].cla()
    if i%2==0:
        ax[i, j].imshow(clist[n])
        n+=1
    if i%2==1:
        ax[i, j].imshow(qlist[m], cmap='gray')
        m+=1
    plt.savefig('out/result.png')









