import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import warnings

warnings.simplefilter('ignore')
IMAGE_SIZE = (8, 8, 3)  # 8x8 px
TEST_PATH = './data/test/*.png'
TRAIN_PATH = './data/train/*.png'


def img2array(filename):
    img = cv2.imread(filename)
    img = img.flatten()
    img[img == 255] = 1
    return img


def array2img(arr):
    arr[arr == -1] = 0
    arr *= 255
    img = np.reshape(arr, IMAGE_SIZE)
    return img


def array2float(arr):
    flt = np.asfarray(arr)
    flt[flt == 0] = -1
    return flt


def prep_plot(images, name):
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(images[0].astype('uint8'))
    ax.set_title(name + ' Test')

    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(images[1].astype('uint8'))
    ax.set_title(name + ' Result')

def test_net(net, test_file):
    test_image = img2array(test_file)
    test = array2float(test_image)

    out = net.sim([test])

    out_image = array2img(out[0])
    test_image = array2img(test)
    prep_plot([test_image, out_image], "test_file")


# tworzenie zbioru do treningu
target = []
for file in glob.glob(TRAIN_PATH):
    array = img2array(file)
    target.append(array)

target = array2float(target)
#print(target)
# tworzenie i trening sieci
net = nl.net.newhop(target)

# testowanie sieci
test_net(net, './data/test/a-test.png')
test_net(net, './data/test/a-rev-test.png')
test_net(net, './data/test/b-test.png')
test_net(net, './data/test/c-test.png')

plt.show()
