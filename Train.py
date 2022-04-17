# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:40:55 2019

@author: Administrator
"""
import scipy.io as sio
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, BatchNormalization, Activation, Input, Concatenate, GlobalAveragePooling1D
from keras.regularizers import l2
import scipy.io as sio
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import LearningRateScheduler
import itertools

import matplotlib.pyplot as plt
from GCResCNN import *


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,  # 这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # 这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    # plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
    plt.figure(1)
    plt.show()





def scheduler(epoch):
    # 每隔5个epoch，学习率减小为原来的1/10
    if epoch % 50== 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


if __name__ == '__main__':

    sample_size = 512
    xx_train = sio.loadmat('unbalanced\\Train.mat')
    xx_test = sio.loadmat('unbalanced\\Test.mat')
    x_train_s1 = xx_train['Train']
    x_test_s1 = xx_test['Test']
    #
    x_train_s1 = x_train_s1.reshape(x_train_s1.shape[0], 512, 1)
    x_test_s1 = x_test_s1.reshape(x_test_s1.shape[0], 512, 1)

    yy_train = sio.loadmat('unbalanced\\Train_label.mat')
    yy_test = sio.loadmat('unbalanced\\Test_label.mat')
    y_train = yy_train['Train_label']
    y_test = yy_test['Test_label']



    # -----------------构建网络-------------------------------------------------
    inputs1 = Input(shape=(sample_size, 1))
    inputs2 = Input(shape=(sample_size, 1))
    inputs3 = Input(shape=(sample_size, 1))

    # model = rcf_model_resnet.resnet_rcf(input_shape=(sample_size, 1))
    # model = rcf_model_vgg.vgg_rcf(input_shape=(sample_size, 1))
    model = vgg_rcf(input_shape=(sample_size, 1))
    # model = rcf_model_vgg.vgg_rcf(input_shape=(sample_size, 1))
    #
    model.summary()
    print("Training")

    # net = Concatenate()([model1.output, model.output, model2.output])
    output = GlobalAveragePooling1D()(model.output)
    outputs = Dense(6, activation='softmax', bias_regularizer=l2(1e-4), kernel_regularizer=l2(1e-4))(output)
    model = Model(inputs=model.input, outputs=outputs)

    reduce_lr = LearningRateScheduler(scheduler)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.005),
                  metrics=['accuracy'])


    import time
    start = time.clock()
    history = model.fit(x_train_s1,
                        y_train,
                        batch_size=256,
                        epochs=200,
                        validation_data=(x_test_s1, y_test), callbacks=[reduce_lr], shuffle=True)
    print(time.clock() - start)
    acc = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.title('Accuracy and Loss')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_accuracy, 'blue', label='val_accuracy')


    plt.legend()
    plt.figure(2)
    plt.show()

    model.save('GC-ResCNN.h5')
    # score = model.evaluate(x_test_s3, y_test)
    # pred = model.predict(x_test_s3)
    import pickle
    # 保存
    with open('GC-ResCNN.txt', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)



