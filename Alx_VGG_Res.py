# 实战 迁移学习 VGG19、ResNet50、InceptionV3 实践 猫狗大战 问题
# https://blog.csdn.net/pengdali/article/details/79050662

# -*- coding: utf-8 -*-
import os
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from keras.applications.xception import Xception


class PowerTransferMode:
    # 数据准备
    def DataGen(self, dir_path, img_row, img_col, batch_size, is_train):
        if is_train:
            datagen = ImageDataGenerator(rescale=1. / 255,
                                         zoom_range=0.25, rotation_range=15.,
                                         channel_shift_range=25., width_shift_range=0.02, height_shift_range=0.02,
                                         horizontal_flip=True, fill_mode='constant')
        else:
            datagen = ImageDataGenerator(rescale=1. / 255)

        generator = datagen.flow_from_directory(
            dir_path, target_size=(img_row, img_col),
            batch_size=batch_size,
            # class_mode='binary',
            shuffle=is_train)

        return generator

    # ResNet模型
    def ResNet50_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=3, img_rows=224, img_cols=224, RGB=True,
                       is_plot_model=False):
        color = 3 if RGB else 1
        base_model = ResNet50(weights='imagenet', include_top=False, pooling=None,
                              input_shape=(img_rows, img_cols, color),
                              classes=nb_classes)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # 添加自己的全链接分类层
        x = Flatten()(x)
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘制模型
        if is_plot_model:
            plot_model(model, to_file='resnet50_model.png', show_shapes=True)

        return model

    # VGG模型
    def VGG16_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=3, img_rows=224, img_cols=224, RGB=True,
                    is_plot_model=False):
        color = 3 if RGB else 1
        base_model = VGG16(weights='imagenet', include_top=False, pooling=None, input_shape=(img_rows, img_cols, color),
                           classes=nb_classes)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # 添加自己的全链接分类层
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘图
        if is_plot_model:
            plot_model(model, to_file='vgg16_model.png', show_shapes=True)

        return model

    # InceptionV3模型
    def InceptionV3_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=3, img_rows=224, img_cols=224, RGB=True,
                          is_plot_model=False):
        color = 3 if RGB else 1
        base_model = InceptionV3(weights='imagenet', include_top=False, pooling=None,
                                 input_shape=(img_rows, img_cols, color),
                                 classes=nb_classes)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # 添加自己的全链接分类层
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘图
        if is_plot_model:
            plot_model(model, to_file='inception_v3_model.png', show_shapes=True)

        return model

    # 训练模型
    def train_model(self, model, epochs, train_generator, steps_per_epoch, validation_generator, validation_steps,
                    model_url, is_load_model=False):
        # 载入模型
        if is_load_model and os.path.exists(model_url):
            model = load_model(model_url)

        history_ft = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=200,  # epochs=epochs
            validation_data=validation_generator,
            validation_steps=validation_steps)
        # 模型保存
        model.save(model_url, overwrite=True)
        return history_ft

    # 画图
    def plot_training(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'orangered', label="Training acc")
        plt.plot(epochs, val_acc, 'springgreen', label="Validation acc")
        plt.title('Training and validation accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.figure()
        plt.plot(epochs, loss, 'orangered', label="Training loss")
        plt.plot(epochs, val_loss, 'springgreen', label="Validation loss")
        plt.title('Training and validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()


if __name__ == '__main__':
    image_size = 224
    batch_size = 20

    transfer = PowerTransferMode()

    # 得到数据
    train_generator = transfer.DataGen('./chinese_herbal_slices/train', image_size, image_size, batch_size, True)
    validation_generator = transfer.DataGen('./chinese_herbal_slices/test', image_size, image_size, batch_size, False)

    # VGG16
    # model = transfer.VGG16_model(nb_classes=3, img_rows=image_size, img_cols=image_size, is_plot_model=False)
    # history_ft = transfer.train_model(model, 10, train_generator, 200, validation_generator, 60, 'vgg16_model_weights.h5', is_load_model=False)

    # ResNet50
    # model = transfer.ResNet50_model(nb_classes=3, img_rows=image_size, img_cols=image_size, is_plot_model=False)
    # history_ft = transfer.train_model(model, 10, train_generator, 200, validation_generator, 60,
    #                                   'resnet50_model_weights.h5', is_load_model=False)

    # InceptionV3
    model = transfer.InceptionV3_model(nb_classes=3, img_rows=image_size, img_cols=image_size, is_plot_model=False)
    history_ft = transfer.train_model(model, 10, train_generator, 200, validation_generator, 60, 'inception_v3_model_weights.h5', is_load_model=False)

    # 训练的acc_loss图
    transfer.plot_training(history_ft)
    plt.show()



# 可视化分类热力图

# # 为VGG16模型预处理一张输入图像
# from keras import backend as K
# from keras.preprocessing import image
# from keras.applications.vgg16 import  preprocess_input,decode_predictions
# import numpy as np
#
# img_path = './chinese_herbal_slices/DHO1.jpg'
# img = image.load_img(img_path,target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

#
# # 应用Grad-CAM算法
# dendrobium_huoshanense_output = model.output[:, 386]
# last_conv_layer = model.get_layer('block5_conv3')
# grads = K.gradients(dendrobium_huoshanense_output, last_conv_layer)[0]
# pooled_grads = K.mean(grads, axis=(0, 1, 2))
# iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
# pooled_grads_value, conv_layer_output_value = iterate([x])
# for i in range(512):
#     conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#
# heatmap = np.mean(conv_layer_output_value, axis=-1)
#
# # 热力图后处理
# heatmap = np.maximum(heatmap, 0)
# heatmap /= np.max(heatmap)
# plt.matshow(heatmap)
#
# # 将热力图与原始图像叠加
# import cv2
# img = cv2.imread(img_path)
# heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# heatmap = np.unit8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# superimposed_img = heatmap * 0.4 + img
# plt.show()
# cv2.imwrite('./中药识别 训练结果/dendrobium_huoshanense.jpg', superimposed_img)
#



# import tensorflow as tf
# import numpy as np
#
# trX = np.linspace(-1, 1, 100)
# trY = trX + np.random.randn(*trX.shape) * 0.33
#
# teX = np.linspace(2, 3, 100)
# teY = teX
#
# X = tf.placeholder(tf.float32, [100, ], name="input_x")
# Y = tf.placeholder(tf.float32, [100, ], name="output_y")
# X1 = tf.placeholder(tf.float32, [100, ], name="input_x1")
# Y1 = tf.placeholder(tf.float32, [100, ], name="output_y1")
#
#
# def model(X, w):
#     return tf.multiply(X, w)
#
#
# w = tf.Variable(0.1, name='weights')
#
# with tf.name_scope("cost"):
#     y_model = model(X, w)
#     cost = tf.reduce_mean(tf.square(Y - y_model))
# tf.summary.scalar('loss', cost)
#
# with tf.name_scope("train"):
#     train_op = tf.train.AdamOptimizer(0.01).minimize(cost)
#
# with tf.name_scope("test_cost"):
#     y_model = model(X, w)
#     test_cost = tf.reduce_mean(tf.square(Y - y_model))
# tf.summary.scalar('test_loss', test_cost)
#
# merged = tf.summary.merge_all()
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     summary_writer = tf.summary.FileWriter('./log/train', sess.graph)
#     summary_writer1 = tf.summary.FileWriter('./log/test')
#     for i in range(1000):
#         feed_dict = {}
#         if i % 100 == 0:
#             print(sess.run(cost, feed_dict={X: trX, Y: trY}))
#             print(sess.run(test_cost, feed_dict={X: teX, Y: teY}))
#             summary, _ = sess.run([merged, test_cost], feed_dict={X: teX, Y: teY})
#             summary_writer1.add_summary(summary, i)
#         else:
#             summary, _ = sess.run([merged, train_op], feed_dict={X: trX, Y: trY})
#             summary_writer.add_summary(summary, i)
#     print(sess.run(w))
#     summary_writer.close()
#     summary_writer1.close()
