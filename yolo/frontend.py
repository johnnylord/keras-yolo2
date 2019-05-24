import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Reshape, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import Model

from yolo.backend import TinyYoloFeature

from utils.generator import YoloDataGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class YOLO(object):
    """YOLO network"""

    def __init__(self, backend, input_size, labels, anchors, max_box_per_image):
        self.input_size = input_size
        self.labels = labels
        self.anchors = anchors

        self.nb_class = len(self.labels)
        self.nb_box = len(self.anchors) // 2

        self.max_box_per_image = max_box_per_image

        # Construct the model
        # ===================
        input_image = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))

        # Feature extraction layer
        if backend == 'Full Yolo':
            self.feature_extractor = FullYoloFeature(self.input_size)
        elif backend == 'Tiny Yolo':
            self.feature_extractor = TinyYoloFeature(self.input_size)
        else:
            raise Exception('YOLO does not support {} backend'.format(backend))

        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()
        features = self.feature_extractor.extract(input_image)

        # Object detection layer
        output = Conv2D(self.nb_box * (4+1+self.nb_class),
                        (1,1), strides=(1,1),
                        padding='same',
                        name='DetectionLayer')(features)

        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4+1+self.nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        # Plug input and output into the model
        self.model = Model([input_image, self.true_boxes], output)

        # Initialize the weights of the detection layer
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape)
        new_bias = np.random.normal(size=weights[1].shape)

        layer.set_weights([new_kernel, new_bias])

        # MODEL SUMMARY
        self.model.summary()

    def _custom_loss(self, y_true, y_pred):
        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)), tf.float32)
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])

        # Adjust prediction
        # =================
        # adjust x & y (batch, grid_h, grid_w, nb_box, [x, y])
        pred_box_xy = tf.cast(tf.sigmoid(y_pred[...,:2]) + cell_grid, tf.float32)

        # adjust w & h (batch, grid_h, grid_w, nb_box, [w,h])
        pred_box_wh = tf.cast(tf.exp(tf.sigmoid(y_pred[...,2:4])) * np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2]), tf.float32)

        # adjust confidence (batch, grid_h, grid_w, nb_box)
        pred_box_conf = tf.cast(tf.sigmoid(y_pred[..., 4]), tf.float32)

        # adjust class probabilities (batch, grid_h, gird_w, nb_box, [classes])
        pred_box_class = tf.cast(tf.nn.softmax(y_pred[..., 5:]), tf.float32)

        # Adjust ground truth
        # ===================
        # adjust x & y (batch, grid_h, grid_w, nb_box, [x, y])
        true_box_xy = tf.cast(y_true[..., :2], tf.float32)

        # adjust w & h (batch, grid_h, grid_w, nb_box, [w, h])
        true_box_wh = tf.cast(y_true[..., 2:4], tf.float32)

        # adjust confidence (batch, grid_h, grid_w, nb_box)
        true_wh_half = true_box_wh / 2.0
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.0
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        true_box_conf = tf.cast(iou_scores * y_true[..., 4], tf.float32)

        # adjust class probability (batch, grid_h, grid_w, nb_box, [classes])
        true_box_class = tf.cast(y_true[..., 5:], tf.float32)

        # Determine the mask
        # ==================
        # coordinate mask
        coord_mask = tf.cast(tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale, tf.float32)

        # confidence mask
        true_xy = tf.cast(self.true_boxes[..., 0:2], tf.float32)
        true_wh = tf.cast(self.true_boxes[..., 2:4], tf.float32)

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = tf.cast(best_ious > 0.6, tf.float32) * (1 - y_true[..., 4]) * self.no_object_scale

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        # class mask
        class_mask = tf.cast(tf.expand_dims(y_true[..., 4], axis=-1) * self.class_scale, tf.float32)

        # Finalize the loss
        # =================
        nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0., tf.float32))
        nb_conf_box = tf.reduce_sum(tf.cast(conf_mask > 0., tf.float32))
        nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0., tf.float32))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)*coord_mask) / nb_coord_box / 2.
        loss_wh = tf.reduce_sum(tf.square(tf.sqrt(true_box_wh)-tf.sqrt(pred_box_wh))*coord_mask) / nb_coord_box / 2.
        loss_conf = tf.reduce_sum(tf.exp(tf.square(true_box_conf-pred_box_conf)) * conf_mask) / nb_conf_box / 2.
        loss_class = tf.reduce_sum(tf.square(true_box_class-pred_box_class) * class_mask) / nb_class_box / 2.

        return loss_xy + loss_wh + loss_conf + loss_class

    def train(self, train_images, # the list of images to train the mode
                    valid_images, # the list of images to validate the model
                    nb_epochs,    # the number of times to repeat the validation set
                    learning_rate, # the learning rate
                    # loss function related
                    batch_size,
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    class_scale,
                    # path to save the model
                    saved_weights_name,
                    debug=False):

        self.batch_size = batch_size
        self.debug = debug

        # Coefficient used in loss function
        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale

        # Construct train and validate data generator
        # ============================================
        generator_config = {
            'IMAGE_H'   : self.input_size,
            'IMAGE_W'   : self.input_size,
            'GRID_H'    : self.grid_h,
            'GRID_W'    : self.grid_w,
            'BOX'       : self.nb_box,
            'LABELS'    : self.labels,
            'ANCHORS'   : self.anchors,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
            'BATCH_SIZE': self.batch_size
        }

        train_generator = YoloDataGenerator(train_images,
                                            generator_config,
                                            norm=self.feature_extractor.normalize)
        valid_generator = YoloDataGenerator(valid_images,
                                            generator_config,
                                            norm=self.feature_extractor.normalize)
        # Compile the model
        # =================
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self._custom_loss, optimizer=optimizer)

        # Make a few callbacks
        # ====================
        early_stop = EarlyStopping(monitor='val_loss',
                                min_delta=0.001,
                                patience=5,
                                mode='min',
                                verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='min',
                                    period=1)
        tensorboard = TensorBoard(log_dir='logs',
                                histogram_freq=0,
                                write_graph=True,
                                write_images=False)

        # Start the training process
        # ==========================
        self.model.fit_generator(generator      = train_generator,
                                steps_per_epoch  = len(train_generator),
                                validation_data  = valid_generator,
                                validation_steps = len(valid_generator),
                                epochs           = nb_epochs,
                                verbose          = 1,
                                callbacks        = [early_stop, checkpoint, tensorboard])
