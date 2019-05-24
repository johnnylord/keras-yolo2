from tensorflow import keras

import cv2
import numpy as np


class YoloDataGenerator(keras.utils.Sequence):
    """Helper class for generating data for yolo in batch format"""

    def __init__(self, images, config, norm, shuffle=True):
        self.images = images
        self.config = config
        self.norm = norm
        self.shuffle = shuffle

    def __len__(self):
        """Return number of examples divided by batch size"""
        return int(np.ceil(len(self.images)/float(self.config['BATCH_SIZE'])))

    def __getitem__(self, idx):
        # Specify the range of the batch
        lower_bound = int(idx * self.config['BATCH_SIZE'])
        upper_bound = int(lower_bound + self.config['BATCH_SIZE'])

        # Make the range stay in the sample size
        if upper_bound > len(self.images):
            upper_bound = len(self.images)
            lower_bound = upper_bound - self.config['BATCH_SIZE']

        # Setup the input/output batch placeholder
        input_batch = np.zeros((upper_bound-lower_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))
        true_box_batch = np.zeros((upper_bound-lower_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))
        output_batch = np.zeros((upper_bound-lower_bound, self.config['GRID_H'], self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))

        # Proces each image per iteration
        for n_sample, image in enumerate(self.images[lower_bound:upper_bound]):
            # Read in the image file & resize the image to standard size
            input_image = cv2.imread(image['filename'])
            input_image = cv2.resize(input_image, (self.config['IMAGE_H'], self.config['IMAGE_W']))

            # normalize the image data
            input_batch[n_sample] = self.norm(input_image)

            # Process each object in the image per iteration
            true_box_index = 0
            for obj in image['objects']:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    # Original center position of the object
                    orig_cx = (obj['xmax'] + obj['xmin']) * 0.5
                    orig_cy = (obj['ymax'] + obj['ymin']) * 0.5

                    # Convert the original center position in grid view
                    cx = orig_cx / (self.config['IMAGE_W'] / float(self.config['GRID_W']))
                    cy = orig_cy / (self.config['IMAGE_H'] / float(self.config['GRID_H']))

                    # Original dimension of the bbox related the object
                    orig_box_width = obj['xmax'] - obj['xmin']
                    orig_box_height = obj['ymax'] - obj['ymin']

                    # Convert the dimension of the bbox in grid view
                    box_width = orig_box_width / (self.config['IMAGE_W'] / float(self.config['GRID_W']))
                    box_height = orig_box_height / (self.config['IMAGE_H'] / float(self.config['GRID_H']))

                    # Box information
                    box = [cx, cy, box_width, box_height]

                    # Get the index of the cell grid
                    grid_x = int(np.floor(cx))
                    grid_y = int(np.floor(cy))

                    # Get the index of the best anchor box
                    box_id = self._get_box_predictor_id((box_height, box_width))

                    # Get the index of the object of specific class
                    obj_id = self.config['LABELS'].index(obj['name'])

                    # Assign the ground truth
                    if output_batch[n_sample, grid_y, grid_x, box_id, 4] == 0:
                        output_batch[n_sample, grid_y, grid_x, box_id, 0:4] = box
                        output_batch[n_sample, grid_y, grid_x, box_id, 4] = 1.
                        output_batch[n_sample, grid_y, grid_x, box_id, 5+obj_id] = 1.

                        # assign the true box to true_box_batch
                        true_box_batch[n_sample, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                else:
                    print(image['filename'])

        return [input_batch, true_box_batch], output_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

    def _get_box_predictor_id(self, shape):
        anchors = np.reshape(np.array(self.config['ANCHORS']), (-1, 2))

        box_id = 0
        min_diff_area = 10000
        for i, anchor in enumerate(anchors):
            diff_height = abs(anchor[0]-shape[0])
            diff_width = abs(anchor[1]-shape[1])
            diff_area = diff_height*diff_width
            if diff_area < min_diff_area:
                min_diff_area = diff_area
                box_id = i

        return box_id
