import os
import json
import argparse
import numpy as np

from utils.preprocessing import parse_annotations, update_images_annot

from yolo.frontend import YOLO

argparser = argparse.ArgumentParser(
                description='Train and validate YOLO v2 model')

argparser.add_argument(
        '-c',
        '--conf',
        required=True,
        help='path to configuration file')

def _main(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # Parse annotation of the training set
    # ====================================
    images, labels = parse_annotations(
                                config['train']['train_annot_folder'],
                                config['train']['train_image_folder'],
                                config['model']['labels'])

    train_images = update_images_annot(images, (416, 416))

    # Split the training set and validation set
    train_valid_split = int(0.8*len(train_images))
    np.random.shuffle(train_images)

    valid_images = train_images[train_valid_split:]
    train_images = train_images[:train_valid_split]

    # Print out label information
    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(labels.keys()))

        print('Seen labels:\t', labels, end="\n\n")
        print('Given labels:\t', config['model']['labels'], end="\n\n")
        print('Overlap labels:\t', overlap_labels, end="\n\n")

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = labels.keys()

    # Construct the model
    # ===================
    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                anchors=config['model']['anchors'])

    # Start the training process
    # ==========================
    yolo.train(train_images, # the list of images to train the model
                valid_images, # the list of images to validate the model
                config['train']['nb_epochs'],    # the number of times to repeat the validation set
                config['train']['learning_rate'], # the learning rate
                # loss function related
                config['train']['batch_size'],
                config['train']['object_scale'],
                config['train']['no_object_scale'],
                config['train']['coord_scale'],
                config['train']['class_scale'],
                # path to save the model
                config['train']['saved_weights_name'],
                config['train']['debug'])

    # Train Complete
    print("YOLO TRAINING HAS STOPPED")

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
