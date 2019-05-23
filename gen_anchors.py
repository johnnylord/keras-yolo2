import json
import argparse
import numpy as np

from sklearn.cluster import KMeans

from utils.preprocessing import parse_annotations, update_images_annot

# Construct arguement parser
argparser = argparse.ArgumentParser()

argparser.add_argument(
        '-c',
        '--conf',
        required=True,
        help='path to configuration file')

argparser.add_argument(
        '-a',
        '--anchors',
        required=True,
        help='number of anchors to use')

def _main(args):
    config_path = args.conf
    num_anchors = int(args.anchors)

    # Load configuration
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # Get training data information
    train_images, train_labels = parse_annotations(
                                    config['train']['train_annot_folder'],
                                    config['train']['train_image_folder'],
                                    config['model']['labels'])

    input_size = config['model']['input_size']

    # Run K_means to find the anchors
    new_train_images = update_images_annot(
                            train_images,
                            (input_size, input_size))

    cell_width = input_size // 13
    cell_height = input_size // 13

    bboxes = []
    for image in new_train_images:
        for obj in image['objects']:
            width = float(obj['xmax'] - obj['xmin']) / cell_width
            height = float(obj['ymax'] - obj['ymin']) / cell_height
            bboxes.append([width, height])
    bboxes = np.array(bboxes)

    clf = KMeans(n_clusters=num_anchors)
    clf.fit(bboxes)

    box_list = list(clf.cluster_centers_)
    box_list.sort(key=lambda x: x[0]*x[1])

    print("Anchor Boxes(width, height, ...):")
    print(np.reshape(np.array(box_list), -1))

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
