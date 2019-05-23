import os
from copy import deepcopy
from xml.etree import ElementTree

def _parse_xml(filename, labels=[]):
    """Parse xml file

    Argument
        filename - filename of the xml file
        labels - a list of target label

    Return
        a tuple, one dict for image annotation, one dict for seen label in image
    """
    image = {}
    seen_label = {}

    dom = ElementTree.parse(filename)

    # image information
    image['filename'] = dom.find('filename').text

    size = dom.find('size')
    image['width'] = int(size.find('width').text)
    image['height'] = int(size.find('height').text)
    image['depth'] = int(size.find('depth').text)

    # objects information in the image
    image['objects'] = []
    for obj in dom.findall('object'):
        obj_info = {}
        obj_info['name'] = obj.find('name').text

        # Filter out unwanted objects
        if obj_info['name'] not in labels:
            continue

        # Trach the number of seening label
        if obj_info['name'] not in seen_label.keys():
            seen_label[obj_info['name']] = 1
        else:
            seen_label[obj_info['name']] += 1

        location = obj.find('bndbox')
        obj_info['xmin'] = int(location.find('xmin').text)
        obj_info['ymin'] = int(location.find('ymin').text)
        obj_info['xmax'] = int(location.find('xmax').text)
        obj_info['ymax'] = int(location.find('ymax').text)
        image['objects'].append(obj_info)

    return image, seen_label

def parse_annotations(annot_dir, image_dir, labels=[]):
    """Parse all annotation files with specific target

    Arguments
        annot_dir - directory path contains all the annotation files
        image_dir - directory path contains all the image files
        labels - a list of target label

    Return
        Return a tuple, the first element is a list of dict about image annotations,
        the second element is a dictionary summarizing the seen lables
    """
    annot_files = os.listdir(annot_dir)

    images = []
    seen_labels = []

    for f in annot_files:
        image, seen_label = _parse_xml(os.path.join(annot_dir, f), labels)

        images.append(image)
        seen_labels.append(seen_label)

    # append img_dir to image's filename
    for image in images:
        image['filename'] = os.path.join(image_dir, image['filename'])

    # Assemble the seen_labels to a single dictionary
    tb_label = {}
    for seen_label in seen_labels:
        for name, count in seen_label.items():
            if name not in tb_label.keys():
                tb_label[name] = count
            else:
                tb_label[name] += count

    return images, tb_label

def update_images_annot(images, shape=(416, 416)):
    """Update images' annotation to new dimension

    In addition updaing the img dimension info, updating object's xmin, ymin,
    xmax, ymax too.

    Arguemnt
        images - a list a image annotations to be updated
        shape - the rescaling shape of all the images

    Return
        a list of new images
    """
    new_images = deepcopy(images)

    for image in new_images:
        old_height = image['height']
        old_width = image['width']
        new_height = shape[0]
        new_width = shape[1]

        x_scale_factor = float(new_width) / old_width
        y_scale_factor = float(new_height) / old_height

        for obj in image['objects']:
            obj['xmin'] = obj['xmin'] * x_scale_factor
            obj['xmax'] = obj['xmax'] * x_scale_factor
            obj['ymin'] = obj['ymin'] * y_scale_factor
            obj['ymax'] = obj['ymax'] * y_scale_factor

        image['height'] = new_height
        image['width'] = new_width

    return new_images

