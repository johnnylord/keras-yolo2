# keras-yolo2

## Input data
Input data would be a image of size (416, 416).

## Output data
Output data would be a four dimension tensor, (S, S, B, 4+1+CLASSES).
For every grid cell in the image, it will predict B bounding boxes, each bounding box containing the following information.
- bx := the x coordinate of the bounding box. # unit, grid cell
- by := the y coordinate of the bounding box. # unit, grid cell
- bw := the width of the bounding box. # unit, grid cell
- bh := the height of the bounding box. # unit, grid cell
- pc := the confidence of the bounding box. (Pr(object) * IoU(pred-box, true-box))
- classes := a vector of class probability

## Loss Contribution

