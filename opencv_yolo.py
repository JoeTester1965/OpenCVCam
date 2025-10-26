import numpy as np
import cv2

def filter_outputs(layer_output, confidence):

    box_xywh = np.array(layer_output[:, :4])
    box_confidence = np.array(layer_output[:, 4]).reshape(layer_output.shape[0], 1)
    box_class_probs = np.array(layer_output[:, 5:])

    box_scores = box_confidence * box_class_probs
    box_class = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)

    # pick up boxes with box_class_scores that are higher than confidence
    filtering_mask = box_class_scores >= confidence
    class_filtered = box_class[filtering_mask]
    score_filtered = box_class_scores[filtering_mask]
    xywh_filtered = box_xywh[np.nonzero(filtering_mask)]

    return (xywh_filtered, score_filtered, class_filtered)

def iou(box1, box2): # Calculate Itersection over Union

    # get the area of intersection, co-ords are top left and bottom right of boxes
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # get the area of union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    # get iou
    iou = inter_area / union_area

    return iou


def yolo_non_max_supression(boxes, scores, confidence_threshold, iou_threshold):
    """ 
        boxes : Array of coordinates of boxes (x1, y1, x2, y2)
        scores : Array of confidence scores with respect to boxes
        confidence_threshold : Threshold of the score to keep
        iou_threshold : Threshold of IoU to keep

        Return : Indices of boxes and scores to be kept
    """

    # Sort scores in descending order
    sorted_idx = np.argsort(scores)[::-1]

    remove = []
    for i in np.arange(len(scores)):
        # if the score is already removed, skip it
        if i in remove:
            continue
        # if the score is blow the confidence, add it to the remove list
        if scores[sorted_idx[i]] < confidence_threshold:
            remove.append(i)
            continue

        for j in np.arange(i+1, len(scores)): # start the search from the next score
            if j in remove:
                continue
            if scores[sorted_idx[j]] < confidence_threshold:
                remove.append(j)
                continue

            # calculate IoU of two boxes.
            # If IoU is more than the threshold, add the box with the lower score to the remove list
            overlap = iou(boxes[sorted_idx[i]], boxes[sorted_idx[j]])
            if overlap > iou_threshold:
                remove.append(j)

    sorted_idx = np.delete(sorted_idx, remove)
    return sorted(sorted_idx)

def rescale_box_coord(boxes, width, height):
    """ Rescale bounding boxes to fit the original image, and calculate the coordinates
        of the top left corner and the bottom right corner.
        boxes : Array of (x,y,w,h) of the box
        width : Width of the original image
        height : Height of the original image
    """
    boxes_orig = boxes * np.array([width, height, width, height])
    boxes_orig[:, 0] -= boxes_orig[:, 2] / 2
    boxes_orig[:, 1] -= boxes_orig[:, 3] / 2

    # make an array of box coordinates.
    # boxes_coord = array of [[x1, y1, x2, y2], ...]: where (x1, y1) = upper left, (x2, y2) = lower right
    boxes_coord = boxes_orig
    # set x2 = x1 + w
    boxes_coord[:, 2] = boxes_orig[:, 0] + boxes_orig[:, 2]
    # set y2 = y1 + h
    boxes_coord[:, 3] = boxes_orig[:, 1] + boxes_orig[:, 3]

    np.floor(boxes_coord)

    return boxes_coord

def opencv_yolo_detection(image, net, confidence, threshold, labels, colors):
    """ Apply YOLO object detection on a image_file.
        image_filename : Input image numopy array
        net : YOLO v3 network object
        confidence : yoloy confidence threshold
        threshold : Intersection over Union (IoU) threshold for Non Maximum Suppression (NMS)
        labels : Class labels specified in coco.names
        colors : Colors assigned to the classes
    """
    retval = []

    if image is not None:

        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)

        ln = net.getLayerNames()
        ln_out = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        #ln_out = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        layerOutputs = net.forward(ln_out)

        boxes = []
        scores = []
        classes = []
        for output in layerOutputs:

            (xywh_filterd, score_filtered, class_filtered) = filter_outputs(output, confidence)

            boxes.append(xywh_filterd)
            scores.append(score_filtered)
            classes.append(class_filtered)

        boxes = np.vstack([r for r in boxes])
        scores = np.concatenate([r for r in scores], axis=None)
        classes = np.concatenate([r for r in classes], axis=None)

        boxes_coord = rescale_box_coord(boxes, W, H)
        nms_idx = yolo_non_max_supression(boxes_coord, scores, confidence, threshold)

        retval_list = []   

        if len(nms_idx) > 0:
    
            for i in nms_idx:
                retval  = labels[classes[i]], scores[i], np.floor(boxes_coord[i]).astype(int)
                retval_list.append(retval)
        
        return retval_list



 
