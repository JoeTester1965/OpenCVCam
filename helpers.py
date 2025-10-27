import numpy as np
import cv2

def hailo_yolo_detection(image, model, model_confidence):
                
    rescaled_image, scale, pad_top, pad_left = resize_with_letterbox(image, model.input_shape[0])
    retval_list = [] 
    inference_result = model(rescaled_image[0])  
    scaled_inference_results = reverse_rescale_bboxes(inference_result.results, scale, pad_top, pad_left,image.shape[:2])
    for result in scaled_inference_results:
        if np.float32(result['score']) > model_confidence:
            result_row = result['label'],np.float32(result['score']), np.floor(result['bbox']).astype(int)
            retval_list.append(result_row)
    return retval_list

def resize_with_letterbox(image, target_shape, padding_value=(0, 0, 0)):
    """
    Resizes an image with letterboxing to fit the target size, preserving aspect ratio.
    
    Parameters:
        image_path (str): Path to the input image.
        target_shape (tuple): Target shape in NHWC format (batch_size, target_height, target_width, channels).
        padding_value (tuple): RGB values for padding (default is black padding).
        
    Returns:
        letterboxed_image (ndarray): The resized image with letterboxing.
        scale (float): Scaling ratio applied to the original image.
        pad_top (int): Padding applied to the top.
        pad_left (int): Padding applied to the left.
    """
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the original image dimensions (height, width, channels)
    h, w, c = image.shape
    
    # Extract target height and width from target_shape (NHWC format)
    target_height, target_width = target_shape[1], target_shape[2]
    
    # Calculate the scaling factors for width and height
    scale_x = target_width / w
    scale_y = target_height / h
    
    # Choose the smaller scale factor to preserve the aspect ratio
    scale = min(scale_x, scale_y)
    
    # Calculate the new dimensions based on the scaling factor
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image to the new dimensions
    resized_image = cv2.resize(image, (new_w, new_h),interpolation=cv2.INTER_LINEAR)
    
    # Create a new image with the target size, filled with the padding value
    letterboxed_image = np.full((target_height, target_width, c), padding_value, dtype=np.uint8)
    
    # Compute the position where the resized image should be placed (padding)
    pad_top = (target_height - new_h) // 2
    pad_left = (target_width - new_w) // 2
    
    # Place the resized image onto the letterbox background
    letterboxed_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_image

    final_image = np.expand_dims(letterboxed_image, axis=0)
    
    # Return the letterboxed image, scaling ratio, and padding (top, left)
    return final_image, scale, pad_top, pad_left

def reverse_rescale_bboxes(annotations, scale, pad_top, pad_left, original_shape):
    """
    Reverse rescales bounding boxes from the letterbox image to the original image, returning new annotations.

    Parameters:
        annotations (list of dicts): List of dictionaries, each containing a 'bbox' (x1, y1, x2, y2) and other fields.
        scale (float): The scale factor used for resizing the image.
        pad_top (int): The padding added to the top of the image.
        pad_left (int): The padding added to the left of the image.
        original_shape (tuple): The shape (height, width) of the original image before resizing.

    Returns:
        new_annotations (list of dicts): New annotations with rescaled bounding boxes adjusted back to the original image.
    """
    orig_h, orig_w = original_shape  # original image height and width
    
    new_annotations = []
    
    for annotation in annotations:
        bbox = annotation['bbox']  # Bounding box as (x1, y1, x2, y2)
        
        # Reverse padding
        x1, y1, x2, y2 = bbox
        x1 -= pad_left
        y1 -= pad_top
        x2 -= pad_left
        y2 -= pad_top
        
        # Reverse scaling
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        
        # Clip the bounding box to make sure it fits within the original image dimensions
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        # Create a new annotation with the rescaled bounding box and the original label
        new_annotation = annotation.copy()
        new_annotation['bbox'] = (x1, y1, x2, y2)
        
        # Append the new annotation to the list
        new_annotations.append(new_annotation)
    
    return new_annotations

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

def opencv_yolo_detection(image, net, confidence, threshold, labels, model_width, model_height):
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

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (model_width, model_height), swapRB=True, crop=False)

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



 
