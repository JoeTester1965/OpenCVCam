def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def process_events(yolyo_candidate, camera_stream, camera_name):
     
    logger.debug("%s : Yolo candidate being processed, outstanding queue size %d")
    Width,Height = camera_stream.last_frame.shape
    class_ids = []
    confidences = []
    boxes = []
    DNNWidth = int(dnn_config['dnn_width']) 
    DNNHeight = int(dnn_config['dnn_height']) 
    blob = cv2.dnn.blobFromImage(yolyo_candidate, 1/255, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    have_best_detection = False

    #
    # To do, get best dectecion after below is gpu optimised!
    # ignore if in blacklist, draw and log all infio if in whitelist, log debug if in greylist
    #

    for out in outs:
        #
        # 
        #
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > float(dnn_config['dnn_confidence']):
                #
                have_best_detection = True
                #
                center_x = int(detection[0] * DNNWidth)
                center_y = int(detection[1] * DNNHeight)
                w = int(detection[2] * DNNWidth)
                h = int(detection[3] * DNNHeight)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                indices = cv2.dnn.NMSBoxes(boxes, confidences, float(dnn_config['bounding_box_score_threshold']), float(dnn_config['bounding_box_nms_threshold']))

                for i in indices:
                    try:
                        box = boxes[i]
                    except:
                        i = i[0]
                        box = boxes[i]
                    
                    y_scale = float(Width)/float(dnn_config['dnn_width'])
                    x_scale = float(Height)/float(dnn_config['dnn_height'])
                    x = box[0] * x_scale
                    y = box[1] * y_scale
                    w = box[2] * x_scale
                    h = box[3] * y_scale
                    
                    label = str(classes[class_id])
                    color = (0,0,255)
                    cv2.rectangle(yolyo_candidate, (round(x),round(y)), (round(x+w),round(y+h)), color, 2)
                    cv2.putText(yolyo_candidate, label, (round(x)-10,round(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    logger.info("%s : Detected %s at %d,%d", camera_name, label, (x + (x+w))/2, (y + (y+h))/2)

    if have_best_detection:
        if int(dnn_config['display_object_detect_debug']):
            cv2.imshow("DNN object detection", yolyo_candidate)
            cv2.waitKey(1)

#dnn_config
dnn_config=dict(config['dnn']) 

with open(dnn_config['classes'], 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(dnn_config['weights'], dnn_config['config'])

