import cv2
import numpy as np
import math

def prepare_image(image, target_size=(300, 300), target_layout="NCWH"):

    # Resize image, [H, W, C] -> [300, 300, C]
    image_copy = cv2.resize(image, target_size)

    # Swap axes, [H, W, C] -> [C, H, W]
    if target_layout == "NCHW":
        image_copy = np.swapaxes(image_copy, 0, 2)
        image_copy = np.swapaxes(image_copy, 1, 2)
    
    # Expand dimensions, [1, C, H, W]
    image_copy = np.expand_dims(image_copy, 0)

    return image_copy

def draw_bounding_boxes(image, detections, classes, threshold=0.5, box_color=(255, 0, 0)):

    image_copy = np.copy(image)
    res_list = []
    # Get image dimensions
    image_height = image_copy.shape[0]
    image_width = image_copy.shape[1]

    # Iterate through detections
    no_detections = detections.shape[2]
    x = []
    for i in range(no_detections):
        detection = detections[0, 0, i]
        # Get class text
        class_ = classes[str(int(detection[1]))]

        if class_ == 'person':
            # Skip detections with confidence below threshold
            confidence = detection[2]
            if confidence < threshold:
                continue

            # Draw bounding box
            x_min = int(detection[3]*image_width)
            y_min = int(detection[4]*image_height)

            x_max = int(detection[5]*image_width)
            y_max = int(detection[6]*image_height)

            centroid = (int((x_min + x_max)/2), int(y_min))
            #print(center[0])
            cv2.circle(image_copy, (centroid[0],centroid[1]), radius=5, color=(0, 255, 0), thickness=-1)
            x.append(centroid)
            top_left = (x_min, y_min)
            bottom_right = (x_max, y_max)

            cv2.rectangle(image_copy, top_left, bottom_right, box_color, 2)

            # Draw text background
            text_size = cv2.getTextSize(class_, cv2.FONT_HERSHEY_PLAIN, 2, 1)
            
            top_left_a = (x_min, y_max-text_size[0][1])
            bottom_right_b = (x_min+text_size[0][0], y_max)

            cv2.rectangle(image_copy, top_left_a, bottom_right_b, box_color, cv2.FILLED)

            # Draw text
            cv2.putText(image_copy, class_, (x_min,y_max), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    dist = math.sqrt((x[i][0] - x[j][0])**2 + (x[i][1] - x[j][1])**2)
                    if dist < 250:
                        image_copy = cv2.line(image_copy, x[i], x[j], (0, 0, 255), 4)
                        cv2.rectangle(image_copy, top_left, bottom_right, (0, 0, 255), 2)
                        print('Detected at: ', x[i], x[j])

    return image_copy