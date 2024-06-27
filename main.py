import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

LABELS_PATH = os.sep.join(["cfg", "coco.names"])
WEIGHT_PATH = os.sep.join(["yolov4.weights"])
CONFIG_PATH = os.sep.join(["cfg", "yolov4.cfg"])


def load_labels():
    with open(LABELS_PATH) as labels_file:
        return labels_file.read().strip().split("\n")


def gen_rand_color():
    labels_size = len(load_labels())
    return np.random.randint(0, 255, size=(labels_size, 3), dtype="uint8")


def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def resizing(image, max_width=600):
    if image.shape[1] > max_width:
        proportion = image.shape[1] / image.shape[0]
        image_width = max_width
        image_height = int(image_width / proportion)

        return cv2.resize(image, (image_width, image_height))

    return cv2.resize(image, (image.shape[1], image.shape[0]))


def get_out_layers(net):
    layer_names = net.getLayerNames()
    out_layers_index = net.getUnconnectedOutLayers()
    out_layer_names = []

    for index in out_layers_index:
        out_layer_names.append(layer_names[index - 1])
        
    return out_layer_names


def start_processing(img):
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHT_PATH)
    out_layer_names = get_out_layers(net)

    image = cv2.imread(img)
    # image_cp = image.copy()

    (h, w) = image.shape[:2]

    image_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(image_blob)

    layer_outputs = net.forward(out_layer_names)

    threshold = 0.5
    threshold_nms = 0.3
    boxes = []
    confidences = []
    classes_id = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > threshold:
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classes_id.append(class_id)

    objects = cv2.dnn.NMSBoxes(boxes, confidences, threshold, threshold_nms)

    if len(objects) > 0:
        for i in objects:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # obj = image_cp[y:y + h, x:x + w]
            color = [int(c) for c in gen_rand_color()[classes_id[i]]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(load_labels()[classes_id[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    show_image(image)


if __name__ == '__main__':
    img = os.sep.join(["data", "humans.jpg"])
    start_processing(img)


