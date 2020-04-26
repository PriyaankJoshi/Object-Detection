from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
import cv2
import numpy as np

root = Tk()
root.geometry("500x400")

cfg_file = r"C:\Users\PRIYAANK\darkflow\cfg\yolo.cfg"
classes_file = r"C:\Users\PRIYAANK\darkflow\cfg\coco.names"
weights_file = r"C:\Users\PRIYAANK\darkflow\bin\yolo.weights"

classes = None
with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def select_files():
        file = askopenfile(mode = 'r', filetypes = [('Model Files', '.*')])
        return file.name

def select_models():
        cfg_file = r"" + str(Button(root, text="Select cfg", command=lambda:select_files()).pack(side=LEFT, pady=10))
        weights_file = r"" + str(Button(root, text="Select models", command=lambda:select_files()).pack(side=LEFT, pady=10))
        classes_file = r"" + str(Button(root, text="Select classes", command=lambda:select_files()).pack(side=LEFT, pady=10))

def open_file():
        file = askopenfile(mode = 'r', filetypes = [('Image files', ('.png', '.jpg', '.jpeg'))])
        
        image = cv2.imread(file.name)
        
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.000392

        if Width > 900 and Height > 800:
                #dim = (int(image.shape[1] * 0.2), int(image.shape[0] * 0.2))
                dim = (550, 450)
                image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                print(image.shape)
        
        net = cv2.dnn.readNet(weights_file, cfg_file)
        
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        before = cv2.imread(file.name)
        cv2.imshow("Image Before Object Detection", before)
        
        cv2.imshow("Image After Object Detection", image)
        cv2.waitKey()
            
        cv2.imwrite("object-detection.jpg", image)
        cv2.destroyAllWindows()

btn = Button(root, text="Select Image", command = lambda:open_file())
btn.pack(side = TOP, pady = 10)

check = IntVar()
c = Checkbutton(root, text="Choose Different Models", variable = check, command = select_models)
c.pack(side = LEFT, pady = 10)


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

mainloop()
