import torch
import utils
import torchvision
from torchvision import transforms as T
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import numpy as np

device = ("cuda" if torch.cuda.is_available() else "cpu")

# Class names given by PyTorch's official Docs
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(model, img_path, labels, threshold=0.5):
    """Produces a prediction result from the model for a given threshold."""

    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i]
                for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                for i in list(pred[0]['boxes'].detach().numpy())]

    # a list of indices with score greater than threshold
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x)
                for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    return pred_boxes, pred_class


def object_detection_api(model, img_path, threshold=0.5, rect_th=3, text_size=1.5, text_th=3):
    """Displays detected bounding boxes for a given image."""

    boxes, pred_cls = get_prediction(model, img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],
                      color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0],
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

    plt.figure(figsize=(15,15))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def evaluate(model, dataloader, image_batch_size=8):
    model.eval()

    running_loss = 0.0
    loss_value = 0.0

    for images, targets in dataloader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images)

            # this returned object from the model:
            # len is 4 (so index here), which is probably because of the size of the batch
            # loss_dict[index]['boxes']
            # loss_dict[index]['labels']
            # loss_dict[index]['scores']
            for x in range(image_batch_size):
                loss_value += sum(loss for loss in loss_dict[x]['scores'])

        running_loss += loss_value

    return running_loss


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

traffic_img = cv2.imread("./traffic.jpg")
cv2.imshow("traffic_img", traffic_img)

object_detection_api(model, "traffic.jpg")

# use our dataset and defined transformations
dataset_test = CustomDataset('export', get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset_test)).tolist()
dataset = torch.utils.data.Subset(dataset_test, indices[:-50])

data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)

