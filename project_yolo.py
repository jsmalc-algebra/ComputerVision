from sympy import false
from ultralytics import YOLO


def pretrain():
    model = YOLO("yolo26n-cls.pt")  # loading a pretrained model
    model.train(
        data='dataset',
        epochs=25,
        imgsz=224,
        batch=32,
        name='non_pretrained_classifier_v26_',
        pretrained=false,
    )


if __name__ == '__main__':
    pretrain()
