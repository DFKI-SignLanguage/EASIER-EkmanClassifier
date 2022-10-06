import torch.nn as nn
from base import BaseModel
from torchvision import models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class VggModel(BaseModel):
    def __init__(self, n_classes=8, fine_tune=True, pretrained=True):
        super(VggModel, self).__init__()

        # load the pretrained model
        vgg11_bn = models.vgg11_bn(pretrained=pretrained)

        set_parameter_requires_grad(vgg11_bn.features, fine_tune)

        vgg11_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )
        self.base_model = vgg11_bn
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        out = self.sigm(x)
        return out


class ResnetModel(BaseModel):
    def __init__(self, n_classes=8):
        super(ResnetModel, self).__init__()

        resnet50 = models.resnet50(pretrained=True)

        modules = list(resnet50.children())[:-1]  # delete the last fc layer.
        self.features = nn.Sequential(*modules)

        self.classifier = nn.Sequential(
            nn.Linear(2048, n_classes))

        # self.sigm = nn.Sigmoid()

    def forward(self, image):
        features = self.features(image).squeeze(-1).squeeze(-1)

        out = self.classifier(features)

        return out


class MobilenetModel(BaseModel):
    def __init__(self, n_classes=8):
        super(MobilenetModel, self).__init__()

        mobilenet_mod = models.mobilenet_v2(pretrained=True)
        self.model = mobilenet_mod
        self.model.classifier = nn.Linear(1280, n_classes)

    def forward(self, image):
        out = self.model(image)
        return out


class EfficientNetModel(BaseModel):
    def __init__(self, n_classes=8):
        super(EfficientNetModel, self).__init__()

        mobilenet_mod = models.efficientnet_b0(pretrained=True)
        self.model = mobilenet_mod
        self.model.classifier = nn.Linear(1280, n_classes)

    def forward(self, image):
        out = self.model(image)
        return out
