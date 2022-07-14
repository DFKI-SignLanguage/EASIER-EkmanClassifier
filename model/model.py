import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


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
