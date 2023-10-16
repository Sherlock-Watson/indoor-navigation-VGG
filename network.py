import torch.nn as nn
from torchvision.models import resnet, vgg16
import torch
import numpy as np
device = 'cuda' if torch.cuda.is_initialized() else 'cpu'
torch.device(device)


class MLP(nn.Module):
    def __init__(self, inputsize, h1size, h2size, outputsize):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(inputsize, h1size),
                                   nn.ReLU(),
                                   nn.Linear(h1size, h2size),
                                   nn.ReLU(),
                                   nn.Linear(h2size, h2size),
                                   nn.ReLU())
        self.head = nn.Sequential(nn.Linear(h2size + 1024, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, h2size),
                                  nn.ReLU(),
                                  nn.Linear(h2size, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, outputsize))
        self.backbone = resnet.resnet18(pretrained=True, progress=True)
        self.backbone.layer4 = nn.Sequential()
        self.reduce_chn = nn.Sequential(nn.Conv2d(256, 16, kernel_size=1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU())


    # def forward(self, feature, image):
    #     x = self.backbone.conv1(image)
    #     x = self.backbone.bn1(x)
    #     x = self.backbone.relu(x)
    #     x = self.backbone.maxpool(x)
    #
    #     x = self.backbone.layer1(x)
    #     x = self.backbone.layer2(x)
    #     x = self.backbone.layer3(x)
    #     x = self.reduce_chn(x)
    #     b, _, _, _ = x.shape
    #     x = x.view((b, -1))
    #     f = self.model(feature)
    #     out = torch.cat([f, x], axis=-1)
    #
    #     return self.head(out)

class MLP_without_CNN(nn.Module):
    def __init__(self, inputsize, h1size, h2size, outputsize):
        super(MLP_without_CNN, self).__init__()
        self.model = nn.Sequential(nn.Linear(inputsize, h1size),
                                   nn.ReLU(),
                                   nn.Linear(h1size, h2size),
                                   nn.ReLU(),
                                   nn.Linear(h2size, h2size),
                                   nn.ReLU())
        self.head = nn.Sequential(nn.Linear(inputsize + 3*28*28, h1size),
                                  nn.ReLU(),
                                  nn.Linear(h1size, h2size),
                                  nn.ReLU(),
                                  nn.Linear(h2size, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, outputsize))
        # Replace the ResNet backbone with VGG16
        self.backbone = vgg16(pretrained=True)

        # self.backbone = resnet.resnet18(pretrained=True, progress=True)
        # self.backbone.layer4 = nn.Sequential()
        # self.reduce_chn = nn.Sequential(nn.Conv2d(256, 16, kernel_size=1),
        #                                 nn.BatchNorm2d(16),
        #                                 nn.ReLU())


#     def forward(self, feature, image):
# #         print(image.size())
#         x=image.view(-1,image.size()[1]*image.size()[2]*image.size()[3])
#
# #         f = self.model(feature)
#         x_=x
# #         for _ in range(f.size()[0]//x.size()[0]-1):
# #             x_=torch.cat([x_,x],0)
# #         print(f.size(),x_.size())
# #         exit(0)
#         out = torch.cat([feature, x_], axis=-1)
#         if torch.isnan(out).any():
#             print('nan!')
#             exit(1)
#         preds=self.head(out)
# #         print(out[63],preds[63])
#
#         return preds

    def forward(self, x):
        # Implement the forward pass using the VGG16 backbone
        x = self.backbone(x)
        x = self.reduce_chn(x)
        # Continue with the rest of your model
        x = self.model(x)
        output = self.head(x)
        return output


class VGG(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, output_size):
        super(VGG, self).__init__()
        self.feature_extractor = self._initialize_feature_extractor()
        self.classifier = self._initialize_classifier(input_size, h1_size, h2_size, output_size)

    def _initialize_feature_extractor(self):
        # Initialize and configure the feature extractor (e.g., VGG16)
        feature_extractor = vgg16(pretrained=True)
        feature_extractor.classifier = nn.Sequential()  # Remove the fully connected layers
        return feature_extractor

    def _initialize_classifier(self, input_size, h1_size, h2_size, output_size):
        # Initialize the classifier (e.g., fully connected layers)
        classifier = nn.Sequential(
            nn.Linear(input_size, h1_size),
            nn.ReLU(),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
            nn.Linear(h2_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        return classifier

    def forward(self, x):
        # Implement the forward pass
        # features = self.feature_extractor(x)
        output = self.classifier(x)
        return output

