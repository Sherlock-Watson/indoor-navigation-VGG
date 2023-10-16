import torch.nn as nn
from torchvision.models import vgg16
import torch

device = 'cuda' if torch.cuda.is_initialized() else 'cpu'
torch.device(device)


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
