import torch
import torch.nn as nn
import pretrainedmodels

# Resnet34 Custom implementation

class CustomResnet34:
    def __init__(self, training):
        super(CustomResnet34, self).__init__()
        self.model_name = "resnet34"
        if(training):
            self.model = pretrainedmodels.__dict__[self.model_name](pretrained = "imagenet")
        else:
            self.model = pretrainedmodels.__dict__[self.model_name](pretrained = None)
        
        # Bengali Ai Custom Layers
        self.grapheme_layer = nn.Linear(in_features = 512, out_features = 168)
        self.vowel_layer = nn.Linear(in_features = 512, out_features = 11)
        self.consonant_layer = nn.Linear(in_features = 512, out_features = 7)

    def verbose(self):
        print(self.model)
    
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)

        grapheme_layer = self.grapheme_layer(x)
        vowel_layer = self.vowel_layer(x)
        consonant_layer = self.consonant_layer(x)

        return grapheme_layer, vowel_layer, consonant_layer

if __name__ == "__main__":
    obj = CustomResnet34(True)
