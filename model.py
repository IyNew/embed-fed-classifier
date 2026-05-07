import importlib
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import ViT


def _is_transient_torchvision_import_error(exc):
    message = str(exc)
    return (
        "partially initialized module 'torchvision'" in message
        or ("torchvision" in message and "extension" in message)
    )


def _clear_torchvision_modules():
    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            sys.modules.pop(name, None)


def _get_torchvision_builder(builder_name, attempts=3):
    last_exc = None
    for attempt in range(attempts):
        try:
            models = importlib.import_module("torchvision.models")
            return getattr(models, builder_name)
        except (AttributeError, ImportError) as exc:
            last_exc = exc
            if not _is_transient_torchvision_import_error(exc) or attempt == attempts - 1:
                raise
            _clear_torchvision_modules()
            time.sleep(1.0 + attempt)
    raise last_exc
    

class FedViT(nn.Module):
    def __init__(self):
        super(FedViT, self).__init__()
        self.vit = ViT(
            in_channels=1,
            img_size=2048,
            patch_size=32,
            num_layers=6, # 12
            num_heads=4, # 8
            hidden_size=768,
            mlp_dim=3072,
            dropout_rate=0,
            proj_type='conv',
            classification=True,
            num_classes=2,
            pos_embed_type='learnable',
            spatial_dims=2,
            post_activation=None # remove tanh activation
        )

    def forward(self, x):
        x = self.vit(x)
        return x
    

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)   # Output: 16 x 128 x 128
        self.pool = nn.MaxPool2d(2, 2)                            # Downsample by 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2) # Output: 32 x 64 x 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2) # Output: 64 x 32 x 32

        # After 3 poolings: 64 x 32 x 32
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 16 x 128 x 128
        x = self.pool(F.relu(self.conv2(x)))  # -> 32 x 64 x 64
        x = self.pool(F.relu(self.conv3(x)))  # -> 64 x 32 x 32
        x = torch.flatten(x, 1)               # Flatten all but batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        self.pretrained = pretrained
        convnext_tiny = _get_torchvision_builder("convnext_tiny")
        if self.pretrained:
            self.model = convnext_tiny(num_classes=1000, dropout=dropout, weights='IMAGENET1K_V1')
            # Adapt first conv for single channel
            orig_conv = self.model.features[0][0]
            with torch.no_grad():
                w = orig_conv.weight.data
                if w.size(1) == 3:
                    w_mean = w.mean(dim=1, keepdim=True)
                    new_conv = nn.Conv2d(
                        in_channels=1,
                        out_channels=orig_conv.out_channels,
                        kernel_size=orig_conv.kernel_size,
                        stride=orig_conv.stride,
                        padding=orig_conv.padding,
                        dilation=orig_conv.dilation,
                        groups=orig_conv.groups,
                        bias=(orig_conv.bias is not None),
                        padding_mode=getattr(orig_conv, "padding_mode", "zeros")
                    )
                    new_conv.weight.data.copy_(w_mean)
                    if orig_conv.bias is not None:
                        new_conv.bias.data.copy_(orig_conv.bias.data)
                    self.model.features[0][0] = new_conv

            # Replace classifier with BatchNorm1d version
            previous_head_in_features = self.model.classifier[2].in_features
            self.model.classifier = nn.Sequential(
                nn.Flatten(),  # Important for BatchNorm1d
                nn.BatchNorm1d(previous_head_in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            self.head = nn.Linear(previous_head_in_features, num_classes)
        else:
            self.model = convnext_tiny(num_classes=num_classes, dropout=dropout)
            self.model.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)

    def forward(self, x):
        return self.head(self.model(x)) if self.pretrained else self.model(x)
    

class MyConvNeXtSmall(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.4):
        super().__init__()
        self.pretrained = pretrained
        convnext_small = _get_torchvision_builder("convnext_small")
        if self.pretrained:
            self.model = convnext_small(num_classes=1000, dropout=dropout, weights='IMAGENET1K_V1')
            # Adapt first conv for single channel
            orig_conv = self.model.features[0][0]
            with torch.no_grad():
                w = orig_conv.weight.data
                if w.size(1) == 3:
                    w_mean = w.mean(dim=1, keepdim=True)
                    new_conv = nn.Conv2d(
                        in_channels=1,
                        out_channels=orig_conv.out_channels,
                        kernel_size=orig_conv.kernel_size,
                        stride=orig_conv.stride,
                        padding=orig_conv.padding,
                        dilation=orig_conv.dilation,
                        groups=orig_conv.groups,
                        bias=(orig_conv.bias is not None),
                        padding_mode=getattr(orig_conv, "padding_mode", "zeros")
                    )
                    new_conv.weight.data.copy_(w_mean)
                    if orig_conv.bias is not None:
                        new_conv.bias.data.copy_(orig_conv.bias.data)
                    self.model.features[0][0] = new_conv

            # Replace classifier with BatchNorm1d version
            previous_head_in_features = self.model.classifier[2].in_features
            self.model.classifier = nn.Sequential(
                nn.Flatten(),  # Important for BatchNorm1d
                nn.BatchNorm1d(previous_head_in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            self.head = nn.Linear(previous_head_in_features, num_classes)
        else:
            self.model = convnext_small(num_classes=num_classes, dropout=dropout)
            self.model.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)  # Modify first conv layer for 1 channel input

    def forward(self, x):
        return self.head(self.model(x)) if self.pretrained else self.model(x)
    

class MyConvNeXtBase(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        self.pretrained = pretrained
        convnext_base = _get_torchvision_builder("convnext_base")
        if self.pretrained:
            self.model = convnext_base(num_classes=1000, dropout=dropout, weights='IMAGENET1K_V1')
            # Adapt first conv for single channel
            orig_conv = self.model.features[0][0]
            with torch.no_grad():
                w = orig_conv.weight.data
                if w.size(1) == 3:
                    w_mean = w.mean(dim=1, keepdim=True)
                    new_conv = nn.Conv2d(
                        in_channels=1,
                        out_channels=orig_conv.out_channels,
                        kernel_size=orig_conv.kernel_size,
                        stride=orig_conv.stride,
                        padding=orig_conv.padding,
                        dilation=orig_conv.dilation,
                        groups=orig_conv.groups,
                        bias=(orig_conv.bias is not None),
                        padding_mode=getattr(orig_conv, "padding_mode", "zeros")
                    )
                    new_conv.weight.data.copy_(w_mean)
                    if orig_conv.bias is not None:
                        new_conv.bias.data.copy_(orig_conv.bias.data)
                    self.model.features[0][0] = new_conv

            # Replace classifier with BatchNorm1d version
            previous_head_in_features = self.model.classifier[2].in_features
            self.model.classifier = nn.Sequential(
                nn.Flatten(),  # Important for BatchNorm1d
                nn.BatchNorm1d(previous_head_in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            self.head = nn.Linear(previous_head_in_features, num_classes)
        else:
            self.model = convnext_base(num_classes=num_classes, dropout=dropout)
            self.model.features[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)  # Modify first conv layer for 1 channel input

    def forward(self, x):
        return self.head(self.model(x)) if self.pretrained else self.model(x)


def get_model(model_args):
    if model_args.get('name') == 'ViT':
        return FedViT() # ViT returns logits and hidden states (tuple of 2 elements)
    elif model_args.get('name') == 'ConvNeXtTiny':
        return MyConvNeXtTiny(num_classes=model_args.get('num_classes', 2), pretrained=model_args.get('pretrained', True), dropout=model_args.get('dropout', 0.3))
    elif model_args.get('name') == 'ConvNeXtSmall':
        return MyConvNeXtSmall(num_classes=model_args.get('num_classes', 2), pretrained=model_args.get('pretrained', True), dropout=model_args.get('dropout', 0.3))
    elif model_args.get('name') == 'ConvNeXtBase':
        return MyConvNeXtBase(num_classes=model_args.get('num_classes', 2), pretrained=model_args.get('pretrained', True), dropout=model_args.get('dropout', 0.3))
    else:
        return SimpleNetwork()
