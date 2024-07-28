"""AGXNet_Siamese model architecture."""
import timm
import torch
from torch import nn


class AGXNet_Siamese(nn.Module):
    def __init__(self, args, agxnet_state_dict, pretrained_state_dict):
        super(AGXNet_Siamese, self).__init__()
        self.args = args

        # initialize pretrained anatomy network and classifier
        net1_state_dict = {}
        fc1_state_dict = {}
        self.net1 = timm.create_model(args.arch, pretrained=True, features_only=True)
        self.fc1 = nn.Linear(1024, len(self.args.landmark_names_spec))
        for k, v in agxnet_state_dict.items():
            net = k.partition('.')[0]  # e.g., 'net1'
            par = k.partition('.')[-1]  # e.g., 'features_conv0.weight'
            # find net1 weights
            if (net == 'net1') and (par in self.net1.state_dict()) and (v.size() == self.net1.state_dict()[par].size()):
                net1_state_dict[par] = v
            # find fc1 weights
            if (net == 'fc1') and (par in self.fc1.state_dict()) and (v.size() == self.fc1.state_dict()[par].size()):
                fc1_state_dict[par] = v
        self.net1.load_state_dict(net1_state_dict)
        self.fc1.load_state_dict(fc1_state_dict)

        # freezing layers in anatomy network
        if self.args.freeze_net1 == 'T':
            for param in self.net1.parameters():
                param.requires_grad = False
            for param in self.fc1.parameters():
                param.requires_grad = False

        # initialize observation network
        # 'Random', 'ImageNet', 'GLoRIA', 'GLoRIA_MIMIC', 'ConVIRT', 'BioVIL' are used for baseline models
        if self.args.pretrained_type == 'Random':
            self.net2 = timm.create_model(args.arch, pretrained=False, features_only=True)
        if self.args.pretrained_type == 'ImageNet':
            self.net2 = timm.create_model(args.arch, pretrained=True, features_only=True)
        if self.args.pretrained_type == 'AGXNet':
            self.net2 = timm.create_model(args.arch, pretrained=True, features_only=True)
            net2_state_dict = {}
            for k, v in agxnet_state_dict.items():
                net = k.partition('.')[0]  # e.g., 'net1'
                par = k.partition('.')[-1]  # e.g., 'features_conv0.weight'
                # find net2 weights
                if (net == 'net2') and (par in self.net2.state_dict()) and (v.size() == self.net2.state_dict()[par].size()):
                    net2_state_dict[par] = v
            self.net2.load_state_dict(net2_state_dict)
        if self.args.pretrained_type in ['GLoRIA', 'GLoRIA_MIMIC']:
            self.net2 = timm.create_model(args.arch, pretrained=True, features_only=True)
            net2_state_dict = {}
            for k, v in pretrained_state_dict.items():
                pars = k.split('.')
                if (pars[1] == 'img_encoder') and (pars[2] == 'model'):
                    layer_name = pars[3] + '_' + pars[4] + '.' + '.'.join(pars[5:])
                    if (layer_name in self.net2.state_dict()) and (v.size() == self.net2.state_dict()[layer_name].size()):
                        net2_state_dict[layer_name] = v
            self.net2.load_state_dict(net2_state_dict)
        if self.args.pretrained_type == 'ConVIRT':
            self.net2 = timm.create_model(args.arch, pretrained=True, features_only=True)
            net2_state_dict = {}
            for k, v in pretrained_state_dict.items():
                net = k.partition('.')[0]  # e.g., '0'
                par = k.partition('.')[-1]  # e.g., 'conv0.weight'
                # find net2 weights
                layer_name = 'features_' + par
                if (layer_name in self.net2.state_dict()) and (v.size() == self.net2.state_dict()[layer_name].size()):
                    net2_state_dict[layer_name] = v
            self.net2.load_state_dict(net2_state_dict)
        if self.args.pretrained_type == 'BioVIL':
            self.net2 = timm.create_model('resnet50', pretrained=True, features_only=True)
            net2_state_dict = {}
            for k, v in pretrained_state_dict.items():
                layer_name = '.'.join(k.split('.')[2:])  # remove 'encoder.encoder.'
                if (layer_name in self.net2.state_dict()) and (v.size() == self.net2.state_dict()[layer_name].size()):
                    net2_state_dict[layer_name] = v
            self.net2.load_state_dict(net2_state_dict)

        # define pooling layer
        self.pool = nn.AdaptiveAvgPool2d(1)

        # define fully connected dense layers
        if args.pretrained_type == 'BioVIL':
            self.dense = nn.Sequential(
                # nn.Dropout(0.5),
                nn.Linear(4096, 1024),
                nn.ReLU()
            )
        else:
            self.dense = nn.Sequential(
                # nn.Dropout(0.5),
                nn.Linear(2048, 1024),
                nn.ReLU()
            )

        # define classification layer
        self.cls = nn.Linear(1024, 4)  # output classes: improved, unchanged, worsened, new

        # placeholder for the gradients
        self.gradients_x = None
        self.gradients_y = None

        # AGXNet_Siamese is the key of pretrained model for evaluation
        if self.args.pretrained_type == 'AGXNet_Siamese':
            self.net1 = timm.create_model(args.arch, pretrained=True, features_only=True)
            self.fc1 = nn.Linear(1024, len(self.args.landmark_names_spec))
            if self.args.configs.pretrained_type == 'BioVIL':
                self.net2 = timm.create_model('resnet50', pretrained=True, features_only=True)
                self.dense = nn.Sequential(nn.Linear(4096, 1024), nn.ReLU())
            else:
                self.net2 = timm.create_model('densenet121', pretrained=True, features_only=True)
                self.dense = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.cls = nn. Linear(1024, 4)


    def forward(self, img_x, img_y, landmark_idx):

        # obtain observation features
        f2_x = self.net2(img_x)[-1]  # b * c * w * h
        h_x = f2_x.register_hook(self.activations_hook_x)
        f2_y = self.net2(img_y)[-1]
        h_y = f2_y.register_hook(self.activations_hook_y)

        # obtain anatomical attentions
        f1_x = self.net1(img_x)[-1]
        f1_y = self.net1(img_y)[-1]
        cam1_x = torch.einsum('bchw, ac -> bahw', f1_x, self.fc1.weight)  # b * a * h * w
        cam1_y = torch.einsum('bchw, ac -> bahw', f1_y, self.fc1.weight)  # b * a * h * w

        # normalize images independently
        if self.args.cam_norm_type == 'indep':
            cam1_norm_x = self.normalize_cam1(cam1_x)  # b * a * h * w
            cam1_norm_y = self.normalize_cam1(cam1_y)
        # normalize two images together
        elif self.args.cam_norm_type == 'dep':
            cam1_norm_x, cam1_norm_y = self.normalize_cams(cam1_x, cam1_y)
        else:
            raise ValueError('invalid cam normalization type %r' % self.args.cam_norm_type)

        # prepare gather indices along the landmark dimension (dimension 1)
        indices = landmark_idx.unsqueeze(-1)
        indices = indices.unsqueeze(-1)
        indices = indices.repeat(1, 16, 16)
        indices = indices.unsqueeze(1)
        cam1_sel_x = torch.gather(cam1_norm_x, 1, indices)
        cam1_sel_y = torch.gather(cam1_norm_y, 1, indices)
        cam1_sel_x = cam1_sel_x.squeeze(dim=1)  # b * h * w
        cam1_sel_y = cam1_sel_y.squeeze(dim=1)  # b * h * w

        # apply attention
        if self.args.anatomy_attention_type == 'Mask':
            f2_x = torch.einsum('bchw, bhw -> bchw', f2_x, cam1_sel_x)  # b * c * w * h
            f2_y = torch.einsum('bchw, bhw -> bchw', f2_y, cam1_sel_y)
        if self.args.anatomy_attention_type == 'Residual':
            f2_x = torch.einsum('bchw, bhw -> bchw', f2_x, 1 + self.args.epsilon * cam1_sel_x)  # b * c * w * h
            f2_y = torch.einsum('bchw, bhw -> bchw', f2_y, 1 + self.args.epsilon * cam1_sel_y)

        # pooling
        f2_p_x = self.pool(f2_x).squeeze(dim=-1).squeeze(dim=-1)  # b * c
        f2_p_y = self.pool(f2_y).squeeze(dim=-1).squeeze(dim=-1)

        # concatenating
        f2_p_cat = torch.cat((f2_p_x, f2_p_y), 1) # b * (c + c)

        # dense layers that output logits
        x = self.dense(f2_p_cat)
        logit = self.cls(x)

        return logit

    def activations_hook_x(self, grad_x):
        self.gradients_x = grad_x

    def activations_hook_y(self, grad_y):
        self.gradients_y = grad_y

    def get_activations_gradient_x(self):
        return self.gradients_x

    def get_activations_gradient_y(self):
        return self.gradients_y

    def get_activations_x(self, img_x):
        return self.net2(img_x)[-1]

    def get_activations_y(self, img_y):
        return self.net2(img_y)[-1]

    def normalize_cam1(self, cam1):
        [b, a, h, w] = cam1.shape
        cam1_norm = cam1.view(b, a, -1)
        cam1_norm -= cam1_norm.min(2, keepdim=True)[0]
        cam1_norm /= (cam1_norm.max(2, keepdim=True)[0] + 1e-12) # pervent from dividing 0
        cam1_norm = cam1_norm.view(b, a, h, w)
        return cam1_norm

    def normalize_cams(self, cam_x, cam_y):
        [b, a, h, w] = cam_x.shape
        cam_x_norm = cam_x.view(b, a, -1)
        cam_y_norm = cam_y.view(b, a, -1)
        cam_xy_norm = torch.cat((cam_x_norm, cam_y_norm), dim=2)
        cam_x_norm -= cam_xy_norm.min(2, keepdim=True)[0]
        cam_y_norm -= cam_xy_norm.min(2, keepdim=True)[0]
        cam_x_norm /= (cam_xy_norm.max(2, keepdim=True)[0] + 1e-12)  # pervent from dividing 0
        cam_y_norm /= (cam_xy_norm.max(2, keepdim=True)[0] + 1e-12)  # pervent from dividing 0
        cam_x_norm = cam_x_norm.view(b, a, h, w)
        cam_y_norm = cam_y_norm.view(b, a, h, w)
        return cam_x_norm, cam_y_norm