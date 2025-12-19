#Torch and PyTorch specific
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout_rate=0.2):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(in_channels // 2, num_classes)

    def forward(self, x):          # [Batch, Chanels, Time]
        x = x.permute(0, 2, 1)     # [Batch,  Time, Chanels,]
        x = self.linear(x)    
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)      # [B, num_classes, T]
        return x


class AttentionBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = "linear",
                 image_shape: tuple = (1,1),
                ):
        super().__init__()

        self.activation = activation
        
        self.attention = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,  #So we're doing per-class attention, because number of classes per sample is unknown
            kernel_size=3, #was 1 originally, changed to 3 with good results
            stride=1,
            padding=1,  #was 0 originally, changed to 1 to match above
            bias=True)
        
        with torch.no_grad():
            self.attention.weight.fill_(1.0 / (self.attention.kernel_size[0] * in_features))
            self.attention.bias.zero_()
        
        self.classify = ClassifierHead(in_channels=in_features, num_classes=out_features)
        self.image_shape=image_shape
        self.num_chunks = int(self.image_shape[0]*self.image_shape[1])

    def init_layer(self, layer): #could access the outer class init_layer method instead
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)    

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

    def forward(self, x):
        # x: (batch_size, n_features, n_chunks * n_segments_per_chunk)

        # We can reshape to convolve only along the frequency dimension to operate on the time chunks independently. 
        # We don't need to do this for the logits, but keeping the same form in case we want to change the activation, 
        # or kernel size in a way that they are not independent of each other.

        batch_size = x.shape[0]  # Split along the third dimension
        split_length = x.shape[2] // self.num_chunks
        
        x = torch.split(x, split_length, dim=2)
        x = torch.cat(x, dim=0)  #  (128, 1280, 4)
        
        attn = self.attention(x) #.squeeze(1)
        norm_att = torch.softmax(torch.tanh(attn), dim=-1)/self.num_chunks  #so that they have a mean value of 1/16 each
        split_attn = torch.split(norm_att, batch_size, dim=0) #Put the weights back to their original shape
        norm_att = torch.cat(split_attn, dim=2)#.unsqueeze(-1) 

        seg_logits = self.classify(x)

        seg_logits = F.dropout(seg_logits, p=0.3, training=self.training)
        classify = self.nonlinear_transform(seg_logits)  #note - this is OK, because we're just doing a sigmoid, would be

        split_logits = torch.split(seg_logits, batch_size, dim=0)
        seg_logits = torch.cat(split_logits, dim=2)

        split_classify = torch.split(classify, batch_size, dim=0)
        classify = torch.cat(split_classify, dim=2)
        
        weighted_preds = norm_att * classify
        weighted_seg_logits = norm_att * seg_logits
        preds = weighted_preds.sum(dim=-1)   
        logit = weighted_seg_logits.sum(dim=-1)  #equivalent to mean because the weights are /16
        seg_logits = seg_logits.transpose(1, 2)

        return logit, seg_logits, preds


class BirdSoundModel(nn.Module):
    
    def init_layer(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.0)

    def init_weight(self):
        self.init_bn(self.bn0)
        self.init_layer(self.fc1)

    def __init__(self,
                 backbone_name, 
                 num_classes,
                 image_shape,
                 in_channels=3,
                 backbone_checkpoint_path=None,
                 device = None,
                ):
        super().__init__()
        available = torch.cuda.is_available()
        self.device = device
        if self.device is None:
            self.device = torch.device(("cuda" if available else "cpu"))
        self.num_classes = num_classes
        self.bn0 = nn.BatchNorm2d(3) #(audio.IMAGE_WIDTH) #if cfg.RESHAPE_IMAGE else nn.BatchNorm2d(audio.N_MELS)
        self.backbone_name = backbone_name
        self.base_model = timm.create_model(
                                    self.backbone_name, 
                                    pretrained=True, 
                                    in_chans=in_channels
                                    )

        layers = list(self.base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if backbone_checkpoint_path is not None:
            checkpoint = torch.load(backbone_checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Filter to only load encoder/backbone weights
            encoder_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if "encoder." in k}

            missing, unexpected = self.encoder.load_state_dict(encoder_dict, strict=False)
            print(f"Backbone weights loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")

        try:
            classifier = self.base_model.get_classifier()
            if isinstance(classifier, nn.Identity):
                # Classifier was stripped; use num_features to build your own
                in_features = self.base_model.num_features
            else:
                in_features = classifier.in_features
        except AttributeError:
            # Fallback for edge cases
            if hasattr(self.base_model, "fc") and hasattr(self.base_model.fc, "in_features"):
                in_features = self.base_model.fc.in_features
            elif hasattr(self.base_model, "head") and hasattr(self.base_model.head, "fc"):
                in_features = self.base_model.head.fc.in_features
            elif hasattr(self.base_model, "classifier") and hasattr(self.base_model.classifier, "in_features"):
                in_features = self.base_model.classifier.in_features
            else:
                raise ValueError(
                    f"Unable to determine in_features for model: {self.backbone_name}\n"
                    f"Model structure:\n{self.base_model}"
                )
        
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.image_shape = image_shape
        self.att_block = AttentionBlock(in_features, 
                                        self.num_classes, 
                                        activation="sigmoid",
                                        image_shape=self.image_shape
                                        )
        self.init_weight()


    def forward(self, x):

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W) spectrograms, got {x.shape}")

        # (batch_size, channels, mel_bins,  time_steps)
        x = self.bn0(x)
        x = self.encoder(x)  #This is the image passing through the base model  8x8 out with a 256x256 image
        
        #This is the guts of the SED part.  So first we need to unfold back into original shape so agregation has it's spatial meaning
        
        if self.image_shape == (2,2):  #Stack the (1,2) and (2,2) scenarios in the frequency direction
            half = x.shape[2]//2
            x0 = x[:,:,:half,:half]
            x1 = x[:,:,:half,half:]
            x2 = x[:,:,half:,:half]
            x3 = x[:,:,half:,half:]
            x = torch.cat((x0,x1,x2,x3), dim=2) #stack vertically along the frequency direction, so now it's 16 high, 4 wide for a 256x256 input image
        elif self.image_shape == (1,4):  #Stack the (1,2) and (2,2) scenarios in the frequency direction
            quarter = x.shape[2]//4
            x0 = x[:,:,:,:quarter]
            x1 = x[:,:,:,quarter:2*quarter]
            x2 = x[:,:,:,2*quarter:3*quarter]
            x3 = x[:,:,:,3*quarter:]
            x = torch.cat((x0,x1,x2,x3), dim=2) #stack vertically along the frequency direction, so now it's 16 high, 4 wide for a 256x256 input image
        elif self.image_shape == (1,2):
            half = x.shape[3]//2
            x0 = x[:,:,:,:half]
            x1 = x[:,:,:,half:]
            x = torch.cat((x0,x1), dim=2) #For a 128x128 (2,1) image, we'd now have 8 high in frequency, 2 wide in time
        elif self.image_shape == (2, 0.5):
            half = x.shape[2]//2
            x0 = x[:,:,:half,:]
            x1 = x[:,:,half:,:]
            
            x = torch.cat((x0,x1), dim=3)  #For a 256x256 (2, 0.5) this should be now 4 high in frequency, 16 wide in time

        #For the (2,1) and (1,1) cases we donn't need to do anything here, there is only one chunk represented along the horizontal axis.
        dimension = 2 if self.image_shape == (2, 0.5) else 3
        x = torch.mean(x, dim=dimension) # Aggregate in short axis, but only over each chunk
        #print(f'shape after mean {x.shape}')
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.3, training=self.training)

        (logit, segment_logits, preds) = self.att_block(x) #  for (2,1) in: [64, 1280, 16], out: [64, 182], [64, 16, 182],  [64, 182]
        
        output_dict = {
            'clip_preds': preds,  #predictions for AP and CMAP calculation
            'segmentwise_logit': segment_logits,   #[64, 16, 182]  # doesn't use the attn or activation function
            'logit': logit,  #torch.Size([64, 182]) used for the loss calculation, includes attention   
        }

        return output_dict