use_case = {'debug': True, 'experiment': 20}  #Run only one soundscape for visualisation purposes only

#General Python
import os
from pathlib import Path
from tqdm import tqdm
import time
import functools
import json

#Math & Plotting
import numpy as np
import pandas as pd
import random

#Machine Learning 
import albumentations as A

#Torch and PyTorch specific
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import  DataLoader, Dataset

#Audio
import librosa
import torchaudio



class DefaultConfig:
    def __init__(self, options=None):
        self.EXPERIMENT = int(options['experiment'])
        self.TEST_BATCH_SIZE = 32
        self.MODEL = "tf_efficientnet_b0.ns_jft_in1k"  #tf_efficientnetv2_s_in21k"  #tf_efficientnetv2_s 
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.PRECISION = '16-mixed' if torch.cuda.is_available() else 32
        self.DEBUG = True if options and options['debug'] == True else False
        self.NUM_WORKERS = 0 #os.cpu_count()//2 -1    #Set to zero for windows or it will be super slow spawning new threads with every file
        self.CHECKPOINT_DIR = Path(r'C:\Users\ollyp\Desktop\Bird_Classifier\Data') / f'Exp_{self.EXPERIMENT}' / 'Results' / 'Best_Weights'
        self.CHECKPOINT_PATH = str(self.CHECKPOINT_DIR / f'exp_{self.EXPERIMENT}_best_weights.ckpt')
        self.OUT_DIR = r'C:\Users\ollyp\Desktop\Bird_Classifier\Data\Exp_21\Results'
        self.COMMON_NAMES_DICT = r'C:\Users\ollyp\Desktop\Bird_Classifier\Data\Bird_Names\ebird_common_name_birdclef_24_dict.json' 
        self.SOUNDSCAPES = r'C:\Users\ollyp\Desktop\Bird_Classifier\Data\birdclef-2024\train_audio'
        self.PREDICTIONS_PATH = Path(self.OUT_DIR) / 'segment_predictions.csv'
        self.RESHAPE_IMAGE = True

#total width (about 128) = 1+ (num_samples - frame_length)//hop_length 
# For 256x256   1 + (10*32000 - 2048)//1200 = 266   (10 seconds, or 20 seconds folded)
# For 128x128   1 + (3*3200 - 2048)//700 = 135      (3 seconds, no-fold)

class DefaultAudio:
    def __init__(self, options=None):
        self.SR = 32000
        self.DURATION = 20  # Duration the loaded sound file will be cropped to.
        self.CHUNK_LENGTH = 20 # Maximum duration of the sound files
        self.N_MELS = 128 #Height of the spectrogram before subtracting upper and lower frequency limits
        self.N_FFT = 2048 #N_fft/2 + 1 bins will get made prior to downsampling to the value of N_MELS
        self.IMAGE_WIDTH = 256 #The spectrogram will get cropped/padded to this square regardless of audio
        self.FMIN = 20
        self.FMAX = 14000 
        self.HOP_LENGTH = 1200
        self.PCEN = False

#Ideally config settings should get saved by the training notebook, and updated here.
cfg = DefaultConfig(options=use_case)
audio = DefaultAudio(options=use_case)

class Stop: #bold red
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'
    
class Go: #bold green
    S = '\033[1m' + '\033[32m'
    E = '\033[0m'
    
class Blue: #for general info
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def load_sf(wav_path, sr=audio.SR):
    y, _ = librosa.load(wav_path, sr=sr)
    return y


def compute_pcen(y):
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.zeros_like(y)
        y[np.isinf(y)] = np.max(y)
    
    melspec = librosa.feature.melspectrogram(y=y, 
                                             sr=audio.SR, 
                                             n_mels=audio.N_MELS, 
                                             n_fft= audio.N_FFT, 
                                             fmin=audio.FMIN, 
                                             fmax=audio.FMAX
                                            )
    pcen = librosa.pcen(melspec, 
                        sr=audio.SR, 
                        gain=0.98, 
                        bias=2, 
                        power=0.5, 
                        time_constant=0.4, 
                        eps=0.000001
                       )
    return pcen.astype(np.float32)


def compute_melspec(y):
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.zeros_like(y)
        y[np.isinf(y)] = np.max(y)
    #print(f'Clip length: {len(y)//32000}(s)')
    
    melspec = librosa.feature.melspectrogram(y=y, 
                                             sr=audio.SR, 
                                             n_mels=audio.N_MELS, 
                                             n_fft=audio.N_FFT, 
                                             hop_length = audio.HOP_LENGTH, 
                                             fmin=audio.FMIN, 
                                             fmax=audio.FMAX
                                            ) 
    #print(f'Spectrogram shape: {melspec.shape}')
    return librosa.power_to_db(melspec)


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        X = (X - _min) / (_max - _min) #scales to a range of [0,1]
        X = X.astype(np.float32)
    else:
        X = np.zeros_like(X, dtype=np.float32)
    X = np.stack([X, X, X], axis=-1) #puts the chanels last, like a normal image, for the ablu_trasformations
    return X


def crop_or_pad(y, length):
    if len(y) <= length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    return y


def load_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y
    

class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError


class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)

class AbluTransforms():
    mean = (0.485, 0.456, 0.406) # RGB
    std = (0.229, 0.224, 0.225) # RGB

    train = A.Compose([
                    A.Normalize(mean, std, max_pixel_value=1.0,always_apply=True),
                    A.CoarseDropout(max_holes=4, p=0.4),
                    A.PadIfNeeded(min_height=audio.IMAGE_WIDTH, min_width=audio.IMAGE_WIDTH),
                    A.RandomCrop(width=audio.IMAGE_WIDTH, height=audio.IMAGE_WIDTH),      
                    ])
    
    valid = A.Compose([
                    A.Normalize(mean, std, max_pixel_value=1.0,always_apply=True),
                    A.PadIfNeeded(min_height=audio.IMAGE_WIDTH, min_width=audio.IMAGE_WIDTH, position='bottom_left'),  #adjust the 
                    A.Crop(x_min=0, y_min=0, x_max=audio.IMAGE_WIDTH, y_max=audio.IMAGE_WIDTH, always_apply=True,),
                    ])

def fold_image(arr): 
    '''chop the image in half along the temporal dimension and stack to a square image
    Goal is to allow more pixels and segments in the temporal domain than frequency'''
    cols = arr.shape[1]//2 
    remainder = arr.shape[1] % 2
    half1 = arr[:, :cols + remainder]
    half2 = arr[:, cols:]
    arr =  np.vstack((half1, half2))
    return arr


def open_audio_clip(path):
    y, _ = torchaudio.load(path)
    y = y.squeeze().numpy()
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.mean(y)
        y[np.isinf(y)] = np.mean(y)
    return y


class WaveformDataset(Dataset):
    def __init__(self, 
                 df,
                 sr=audio.SR, 
                 duration=audio.DURATION, 
                 chunk=audio.CHUNK_LENGTH, 
                 train=False, 
                 soundscape_array=None,
                 reshape_image = cfg.RESHAPE_IMAGE): 
        self.df = df
        self.sr = sr 
        self.train = train
        self.d_len = duration * self.sr
        self.c_len = chunk * self.sr
        self.image_transform = AbluTransforms.valid
        self.wave_transforms = Normalize(p=1)
        self.ss_array = soundscape_array
        self.reshape_image = reshape_image
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        '''For training open the path from the row filepath, 
        or for a soundscape we just need to crop out the 
        relevent chunk of the array opened in the init.'''
        row = self.df.iloc[idx]

        # Assuming here that we're dealing with soundscapes
        start = self.sr * row.start
        stop = self.sr * row.stop
        #print(start, stop, len(self.ss_array)//32000)
        y = self.ss_array[start: stop]  #arrays of 0 after the first iteration 
        print(f'The soundscape is {len(y)//32000} seconds long before padding')
        y = crop_or_pad(y, 20*32000)
        y=self.wave_transforms(y, sr=self.sr) #Normalize
        #y = crop_or_pad(y, 20*32000)        
        print(f'The soundscape is {len(y)//32000} seconds long after padding')
        if audio.PCEN: 
            image = compute_pcen(y)
        else:
            image = compute_melspec(y)

        if self.reshape_image:
            image = fold_image(image)
        image = mono_to_color(image)
        image = self.image_transform(image=image)['image']
        image = image.transpose(2,0,1).astype(np.float32) # swapping the image channels to the first axis
        print(image)

        return image, idx


class BirdSoundModel(pl.LightningModule):

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
    
    def interpolate(self, x: torch.Tensor, ratio: int):
        """Interpolate data in time domain. This is used to compensate the
        resolution reduction in downsampling of a CNN.
        Args:
          x: (batch_size, time_steps, classes_num)
          ratio: int, ratio to interpolate
        Returns:
          upsampled: (batch_size, time_steps * ratio, classes_num)
        """
        (batch_size, time_steps, classes_num) = x.shape
        upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
        upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
        return upsampled

    def pad_framewise_output(self, framewise_output: torch.Tensor, frames_num: int):
        """Pad framewise_output to the same length as input frames. The pad value
        is the same as the value of the last frame.
        Args:
          framewise_output: (batch_size, frames_num, classes_num)
          frames_num: int, number of frames to pad
        Outputs:
          output: (batch_size, frames_num, classes_num)
        """
        output = F.interpolate(
            framewise_output.unsqueeze(1),
            size=(frames_num, framewise_output.size(2)),
            align_corners=True,
            mode="bilinear").squeeze(1)

        return output

    class AttentionBlock(nn.Module):
        def __init__(self,
                     in_features: int,
                     out_features: int,
                     activation="linear"):
            super().__init__()

            self.activation = activation
            self.attention = nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True)
            self.classify = nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True)
            self.init_weights()
        
        def init_layer(self, layer): #could access the outer class init_layer method instead
            nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, "bias"):
                if layer.bias is not None:
                    layer.bias.data.fill_(0.)
               
        def init_weights(self):
            self.init_layer(self.attention)
            self.init_layer(self.classify)

        def nonlinear_transform(self, x):
            if self.activation == 'linear':
                return x
            elif self.activation == 'sigmoid':
                return torch.sigmoid(x)

        def forward(self, x):
            # x: (n_samples, n_bins, n_time)
            norm_att = torch.softmax(torch.tanh(self.attention(x)), dim=-1)
            classify = self.nonlinear_transform(self.classify(x))
            x = torch.sum(norm_att * classify, dim=2)
            return x, norm_att, classify

    def __init__(self, 
                 classes, 
                 base_model_name=cfg.MODEL,
                 in_channels=3,
                 reshape_image=cfg.RESHAPE_IMAGE
                ):
        super().__init__()

        self.classes = classes
        self.num_classes = len(classes)
        self.bn0 = nn.BatchNorm2d(audio.IMAGE_WIDTH)
        self.base_model = timm.create_model(
                                    base_model_name, 
                                    pretrained=False, 
                                    in_chans=in_channels,
                                    )
        layers = list(self.base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.reshape_image = reshape_image

        if hasattr(self.base_model, "fc"):
            in_features = self.base_model.fc.in_features
        else:
            in_features = self.base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = self.AttentionBlock(in_features, 
                                            self.num_classes, 
                                            activation="sigmoid")
        self.init_weight()
        self.val_outputs = []
        self.train_outputs = []
        self.metrics_list = []
        self.val_epoch = 0
        
    def forward(self, input_data):
        x = input_data # (batch_size, 3, time_steps, mel_bins)  #This needs to match the the output of dataloader & getitem 
        frames_num = x.shape[2] * 2 if self.reshape_image else x.shape[2]

        x = x.transpose(1, 3) #(batch_size, mel_bins, time_steps, channels)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.transpose(2, 3)
        x = self.encoder(x)  #This is the image passing through the base model  8x8 out with a 256x256 image
        #spatial_size = x.shape[2:]  # Get spatial dimensions (height, width)
        #print("Spatial layer size before global pooling:", spatial_size)  
        
        if self.reshape_image:
            tensor_shape = x.size()
            x = x.view(tensor_shape[0], tensor_shape[1], 2*tensor_shape[2], tensor_shape[3]//2)
        
        x = torch.mean(x, dim=3) # Aggregate in frequency axis
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2  #Summing max and average as per Qiuqiang Kong's paper
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        
        #This is the guts of the SED part.  
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.classify(x), dim=2)
        segmentwise_logit = self.att_block.classify(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)
        interpolate_ratio = frames_num // segmentwise_output.size(1)
        framewise_output = self.interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = self.pad_framewise_output(framewise_output, frames_num)
        framewise_logit = self.interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = self.pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            'framewise_output': framewise_output,  #torch.Size([64, 256, 79])
            'clipwise_output': clipwise_output,    #torch.Size([64, 79])
            'logit': logit,                        #torch.Size([64, 79])   
            'framewise_logit': framewise_logit,    #torch.Size([64, 256, 79])
            'segmentwise_logit': segmentwise_logit,   #[64, 8, 79]
            'segmentwise_output': segmentwise_output, #[64, 8, 79]
        }

        return output_dict

    def validation_dataloader(self):
        return self._validation_dataloader

def pool_segments(selected_rows):
    return np.max(selected_rows, axis=0)
    

def get_model_for_inference(class_list, ckpt_path=cfg.CHECKPOINT_PATH):
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(ckpt_path, map_location=map_location)
    model = BirdSoundModel(class_list)
    model.load_state_dict(state_dict)  #["model_state_dict"]
    model.eval()
    model = model.to(torch.device('cuda'))
    return model

def get_clip_df(audio_path, #pathlip Path object
                chunk_length=20, 
                clip_length=240):
    #clip_id = "_".join(audio_path.stem.split("_")[:-1])
    clip_id = audio_path.stem
    row_ids, paths, starts, stops  = [], [], [], []
    row_ids = []
    for start in range(0, clip_length, chunk_length):
        row_id = clip_id + f"_{start+chunk_length}"
        row_ids.append(row_id)
        paths.append(audio_path)
        starts.append(start)
        stops.append(start + chunk_length)
            
    #Generate a df for the whole clip   
    df = pd.DataFrame({
            "row_id": row_ids,
            "filepath": Path(audio_path).name,
            "start": starts,
            "stop": stops
        })     
    return df

def prediction_for_clip(audio_path,
                        model,
                        sub_process=False):
    

    soundscape = open_audio_clip(audio_path)
    clip_length = len(soundscape)//32000
    print(f'\nThe clip is of length {clip_length} seconds')
    df = get_clip_df(audio_path, chunk_length=audio.CHUNK_LENGTH, clip_length=clip_length)
    #print(df.head())

    dataset = WaveformDataset(df=df, soundscape_array=soundscape, train=False)
    loader = DataLoader(dataset, batch_size=cfg.TEST_BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not sub_process:
        progress = tqdm(range(len(loader)), desc="Inferring a single soundscape")
    
    clip_preds_dict = {}
    segment_preds_dict = {}
    
    for images, batch_idxs, in loader:
        print(images.shape)
        row_idxs = batch_idxs.tolist()
        images=images.to(device)

        with torch.no_grad():
            prediction = model(images)  
            batch_preds = prediction["clipwise_output"].detach().cpu().numpy()  #This will be a tensor of batch_size x num_classes
            batch_segment_preds = prediction["segmentwise_output"].detach().cpu().numpy()  #batch_size x num_segments x num_classes
        
        for idx, clip_preds, segment_preds in zip(row_idxs, batch_preds, batch_segment_preds):
            target_row = df.iloc[idx] 
            row_id = target_row['row_id']
            
            segment_preds_dict[row_id] = segment_preds
            #print(f'the shape of this segment is: {segment_preds.shape}')
             #row_id identifies the chunk, (eg for 20 sec chunks it will be filename _20, _40 etc), 
             #within the chunk each row of the array is a segment eg. 21.125, 22.25...
            clip_preds_dict[row_id] = clip_preds
    
        if not sub_process:
            progress.update(1)

    return clip_preds_dict, segment_preds_dict 


def inference(audios, bird_list, model):
    print('hello, now you need to figure out what to do with the inference function')
    chunk_length = 20
    num_segments = 16 
    segment_length = chunk_length / num_segments
    
    file_ids = []
    seg_times = []
    segment_preds_list = []
    parent_list = []
    
    
    if cfg.DEBUG:
        audios = [Path(r'C:\Users\ollyp\Desktop\Bird_Classifier\Data\birdclef-2024\train_audio\asbfly\XC134896.ogg')]
    print(f'There are {len(audios)} audio clips to process')
    for audio_path in tqdm(audios, desc="Overall File List"):
        
        parent = Path(audio_path).parent.name
        _, segment_dict = prediction_for_clip(audio_path,
                                              model=model,
                                              sub_process=True)
        
        if cfg.DEBUG:
            print(segment_dict)
        
        #At this point we should have a probability array for all birds for each 20 seconds of the clip, 
        #and a segment array of predictions for each segment from that 20 seconds.  We need to turn the 
        #segment array into row outputs with an ID for the segment time.

        for chunk in segment_dict.keys():
            last_underscore_index = chunk.rfind('_') 
            file_id = chunk[:last_underscore_index]
            chunk_end = int(chunk[last_underscore_index + 1:])
            chunk_start = chunk_end - chunk_length
            segment_chunk_times = [chunk_start + idx * segment_length for idx in range(num_segments)]  #0, 1.125, 2.25 etc
            segment_preds_list.append(segment_dict[chunk])
            file_ids.extend([file_id] * num_segments)
            parent_list.extend([parent] * num_segments)
            seg_times.extend(segment_chunk_times)
        segment_preds_array = np.concatenate(segment_preds_list, axis=0)

    segment_df = pd.DataFrame(segment_preds_array, columns=bird_list)
    segment_df.insert(0, 'Parent_Dir', parent_list)
    segment_df.insert(1, 'File_ID', file_ids)
    segment_df.insert(2, 'Segment_Time', seg_times)
    return segment_df

##########################################################################################################
######################################  Main Code  #######################################################
if __name__ == '__main__':  
    print(Go.S + 'Imports Complete' + Go.E)
    debug_text = Stop.S + 'Notebook will run reduced soundscapes' + Stop.E 
    text = Go.S + 'Notebook will infer on all soundscapes' + Go.E if not use_case['debug'] else debug_text
    print(text)
    print(Blue.S + 'Running inference on:' + Blue.E, cfg.DEVICE)

    bird_mapper=load_json(cfg.COMMON_NAMES_DICT)
    bird_list=list(bird_mapper.keys())
    model = get_model_for_inference(bird_list)
    soundscapes = [path for path in Path(cfg.SOUNDSCAPES).rglob('*.ogg')]
    soundscapes = soundscapes[:10] if cfg.DEBUG else soundscapes
    #print(soundscapes)

    predictions = inference(soundscapes,
                        bird_list,
                        model=model,
                        )
    
    print(predictions.head())
    print(predictions.tail(10))  #These should all be about 0 for the 27 second sample XC134896
    predictions.to_csv(cfg.PREDICTIONS_PATH, index=False)