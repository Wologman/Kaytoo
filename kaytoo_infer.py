import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning, message='A new version')
from typing import Dict, List, Optional, Tuple, Union, Any
import albumentations as A
import numpy as np
import torchaudio
import torch
import librosa
from scipy.signal import resample
from torchaudio.functional import compute_deltas
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch.nn.functional as F
import timm
import torch.nn as nn
from joblib import Parallel, delayed
import multiprocessing
multiprocessing.freeze_support()
from tqdm import tqdm
import pandas as pd
from torch.utils.data import  DataLoader  # ISSUE #15: Extra space in import
from bird_naming_utils import BirdNamer
from pathlib import Path
import re  # ISSUE #11: Unused import - remove if not needed
import argparse
import yaml
import random
import ast
import matplotlib.pyplot as plt  # ISSUE #11: Unused import - only used in commented code
from test_cuda import test_cuda

############################################# Parameters  ######################################
##################################################################################################

class DefaultConfig:
    def __init__(self,
                 bird_namer: BirdNamer,
                 options: Optional[dict]=None):
        self.classes = bird_namer.bird_list
        if options:
            if options['cpu_only']:
                self.device = torch.device('cpu')
            else:
                device, gpu = test_cuda()
                if gpu:
                    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
                    print("Using GPU:", device_name)
                    self.device = torch.device(device)
            if options['num_cores']:
                self.CORES = options['num_cores']
            else:
                self.CORES = 1
            self.naming = options['naming_scheme']
        else:
            device, gpu = test_cuda()
            self.naming = 'eBird'
            self.device = torch.device(device)
            self.cores = os.cpu_count()//2 or 1


class AudioParameters:
    def __init__(self):
        self.SR = 32000
        self.FMIN = 20
        self.FMAX = 14000 


class FilePaths:
    AUDIO_TYPES = {'.ogg','.wav', '.flac', '.mp3'}
    def __init__(self, options=None):
        self.root_folder = Path(options['project_root'])  # ISSUE #1: No validation that paths exist or are accessible
        self.models_folder = self.root_folder / 'models'
        self.data_folder = self.root_folder / 'data'
        self.predictions = Path(options['results_folder'])
        self.bird_list_path = self.root_folder / 'resources/bird_map.csv'
        self.soundscapes_folder = Path(options['folder_to_process'])  # ISSUE #1: No validation that paths exist or are accessible
        self.soundscapes = [path for path in self.soundscapes_folder.rglob('*') if path.suffix in self.AUDIO_TYPES]
        self.predictions.mkdir(parents=True, exist_ok=True)


class ModelParameters:
   def __init__(self, options=None):  # ISSUE #15: Inconsistent indentation (3 spaces vs 4)
        '''
        _parameters_list = [
                            {'basename':'tf_efficientnet_b0.ns_jft_in1k', 
                                            'ckpt_path': model_path,
                                            'image_shape': (1,2), #The layout of 5-sec spectrograms stacked into the final image (height x width)
                                            'image_time': 10,
                                            'n_mels': 256,
                                            'n_fft': 2048,
                                            'double_audio': False,
                                            'buffer_audio': 0,
                                            'use_deltas' : True,
                                            'hop_length': 1243,
                                            '5_sec_width': 128,
                                            'aggregation': 'mean',
                                            }, 
                            ] 
        '''
        if options['experiment'] is not None:
            _deploy_fldr = f"{options['project_root']}/data/experiments/exp_{options['experiment']}/exp_{options['experiment']}_deploy"
            _deploy_folders = [Path(_deploy_fldr)]
        else:
            _models_dir = Path(f"{options['project_root']}/models/")
            _deploy_folders = [subdir for subdir in _models_dir.iterdir() if subdir.is_dir() and subdir.name.endswith('_deploy')]
            
        self.parameters = []
        for model_fldr in _deploy_folders:
            ckpt_files = list(model_fldr.glob("*.ckpt"))
            cfg_files = list(model_fldr.glob("*.yaml"))
            if ckpt_files:
                latest_ckpt = max(ckpt_files, key=lambda f: f.stat().st_mtime)  #There should only be one  
                if cfg_files:
                    latest_cfg = max(cfg_files, key=lambda f: f.stat().st_mtime)  #There should only be one
                    with open(latest_cfg, "r") as f:
                        model_config = yaml.load(f, Loader=yaml.FullLoader)  # Using FullLoader to support Python tuples in YAML (safer than default loader)
                        model_config['ckpt_path'] = latest_ckpt
                        self.parameters.append(model_config)
                else:
                    print(f'Warning: No configuration file was found in {str(model_fldr)}')  # ISSUE #9: Use logging instead of print
            else:
                    print(f'Warning: No checkpoint file was found in {str(model_fldr)}')  # ISSUE #9: Use logging instead of print    


class Colour:
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def open_audio_clip(path: Path, default_sr: int = 32000, min_duration: int = 10) -> np.ndarray:
    """Open an audio clip and ensure it is a valid, finite 1D numpy array.
    On error or invalid input, replaces with random noise.
    """
    try:
        y, sr = torchaudio.load(path)
        # Convert stereo to mono
        if y.ndim == 2 and y.shape[0] == 2:
            y = torch.mean(y, dim=0)
        y = y.squeeze().numpy()
        if y.size == 0:
            print(f"[WARN] {path} -> empty array returned from torchaudio.load(); replacing with noise")
            y = np.random.randn(default_sr * min_duration)
            sr = default_sr
    except Exception as e:
        print(f"[WARN] Could not open {path}: {e}")
        y = np.random.randn(default_sr * min_duration)
        sr = default_sr

    # Replace NaN or Inf with zeros
    if not np.isfinite(y).all():
        print(f"[WARN] Invalid (NaN/Inf) values found in {path}, replaced with zeros.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize safely (avoid divide by zero)
    max_val = np.max(np.abs(y))
    if max_val == 0 or not np.isfinite(max_val):
        print(f"[WARN] Zero or invalid max value in {path}, skipping normalization.")
    else:
        y = y / max_val

    # Resample if needed
    if sr != default_sr:
        num_samples = int(len(y) * default_sr / sr)
        y = resample(y, num_samples)
        sr = default_sr

    # Pad or trim to at least `min_duration` seconds
    required_samples = int(min_duration * default_sr)
    if len(y) < required_samples:
        pad_len = required_samples - len(y)
        y = np.concatenate([y, np.random.randn(pad_len)])
        print(f"[INFO] Padded {path} to {len(y)/default_sr:.1f} s")

    assert np.isfinite(y).all(), f"[FATAL] Non-finite values persist in {path}!"
    return y



def compute_melspec(y: np.ndarray, sr: int, hop_length: int, n_mels: int, n_fft: int, audio_params: Optional[AudioParameters]) -> np.ndarray:
    if audio_params:
        fmin = audio_params.FMIN
        fmax = audio_params.FMAX
    else:
        fmin = 40
        fmax = 14000

    melspec = librosa.feature.melspectrogram(y=y, 
                                             sr=sr, 
                                             n_mels=n_mels, 
                                             n_fft=n_fft, 
                                             hop_length = hop_length, 
                                             fmin=fmin, 
                                             fmax=fmax
                                            ) 
    return librosa.power_to_db(melspec)


class PrepareImage():
    mean = .5
    std = .22
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.prep = A.Compose([
            A.PadIfNeeded(min_height=self.height, min_width=self.width),
            A.CenterCrop(width=self.width, height=self.height),
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0, always_apply=True),
        ])


def get_images(audio_path: Path,
               model_params: Dict[str, Any],
               audio_params: AudioParameters,
               clip_length: Optional[int] = None,
               sr: int = 32000) -> Tuple[Dict[int, np.ndarray], int]:
    
    double = model_params['double_audio']
    buffer = model_params['buffer_audio'] * sr
    trim = abs(buffer)
    hop_length = model_params['hop_length']
    n_mels = model_params['n_mels']
    n_fft = model_params['n_fft']
    chunk_width = model_params['5_sec_width']
    num_chunks = model_params['image_shape'][0] * model_params['image_shape'][1]
    
    if model_params['double_audio']:  #note that buffer_audio can be < 0
        chunk_length = int(((model_params['image_time']-2*model_params['buffer_audio'])/2)//num_chunks)
    else:
        chunk_length = int((model_params['image_time']-2*model_params['buffer_audio'])//(num_chunks))
    
    prep_image = PrepareImage(height=n_mels, width=chunk_width)

    idxs  = []
    image_dict = {}
    _y = open_audio_clip(audio_path)
    if clip_length is None:
        clip_length = len(_y) // sr


    print("clip_length (sec):", clip_length)
    print("chunk_length (sec):", chunk_length)
    print("loop iterations:", int(clip_length) // chunk_length)

    for index in range(0, int(clip_length) // chunk_length):
        idxs.append(index)
        start = index * chunk_length
        stop = start + chunk_length
        start_idx = sr * start
        stop_idx =  sr * stop
        
        if double and buffer>=0:
            if index == 0:
                y = np.concatenate((_y[:buffer], _y[:stop_idx], _y[:stop_idx + buffer]))
            elif index == 11:  # ISSUE #6: Magic number 11 - should use total_chunks - 1 instead
                y = np.concatenate((_y[start_idx-buffer:],  _y[start_idx:], _y[-buffer:]))
            else:
                y = np.concatenate((_y[start_idx-buffer:stop_idx], _y[start_idx:stop_idx+buffer]))
        elif double and buffer < 0:
                _core = _y[start_idx:stop_idx]
                y = np.concatenate((_core[trim:], _core[:-trim]))
        else:
            if stop_idx > len(_y):
                y = _y[start_idx:]
                remaining_length = stop_idx - len(_y)  # Calculate how much we need to fill
                noise = np.random.randn(remaining_length) * np.std(_y)
                y = np.concatenate((y, noise))
            else: 
                y = _y[start_idx: stop_idx]
        
        max_vol = np.abs(y).max()
        y = y * 1 / max_vol    # ISSUE #15: Comment on same line as code - y, sr, hop_length, n_mels, n_fft, audio_params
        max_vol = np.max(np.abs(y))
        
        if max_vol > 0 and np.isfinite(max_vol):
            y = y / max_vol
        
        image = compute_melspec(y, sr, hop_length, n_mels, n_fft, audio_params)
        image = prep_image.prep(image=image)['image']
        image_dict[index] = image
        
    num_specs = len(image_dict)
    extra_specs = clip_length % num_chunks  #Handle the case where there are more spectrograms needed to make up the combined image
    if extra_specs:
        noise = np.random.randn(chunk_length * sr)
        image = compute_melspec(noise, sr, hop_length, n_mels, n_fft, audio_params)
        image = prep_image.prep(image=image)['image']
        for extra_idx in range(num_specs+1, num_specs+extra_specs+1):
            image_dict[extra_idx] = image

    return image_dict, extra_specs  #a dict of images, with keys from 0 to 47 for the case of a 240 second clip.


def mono_to_color(X: np.ndarray, eps: float = 1e-6, use_deltas: bool = False) -> np.ndarray:
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        X = (X - _min) / (_max - _min) #scales to a range of [0,1]
        X = X.astype(np.float32)
    else:
        X = np.zeros_like(X, dtype=np.float32)
    
    if use_deltas:
        T = torch.tensor(X, dtype=torch.float32)
        delta = compute_deltas(T)
        delta_2 = compute_deltas(delta)
        delta, delta_2 = delta.numpy(), delta_2.numpy()
        X=np.stack([X, delta, delta_2], axis=-1)
    else:
        X = np.stack([X, X, X], axis=-1) #puts the chanels last, like a normal image

    return X


def crop_or_pad(y: np.ndarray, length: int, train: str = 'train') -> np.ndarray:
    y = np.concatenate([y, y, y])
    if len(y) <= length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    else:
        if train != 'train':
            start = 0
        else:
            start = np.random.randint(len(y) - length)
        y = y[start: start + length]
    return y


class AudioTransform:
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr: int) -> np.ndarray:
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
    def __init__(self, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params: Any) -> np.ndarray:
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)



class AbluTransforms():
    mean = (0.485, 0.456, 0.406) # RGB
    std = (0.229, 0.224, 0.225) # RGB
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.valid = A.Compose([
                        A.PadIfNeeded(min_height=self.height, min_width=self.width),
                        A.CenterCrop(width=self.width, height=self.height),
                        A.Normalize(self.mean, self.std, max_pixel_value=1.0,always_apply=True),
                        ])
        self.train = A.Compose([
                        A.CoarseDropout(max_holes=4, p=0.4, max_height=32, max_width=32),
                        A.PadIfNeeded(min_height=self.width, min_width=self.width),
                        A.CenterCrop(width=self.width, height=self.height), 
                        A.Normalize(self.mean, self.std, max_pixel_value=1.0, always_apply=True),  
                        ])


def spec_augment(spec: np.ndarray, 
                 num_mask: int = 3, 
                 freq_masking_max_percentage: float = 0.1,
                 time_masking_max_percentage: float = 0.1, 
                 p: float = 0.5) -> np.ndarray:
    if random.uniform(0, 1) > p:
        return spec

    # frequency masking
    num_freq_masks = random.randint(1, num_mask)
    for i in range(num_freq_masks):
        freq_percentage = random.uniform(0, freq_masking_max_percentage)
        freq_mask_size = int(freq_percentage * spec.shape[0])
        freq_mask_pos = random.randint(0, spec.shape[0] - freq_mask_size)
        spec[freq_mask_pos:freq_mask_pos+freq_mask_size, :] = 0

    # time masking
    num_time_masks = random.randint(1, num_mask)
    for i in range(num_time_masks):
        time_percentage = random.uniform(0, time_masking_max_percentage)
        time_mask_size = int(time_percentage * spec.shape[1])
        time_mask_pos = random.randint(0, spec.shape[1] - time_mask_size)
        spec[:, time_mask_pos:time_mask_pos+time_mask_size] = 0

    return spec


class ImageDataset(Dataset):
    def __init__(self, image_dict, image_shape, use_deltas, train=False): #, model_args
        self.image_dict = image_dict
        self.image_shape = image_shape
        self.image_pixels = self.image_dict[0].shape
        self.height = self.image_shape[0] * self.image_pixels[0]  #Shape of the combined image from one __get_item__
        self.width = self.image_shape[1] * self.image_pixels[1]
        self.train = train
        if not self.train:
            self.image_transform = AbluTransforms(height=self.height, width=self.width).valid
        else:
            self.image_transform = AbluTransforms(height=self.height, width=self.width).train
        self.use_deltas = use_deltas
        self.chunks_per_image = int(self.image_shape[0] * self.image_shape[1])
               
    def __len__(self) -> int:
        whole = len(self.image_dict) // self.chunks_per_image
        remainder =  1 if len(self.image_dict) % self.chunks_per_image != 0 else 0
        return  whole + remainder

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base = idx * self.chunks_per_image
        chunk_idxs = [base + n for n in range(self.chunks_per_image)]
        images = [self.image_dict[img_id] for img_id in chunk_idxs]
        
        if self.image_shape == (2,2):  #The (2,2) case 
            image = np.vstack((np.hstack((images[0], images[1])),   
                               np.hstack((images[2], images[3]))))
        elif self.image_shape == (1,2):  
            image = np.hstack((images[0], images[1]))
        elif self.image_shape == (2,1): 
            image = np.vstack((images[0], images[1]))
        elif self.image_shape == (1,4):
            image = np.hstack((images[0], images[1], images[2], images[3]))
        elif self.image_shape == (2,0.5):
            half = images[0].shape[1]//2
            left = images[0][:, :half]     # First half of columns (128, 256)
            right = images[0][:, half:]    # Second half of columns (128, 256)
            image = np.vstack([left, right])  # Shape becomes (256, 256)
        else:
            image = images[0]
        if self.train:
            image = spec_augment(image, 
                                    p=0.25, 
                                    num_mask=3,
                                    freq_masking_max_percentage=0.1,
                                    time_masking_max_percentage=0.1)

        image = mono_to_color(image, use_deltas=self.use_deltas)
        image = self.image_transform(image=image)['image']  #To be exactly equivalent, this should be applied globally like this, not per chunk.
        image = image.transpose(2,0,1).astype(np.float32) # swapping the image channels to the first axis
        return image, idx


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout_rate=0.2):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(in_channels // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # [Batch, Chanels, Time]
        x = x.permute(0, 2, 1)     # [Batch,  Time, Chanels,]
        x = self.linear(x)    
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)      # [B, num_classes, T]
        return x


class BirdSoundModel(pl.LightningModule):

    def init_layer(self, layer: nn.Module) -> None:
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_bn(self, bn: nn.BatchNorm2d) -> None:
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.0)
        
    def init_weight(self) -> None:
        self.init_bn(self.bn0)
        self.init_layer(self.fc1)



    class AttentionBlock(nn.Module):
        def __init__(self,
                     in_features: int,
                     out_features: int,
                     activation="linear",
                     image_shape = (1,1),
                     aggregation = 'mean',  #not used at the moment
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
            
            self.classify = ClassifierHead(in_channels=in_features, num_classes=out_features)
            self.image_shape=image_shape
            self.num_chunks = int(self.image_shape[0]*self.image_shape[1])

        def init_layer(self, layer: nn.Module) -> None: #could access the outer class init_layer method instead
            nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, "bias"):
                if layer.bias is not None:
                    layer.bias.data.fill_(0.)    
            
        def init_weights(self) -> None:
            self.init_layer(self.attention)
            #self.init_layer(self.classify)

        def nonlinear_transform(self, x: torch.Tensor) -> torch.Tensor:
            if self.activation == 'linear':
                return x
            elif self.activation == 'sigmoid':
                return torch.sigmoid(x)


        def forward(self, x: torch.Tensor) -> torch.Tensor:

            # x: (batch_size, n_features, n_chunks * n_segments_per_chunk)

            # We can reshape to convolve only along the frequency dimension to operate on the time chunks independently. 
            # We don't need to do this for the logits, but keeping the same form in case we want to change the activation, 
            # or kernel size in a way that they are not independent of each other.

            batch_size = x.shape[0]  # Split along the third dimension
            split_length = x.shape[2] // self.num_chunks
            
            x = torch.split(x, split_length, dim=2)
            x = torch.cat(x, dim=0)  #  (128, 1280, 4)
            
            attn = self.attention(x) #.squeeze(1)
            norm_att = torch.softmax(torch.tanh(attn), dim=-1)/self.num_chunks #so that they have a mean value of 1/16 each
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
            preds = weighted_preds.sum(dim=-1)   

            return preds


    def __init__(self, 
                 classes, 
                 kwargs,
                 in_channels=3,
                ):
        super().__init__()
        
        self.image_time = kwargs['image_time'] # The total length of time represented by one complete image
        self.spec_height = kwargs['n_mels'] # The height of each spectrogram, before any stacking into an image
        self.chunk_width = kwargs['5_sec_width'] # The width of each spectrogram, before any stacking into an image
        self.image_shape = kwargs['image_shape'] # The spectrogram arrangement into an image (2,2) or (1,1) or (1,2), height x width
        self.base_model_name = kwargs['basename']
        self.aggregation = kwargs['aggregation']
        self.classes = classes
        self.num_classes = len(classes)
        self.image_width = self.image_shape[1] * self.chunk_width
        self.bn0 = nn.BatchNorm2d(3) #(self.image_width)   #self.image_width  #why is this still 256???
        self.base_model = timm.create_model(
                                    self.base_model_name, 
                                    pretrained=False, 
                                    in_chans=in_channels,
                                    )
        layers = list(self.base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(self.base_model, "fc"):
            in_features = self.base_model.fc.in_features
        elif self.base_model_name == 'eca_nfnet_l0':
            in_features = self.base_model.head.fc.in_features
        elif self.base_model_name == 'convnext_tiny.in12k_ft_in1k':
            in_features = self.base_model.head.fc.in_features
        else:
            in_features = self.base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)

        self.att_block = self.AttentionBlock(in_features,
                                             out_features=self.num_classes,
                                             activation="sigmoid",
                                             image_shape = self.image_shape,
                                             aggregation = self.aggregation,  #currently unused?
                                             )
        self.init_weight()
        self.val_outputs = []
        self.train_outputs = []
        self.metrics_list = []
        self.val_epoch = 0

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = input_data  #(batch_size, 3, frequency, time)  #This needs to match the the output of dataloader & getitem 
        #x = x.transpose(1, 3)  #(batch_size, mel_bins, time_steps, channels)
        x = self.bn0(x)
        #x = x.transpose(1, 3)
        x = self.encoder(x)  #This is the image passing through the base model  8x8 out with a 256x256 original image
        
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
            #print(f'concatenated shape {x.shape}')
            #x = x.transpose()  #should be batch, features, freq

        #For the (2,1) and (1,1) cases we don't need to do anything here, there is only one chunk represented along the horizontal axis.
        
        #This is the guts of the SED part.
        dimension = 2 if self.image_shape == (2, 0.5) else 3
        x = torch.mean(x, dim=dimension) # Aggregate in short axis, but only over each chunk, so now we've just got a 3d tensor (batch_size, n_features, freq-time chunks)       
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        
        chunk_preds = self.att_block(x)
        #print(chunk_preds.shape)
        return chunk_preds #(48,182) regardless of how the images were shaped


class Models:
    def __init__(self, config, model_parameters, audio_parameters):
        self.args_list = model_parameters.parameters
        self.audio = audio_parameters
        self.ebirds = config.classes
        self.device = config.device

    def get_model(self, idx: int) -> 'BirdSoundModel':
        model_args = self.args_list[idx]
        path = model_args['ckpt_path']
        map_location = 'cpu' if self.device == torch.device('cpu') else 'cuda'
        ckpt = torch.load(path, map_location=map_location)  # ISSUE #1: Should specify weights_only=True for security (PyTorch 2.0+)
        model = BirdSoundModel(self.ebirds, model_args)
        model.load_state_dict(ckpt)
        model.eval()
        model.parameters = self.args_list[idx]
        model.audio = self.audio
        model.to(self.device)
        return model  
    

def prediction_for_clip(audio_path: Path,
                        model: 'BirdSoundModel',
                        sub_process: bool = False) -> np.ndarray:
    model_args=model.parameters
    audio_params=model.audio
    device = model.device
    
    image_dict, num_extras = get_images(audio_path, model_args, audio_params) #returns a dict with integers as keys
    
    num_images = len(image_dict)

    dataset = ImageDataset(image_dict, model_args['image_shape'], model_args['use_deltas'])
    shape = model_args['image_shape']
    num_chunks = shape[0] * shape[1]
    #so batch_size here refers to an entire clip.  This will create strife with low-memory, and long clips.
    batch_size = int(num_images // num_chunks)  #should be a whole number, because we made sure of this in the get_images
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)  # ISSUE #5: Hard-coded batch_size=4 - should use calculated batch_size or make configurable
    
    if not sub_process:
        progress = tqdm(range(len(loader)), desc="Inferring a single soundscape")
    
    for images, _ in loader:

    #    example = images[0][0]
    #    plt.figure(figsize=(10, 4))
    #    plt.imshow(example.cpu().numpy(), origin='lower', aspect='auto', cmap='magma')
    #    plt.title("Mel Spectrogram")
    #    plt.xlabel("Time")
    #    plt.ylabel("Frequency")
    #    plt.colorbar(label='Amplitude (dB)')
    #    plt.tight_layout()
    #    plt.show()


        images=images.to(device)

        with torch.no_grad():
            predictions = model(images)  
            batch_segment_preds = predictions.detach().cpu().numpy()  #batch_size x num_chunks x num_classes
      
        if not sub_process:
            progress.update(1)

    del loader, dataset, image_dict
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()

    #Now let's lop off the last num_extras predictions, as these were made on place-holding random noise
    if num_extras:
        batch_segment_preds = batch_segment_preds[:-num_extras, :]  # ISSUE #5: Inefficient array slicing - creates new array

    return batch_segment_preds


def process_clip(audio_path: Path, model: 'BirdSoundModel') -> Tuple[List[np.ndarray], List[str], List[str]]:
    clip_preds = []
    #final_row_ids = []
    batch_preds = prediction_for_clip(audio_path,
                                      model,
                                      sub_process=True)
    #returns a numpy array of length (180 x num_classes)

    num_preds = batch_preds.shape[0]
    
    for idx in range(num_preds):  # ISSUE #5: Unnecessary list building - can use batch_preds.tolist() directly
        row = batch_preds[idx]
        clip_preds.append(row)

    row_ids = [f"{audio_path.with_suffix('')}_{end}".replace("\\", "/") 
           for end in range(5, (num_preds+1)*5, 5)]
    
    file_paths = [str(audio_path).replace('\\', '/')] * num_preds

    return clip_preds, row_ids, file_paths


def inference(test_audios: List[Path], models: 'Models', model_idx: int, cores: int = 1) -> pd.DataFrame:
    bird_list = models.ebirds
    model = models.get_model(model_idx)


    print
    results = Parallel(n_jobs=cores, backend='threading')(
        delayed(process_clip)(audio_path, model=model) for audio_path in tqdm(test_audios, desc="Overall File List")
        )
    
    del model
    if models.device == torch.device('cuda'):
        torch.cuda.empty_cache()

    clip_preds = [result[0] for result in results]      #This should be a list of 1 x num_classes arrays, with a length = the number of predictions
    final_row_ids = [result[1] for result in results]   #This should be a list of row id's?
    file_paths = [result[2] for result in results]

    clip_preds = [item for sublist in clip_preds for item in sublist]
    final_row_ids = [item for sublist in final_row_ids for item in sublist]
    file_paths = [item for sublist in file_paths for item in sublist]
    prediction_df = pd.DataFrame(clip_preds, columns=bird_list)
    prediction_df.insert(0, 'row_id', final_row_ids)
    prediction_df.insert(1, 'File_Path', file_paths)
    
    return prediction_df


class DeriveResults():
    def __init__(self, predictions, save_folder):
        self.predictions = predictions
        self.save_folder = save_folder
        self.chosen_birds = {}
        print('from inside DeriveResults')
        print(self.predictions.head())

    def summarise(self, df: pd.DataFrame) -> pd.DataFrame:
        def _summarise(group):
            group.drop(columns=['row_id'], inplace=True)  # ISSUE #6: In-place operation on view - may fail if group is a view
            group = group.loc[:, (group == 1).any()]
            duration_seconds = len(group) * 5
            remaining_columns = group.columns
            column_sums = group.sum(axis=0).astype(int)
            total_birds = group.sum().sum().astype(int)
            sorted_with_values = [f'{col} ({column_sums[col]})' for col in column_sums.sort_values(ascending=False).index]
            
            all_birds = ', '.join(sorted_with_values)
            num_birds = len(remaining_columns)

            summary = pd.Series({
                'Unique_Species_Ordered': all_birds,
                'Unique_Species': num_birds,
                'Total_Detections': int(total_birds),
                'Length_(s)': duration_seconds
            })
            return summary

        grouped = df.groupby('File_Path')
        summary_df = grouped.apply(_summarise, include_groups=False).reset_index(drop=False)
        return summary_df
    
    def summarise_one_bird(self, df: pd.DataFrame, bird_name: str) -> pd.DataFrame:
        def _process_detections(group):
            end_time = group['time'].iloc[-1]   #get from row_id, rather than assume a row time length
            group = group.reset_index(drop=True)
            non_zero_times = group.loc[group[bird_name] != 0,  'time'].tolist()
            total = len(non_zero_times) 
            fraction = round(total / len(group), 3)
            end_minutes = end_time / 60
            per_minute = round(total/end_minutes, 4)
            if non_zero_times:
                first = non_zero_times[0]
                last = non_zero_times[-1]
                from_end = end_time - last
                all_non_zero = ', '.join(map(str, non_zero_times))
            else:
                first, last, from_end, all_non_zero = None, None, None, None

            series = pd.Series({'First': first, 
                                'Last': last, 
                                'FromEnd': from_end, 
                                'Total': total, 
                                'FileLength' : end_time,
                                'FractionTrue': fraction,
                                'CallsPerMinute': per_minute,
                                'AllDetections': all_non_zero}
                                )

            return series

        df = df[['File_Path', 'row_id', bird_name]].copy()
        df['time'] = df['row_id'].str.extract(r'_(\d+)$').astype(int)
        grouped = df.groupby('File_Path')
        df = grouped.apply(_process_detections, include_groups=False).reset_index(drop=False)       
        return df

    def first_detected(self, df: pd.DataFrame) -> pd.DataFrame:
        def _first_detected(group):
            group.drop(columns=['row_id'], inplace=True)  # ISSUE #6: In-place operation on view - may fail if group is a view 
            group = group.reset_index(drop=True)
            first_non_zero = group.apply(lambda col: col.ne(0).idxmax()*5 if col.ne(0).any() else np.nan)
            return first_non_zero
        grouped = df.groupby('File_Path')
        first_bird = grouped.apply(_first_detected, include_groups=False).reset_index(drop=False)
        columns_to_convert = list(first_bird.columns[1:])
        first_bird.loc[:,columns_to_convert] = first_bird[columns_to_convert]#.astype(pd.Int64Dtype())
        return first_bird

    def last_detected(self, df: pd.DataFrame) -> pd.DataFrame:
        def _last_detected(group):
            group.drop(columns=['row_id'], inplace=True)  # ISSUE #6: In-place operation on view - may fail if group is a view 
            group = group.reset_index(drop=True)
            length = len(group) * 5
            last_non_zero = group.apply(lambda col: length - (col[::-1].ne(0).idxmax()) * 5 if col.ne(0).any() else np.nan)
            return last_non_zero
        grouped = df.groupby('File_Path')
        last_bird = grouped.apply(_last_detected, include_groups=False).reset_index(drop=False)
        columns_to_convert = list(last_bird.columns[1:])
        last_bird.loc[:,columns_to_convert] = last_bird[columns_to_convert]#.astype(pd.Int64Dtype())
        return last_bird

    def detections_per_minute(self, df: pd.DataFrame) -> pd.DataFrame:
        def _detections_per_minute(group):
            group.drop(columns=['row_id'], inplace=True)  # ISSUE #6: In-place operation on view - may fail if group is a view 
            group = group.reset_index(drop=True)
            minutes = len(group) / 12
            column_sums = group.sum(axis=0).astype(int)
            bird_rate = column_sums / minutes
            bird_rates = pd.Series(bird_rate, index=group.columns).round(3).astype('float32')
            return bird_rates
        grouped = df.groupby('File_Path')
        bird_rate = grouped.apply(_detections_per_minute, include_groups=False).reset_index(drop=False) 
        return bird_rate
    
    def summarise_chosen_birds(self, birds_to_summarise: List[str]) -> None:
        for bird in birds_to_summarise:
            self.chosen_birds[bird]=self.summarise_one_bird(self.predictions, bird)

    def derive_results(self) -> None:
        self.summary = self.summarise(self.predictions)
        self.first_bird = self.first_detected(self.predictions)
        self.last_bird = self.last_detected(self.predictions)
        self.bird_rate = self.detections_per_minute(self.predictions)

    def save_results(self, save_folder: Optional[Union[Path, str]] = None) -> None:
        save_folder = save_folder if save_folder is not None else self.save_folder
        save_folder = Path(save_folder) if not isinstance(save_folder, Path) else save_folder
        self.summary.to_csv(save_folder / 'detection_summary.csv', index=False)
        self.first_bird.to_csv(save_folder / 'first_bird.csv', index=False)
        self.last_bird.to_csv(save_folder / 'last_bird.csv', index=False)
        self.bird_rate.to_csv(save_folder / 'detections_per_minute.csv', index=False)
        if self.chosen_birds:
            for bird, summary in self.chosen_birds.items():
                bird_no_spaces = bird.replace(" ", "_")
                summary.to_csv(save_folder / f"{bird_no_spaces}_summary.csv", index=False)

    def print_results(self) -> None:
        print(Colour.S + '\nThe summary dataframe' + Colour.E)
        print(self.summary.iloc[:3,:8])
        print(Colour.S + '\nThe first detection time for each species (s)' + Colour.E)
        print(self.first_bird.iloc[:3,:8])
        print(Colour.S + '\nThe last detection time from the end for each species  (s)' + Colour.E)
        print(self.last_bird.iloc[:3,:8])
        print(Colour.S + '\nThe detection rate, bird per minute for each species' + Colour.E)
        print(self.bird_rate.iloc[:3,:8])

def merge_classes(df: pd.DataFrame, ebirds: List[str], short_names: List[str]) -> pd.DataFrame:
    name_df = pd.DataFrame({'ebirds': ebirds, 'short_name': short_names})
    mergers = name_df.groupby('short_name').agg({'ebirds': list})
    merger_dict = mergers['ebirds'].to_dict()
    print(Colour.S + '\nThe following classes are being merged:' + Colour.E)
    print(mergers[mergers['ebirds'].apply(len) > 1])
    print(Colour.S + '\nThe final number of classes will be reduced to:' + Colour.E, f'{len(set(short_names))}')

    merged_preds = {}
    for name in mergers.index.to_list():
        merging_cols = merger_dict[name]
        merging_pred_vals = df[merging_cols].values
        merged_preds[name] = np.any(merging_pred_vals, axis=1).astype(int)
        merged_df = pd.DataFrame(merged_preds)
        merged_df['row_id'] = df['row_id']
        merged_df['File_Path'] = df['File_Path']
        cols = ['row_id'] + [col for col in merged_df.columns if col != 'row_id']
        merged_df = merged_df[cols]
    return merged_df


############################################# Main Function  #####################################
##################################################################################################

def infer_soundscapes(use_case: Dict[str, Any]) -> None:
    audio = AudioParameters()
    paths = FilePaths(options=use_case)
    bird_map_df = pd.read_csv(paths.bird_list_path)
    birdnames = BirdNamer(bird_map_df)
    cfg = DefaultConfig(bird_namer=birdnames, 
                        options=use_case)
    parameters = ModelParameters(options=use_case)
    models = Models(config=cfg, 
                    model_parameters=parameters, 
                    audio_parameters=audio)
    summary_birds = use_case['birds_to_summarise']
    naming_scheme = use_case['naming_scheme']
    threshold = use_case['threshold']
    
    def _return_same(x: str) -> str:
        return x  
    _naming_methods = {'Short':_return_same, 
                       'Long':birdnames.common_name, 
                       'Scientific':birdnames.scientific_name, 
                       'eBird':_return_same}
    naming_method = _naming_methods[naming_scheme]

    print('The inference folder is:', paths.soundscapes_folder)
    print(f'There are {len(models.args_list)} model(s) to be ensembled')
    print(f'The model(s) will predict the following {len(models.ebirds)} birds (referring to their https://ebird.org code): \n')
    for i in range(0, len(models.ebirds), 10):
        print(", ".join(models.ebirds[i:i + 10]))

    #Run inference on the available models
    prediction_dfs = []
    for idx in range(len(models.args_list)):
        df = inference(paths.soundscapes, models, idx, cores=cfg.CORES)
        prediction_dfs.append(df)

    prediction_columns = prediction_dfs[0].columns[2:]
    values_list = [df[prediction_columns].values for df in prediction_dfs]
    average_vals = np.zeros_like(values_list[0])

    for array in values_list:
        average_vals = average_vals + array 

    average_vals = average_vals / len(values_list)

    #Deal with the various naming schemes, & thresholding
    final_names = [naming_method(col_name) for col_name in prediction_columns]

    predictions = pd.DataFrame(data=average_vals, columns=final_names)
    predictions.insert(0, 'row_id', prediction_dfs[0]['row_id']) 
    predictions.insert(0, 'File_Path', prediction_dfs[0]['File_Path']) 

    print(Colour.S + 'Raw prediction scores for the first 8 birds' + Colour.E)
    pd.set_option("display.max_colwidth", None)
    print(predictions.iloc[:5, :8])

    predictions.to_csv(paths.predictions / 'prediction_probabilities.csv', index=False)
    predictions.iloc[:,2:] = (predictions.iloc[:,2:] > threshold).astype(int)

    print(Colour.S + 'Thresholded scores for the first 8 birds' + Colour.E)
    print(predictions.iloc[:5, :8])

    if naming_scheme == 'Short':
        short_names  = birdnames.extra_names(birdnames.bird_list)  #we need to do this way for the one-many relationship
        predictions = merge_classes(predictions, birdnames.bird_list, short_names)

        print(Colour.S + 'Merged scores for the first 8 birds' + Colour.E)
        print(predictions.iloc[:5, :8])

    #Derive various alternative data represenations
    post_processor = DeriveResults(predictions,  
                                   save_folder=paths.predictions,
                                   )
    post_processor.derive_results()
    if summary_birds and (summary_birds[0] != 'All'):
        print(Colour.S + '\nBirds that will have individual summary files are: ' + Colour.E, summary_birds)
        post_processor.summarise_chosen_birds(summary_birds)
    post_processor.save_results()
    post_processor.print_results()

############################################  Run Main  ##########################################
##################################################################################################

if __name__ == '__main__':
    #Default options for running during development
    options = {
                'project_root': '/home/olly/Desktop/Kaytoo', #'/media/olly/T7/Kaytoo', #'/media/olly/T7/Kaytoo', # 'G:/Kaytoo',  #'/media/olly/T7/Kaytoo'  
                'experiment': None, #None to use what ever is in the Models folder, otherwise an integer for the experiment number
                'threshold': 0.3,
                #'folder_to_process': '/home/olly/Desktop/Kaytoo/data/more_corrupt_files',# 'D:/Kaytoo/Data/Corrupt_Files', #'D:/Kaytoo/Data/Soundscapes/debugging',
                'folder_to_process':'/home/olly/Desktop/Kaytoo/data/corrupt_files',
                'results_folder': '/home/olly/Desktop/Kaytoo/predictions',
                'naming_scheme' : 'Short', #'Scientific', #'Short, Long, Scientific, eBird'
                'cpu_only': False,
                'num_cores': 1,  #Can crank this up if using CPU only.
                'birds_to_summarise':['Spotted Kiwi'], #['Chlidonias albostriatus']  #['Australian Magpie'],  Black-fronted Tern blfter1	Chlidonias albostriatus	Tern
                }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=None, help="Filepath to the root directory (parent of the 'Python' folder)")
    parser.add_argument("--experiment", type=str, default=None, help="An integer for the experiment number, or None to models from /Models")
    parser.add_argument("--threshold", type=str, default=None, help="The prediction threshold.  Currently 0.2 looks about right")
    parser.add_argument("--folder_to_process", type=str, default=None, help="path to the processing folder, relative to the project root")
    parser.add_argument("--results_folder", type=str, default=None, help="Folder to put the predictions")
    parser.add_argument("--naming_scheme", type=str, default=None, help="A string that must be one of: 'Long', 'Short', 'eBird', 'Scientific'")
    parser.add_argument("--cpu_only", type=bool, default=None, help="Force to process the audio files with CPU if you don't have an NVIDIA GPU with sufficent memory")  # ISSUE #14: Boolean argument parsing is incorrect - should use action='store_true'
    parser.add_argument("--num_cores", type=int, default=None, help="Number of CPU cores.  The more the better, but it may crash your system")
    parser.add_argument("--birds_to_summarise", type=str, default=None, help="A string as a list of strings matching birds that need individual summary files")
    args = parser.parse_args()

    for key in options.keys():
        value = getattr(args, key)
        if value is not None:
            if key == 'birds_to_summarise':
                options[key] = ast.literal_eval(value)
            else:
                options[key] = value
    infer_soundscapes(options)