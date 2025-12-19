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
import torch.nn as nn
from joblib import Parallel, delayed
import multiprocessing
multiprocessing.freeze_support()
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from bird_naming_utils import BirdNamer
from pathlib import Path
import argparse
import yaml
import random
import ast
from test_cuda import test_cuda
from kaytoo_sed_model import BirdSoundModel
from kaytoo_processes import PrepareSpec, wav_to_bw_image, mono_to_color

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
            self.cores = (os.cpu_count() or 2)//2


class AudioParameters:
    def __init__(self, model_parameters):
        self.IMAGE_SHAPE = model_parameters['image_shape']  #Tpyical (2,.5)  For 12 seconds with buffer around one
        self.IMAGE_TIME = model_parameters['image_time']
        self.SR = 32000
        self.SPEC_WIDTH = model_parameters['spec_width']    #Typical 768
        self.IMAGE_WIDTH = int(model_parameters['image_shape'][1] * self.SPEC_WIDTH)  #Typical 384
        self.DOUBLE_AUDIO = model_parameters['double_audio']
        self.BUFFER_AUDIO = 1
        self.N_MELS = model_parameters['n_mels']  #Typical 192 (384/2)
        self.N_FFT = model_parameters['n_fft']
        self.FMIN = 20
        self.FMAX = 14000
        self.HOP_LENGTH = model_parameters['hop_length']
        self.PCEN = False
        self.USE_DELTAS = model_parameters['use_deltas']

    def __str__(self):
        lines = [f"{k}: {v}" for k, v in vars(self).items()]
        return "\n".join(lines)


class FilePaths:
    AUDIO_TYPES = {'.ogg','.wav', '.flac', '.mp3'}
    def __init__(self, options: dict):
        self.root_folder = Path(options['project_root'])  # ISSUE #1: No validation that paths exist or are accessible
        self.models_folder = self.root_folder / 'models'
        self.data_folder = self.root_folder / 'data'
        self.predictions = Path(options['results_folder'])
        self.bird_list_path = self.root_folder / 'resources/bird_map.csv'
        self.soundscapes_folder = Path(options['folder_to_process'])  # ISSUE #1: No validation that paths exist or are accessible
        self.soundscapes = [path for path in self.soundscapes_folder.rglob('*') if path.suffix in self.AUDIO_TYPES]
        self.predictions.mkdir(parents=True, exist_ok=True)


class ModelParameters:
    def __init__(self, options: dict):
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
                                            'pcen': False
                                            }, 
                            ] 
        '''
        if options['experiment'] is not None:
            _deploy_fldr = f"{options['project_root']}/data/experiments/exp_{options['experiment']}/exp_{options['experiment']}_deploy"
            last_ckpt =  Path(f"{options['project_root']}/data/experiments/exp_{options['experiment']}") / 'temp/checkpoints/last.ckpt'
            _deploy_folders = [Path(_deploy_fldr)]
        else:
            last_ckpt  = None
            _models_dir = Path(f"{options['project_root']}/models/")
            _deploy_folders = [subdir for subdir in _models_dir.iterdir() if subdir.is_dir() and subdir.name.endswith('_deploy')]
            
        self.parameters = []
        for model_fldr in _deploy_folders:
            print(f'Checking {model_fldr} for models...')
            pt_files = list(model_fldr.glob("*.pt"))
            cfg_files = list(model_fldr.glob("*.yaml"))
            if pt_files:
                latest_pt = max(pt_files, key=lambda f: f.stat().st_mtime)  #There should only be one  
                if cfg_files:
                    latest_cfg = max(cfg_files, key=lambda f: f.stat().st_mtime)  #There should only be one
                    with open(latest_cfg, "r") as f:
                        model_config = yaml.load(f, Loader=yaml.FullLoader)  # Using FullLoader to support Python tuples in YAML (safer than default loader)
                        model_config['pt_path'] = latest_pt
                        model_config['pcen'] = False
                        model_config['ckpt_path'] = last_ckpt
                        self.parameters.append(model_config)
                else:
                    print(f'Warning: No configuration file was found in {str(model_fldr)}')  # ISSUE #9: Use logging instead of print
            else:
                    print(f'Warning: No checkpoint file was found in {str(model_fldr)}')  # ISSUE #9: Use logging instead of print    


class Colour:
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def open_audio_clip(path: Path, default_sr: int = 32000, min_duration: int = 5) -> np.ndarray:
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

    # Resample if needed
    if sr != default_sr:
        num_samples = int(len(y) * default_sr / sr)
        y = resample(y, num_samples)
        sr = default_sr

    # Pad or trim to at least `min_duration` seconds
    min_samples = int(min_duration * default_sr)
    if len(y) < min_samples:
        pad_len = min_samples - len(y)
        y = np.concatenate([y, np.random.randn(pad_len)])
        print(f"[INFO] Padded {path} to {len(y)/default_sr:.1f} s")

    assert np.isfinite(y).all(), f"[FATAL] Non-finite values persist in {path}!"
    return y


def compute_melspec(y: np.ndarray,
                    sr: int,
                    hop_length: int,
                    n_mels: int,
                    n_fft: int,
                    audio_params: Optional[AudioParameters],
                    ) -> np.ndarray:
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




def get_images(audio_path: Path,
               model_params: Dict[str, Any],
               audio_params: AudioParameters,
               clip_length: Optional[int] = None,
               sr: int = 32000) -> Tuple[Dict[int, np.ndarray], int]:
    
    double = audio_params.DOUBLE_AUDIO
    buffer_sec = audio_params.BUFFER_AUDIO
    buffer = int(buffer_sec * sr)
    trim = abs(buffer)
    n_mels = audio_params.N_MELS
    spec_width = audio_params.SPEC_WIDTH  #Could be 2x the image width for the doubled audio scenario
    preparer = PrepareSpec(height=n_mels, width = spec_width)
    num_chunks = model_params['image_shape'][0] * model_params['image_shape'][1]
    image_time = audio_params.IMAGE_TIME
    
    if double:  #note that buffer_audio can be < 0
        chunk_length_sec = int(((image_time - 2 * buffer_sec) / 2) / num_chunks)  #eg (12 - 2*1)/2 / 1  = 5 (seconds)  To chunk the original clip into 5 secs
    else:
        chunk_length_sec = int((image_time-2*buffer_sec)/ num_chunks)

    idxs  = []
    image_dict = {}
    _y = open_audio_clip(audio_path, default_sr=sr) #resamples if needed to ensure sr = sr
    if clip_length is None:
        clip_length = len(_y) // sr

    total_chunks = int(clip_length // chunk_length_sec)
    
    #Let's ensure that _y is a whole number of chunks
    chunk_len_samples = int(chunk_length_sec * sr)
    n_samples = len(_y)
    remainder = n_samples % chunk_len_samples

    if remainder != 0:
        pad_len = chunk_len_samples - remainder
        last_chunk = _y[-remainder:]
        reps = int(np.ceil(pad_len / remainder))
        pad = np.tile(last_chunk, reps)[:pad_len]
        _y = np.concatenate([_y, pad])

    for index in range(0, total_chunks):
        idxs.append(index)
        start = index * chunk_length_sec
        stop = start + chunk_length_sec
        start_idx = sr * start
        stop_idx =  sr * stop

        if double and buffer>=0:
            if index == 0:
                y = np.concatenate((_y[:buffer], _y[:stop_idx], _y[:stop_idx + buffer]))
            elif index == total_chunks - 1:
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

        max_vol = np.max(np.abs(y))

        if max_vol > 0 and np.isfinite(max_vol):
            y = y / max_vol

        image_dict[index] = wav_to_bw_image(y,
                                audio_cfg=audio_params,
                                preparer=preparer
                                )
        #folding, transforms & mono to colour are done in the dataloader

    num_specs = len(image_dict)
    extra_specs = total_chunks % num_chunks  #Handle the case where there are more spectrograms needed to make up the combined image
    if extra_specs:  #For batch shape consistency
        noise = np.random.randn(image_time * sr)
        image = wav_to_bw_image(noise,
                                audio_cfg=audio_params,
                                preparer=preparer
                                )
        for extra_idx in range(num_specs+1, num_specs+extra_specs+1):
            image_dict[extra_idx] = image
    #Returns a dict of images, with keys from 0 to length // chunk_length (so 12 images for a 1-minute / 5-sec config)

    #So at this point it has only been normalised once, still needs scaling on [0,1], then ImageNet scaling 
    return image_dict, extra_specs  


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
        return y_vol


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
        self.image_pixels = self.image_dict.get(0, np.zeros((192, 768))).shape
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
            
        #print(f'before mono_to_colour but after spec_augment the min/max is {image.min()}, {image.max()}')  #Big numbers like -250, +230.  

        image = mono_to_color(image, use_deltas=self.use_deltas)  #Scales onto [0,1] then computes other channels
        #print(image.min(), image.max())
        #print("pre-albu min/max:", image.min(), image.max())
        image = self.image_transform(image=image)['image']  #To be exactly equivalent, this should be applied globally like this, not per chunk.
        #print("post-albu min/max:", image.min(), image.max())
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


class Models:
    def __init__(self, config, model_parameters):
        self.cfg = config
        self.args_list = model_parameters.parameters
        self.ebirds = config.classes
        self.device = config.device

    def get_model(self, idx: int) -> 'BirdSoundModel':
        model_args = self.args_list[idx]
        map_location = 'cpu' if self.device == torch.device('cpu') else 'cuda'
        audio_cfg = AudioParameters(model_args)
        print(model_args['basename'])

        path = model_args['pt_path']
        print(f'Loading PyTorch from pt path: {path}')
        model = BirdSoundModel(model_args['basename'],
                                len(self.ebirds),
                                model_args['image_shape'],
                                device=self.device)
        state_dict = torch.load(path, map_location=map_location)
        
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.parameters = model_args
        model.audio = audio_cfg
        model.to(self.device)
        model.device = self.device  # Ensure device attribute is set correctly
        return model


def prediction_for_clip(audio_path: Path,
                        model: BirdSoundModel,
                        sub_process: bool = False, 
                        MAX_BATCH_SIZE: int = 12
                        ) -> np.ndarray:

    model_args = model.parameters
    audio_params = model.audio
    device = model.device
    num_classes = model.num_classes
    
    image_dict, num_extras = get_images(audio_path, model_args, audio_params)
    num_images = len(image_dict)

    #At this point the images have wide ranges like [-150, +150]  from the get_images()  Because went db scale to scaling with max_pixel=1
    
    #Image dataset normalises onto [0,1], then the imagenet transfrom.
    dataset = ImageDataset(
        image_dict,
        model_args['image_shape'],
        model_args['use_deltas']
    )

    batch_size = min(MAX_BATCH_SIZE, num_images)

    if batch_size == 0:
        return np.zeros((0, num_classes), dtype=np.float32)

    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0
                        )

    if not sub_process:
        progress = tqdm(loader, desc="Inferring a single soundscape")
    else:
        progress = loader

    all_preds = []

    with torch.no_grad():
        for images, _ in progress:
            images = images.to(device)
            predictions = model(images)['clip_preds']
            preds = predictions.detach().cpu().numpy()
            all_preds.append(preds)

    # free memory
    del loader, dataset, image_dict
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()

    # stack all predictions
    all_preds = np.concatenate(all_preds, axis=0)

    # remove extra padding preds
    if num_extras:
        all_preds = all_preds[:-num_extras]

    return all_preds


def process_clip(audio_path: Path,
                 model: 'BirdSoundModel',
                 debug=False,)-> Tuple[List[np.ndarray], List[str], List[str]]:

    if debug:
        print('\nThe model parameters:')
        print(model.parameters)
        print('\nThe audio parameters')
        print(model.audio)

    batch_preds = prediction_for_clip(audio_path,
                                      model,
                                      sub_process=True)

    num_preds = batch_preds.shape[0]
    clip_preds = batch_preds.tolist()

    row_ids = [f"{audio_path.with_suffix('')}_{end}".replace("\\", "/") 
           for end in range(5, (num_preds+1)*5, 5)]
    
    file_paths = [str(audio_path).replace('\\', '/')] * num_preds

    return clip_preds, row_ids, file_paths


def inference(test_audios: List[Path],
              models: 'Models',
              model_idx: int,
              cores: int = 1,
              debug: bool = False)-> pd.DataFrame:
    bird_list = models.ebirds
    model = models.get_model(model_idx)

    results = Parallel(n_jobs=cores, backend='threading')(
        delayed(process_clip)(audio_path, model=model, debug=debug) for audio_path in tqdm(test_audios, desc="Overall File List")
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
            subset = group.drop(columns=['row_id']).copy()
            subset = subset.loc[:, (subset == 1).any()]
            duration_seconds = len(subset) * 5
            remaining_columns = subset.columns
            column_sums = subset.sum(axis=0).astype(int)
            total_birds = subset.sum().sum().astype(int)
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
            subset = group.drop(columns=['row_id']).copy()
            subset = subset.reset_index(drop=True)
            first_non_zero = subset.apply(lambda col: col.ne(0).idxmax()*5 if col.ne(0).any() else np.nan)
            return first_non_zero
        grouped = df.groupby('File_Path')
        first_bird = grouped.apply(_first_detected, include_groups=False).reset_index(drop=False)
        columns_to_convert = list(first_bird.columns[1:])
        first_bird.loc[:,columns_to_convert] = first_bird[columns_to_convert]#.astype(pd.Int64Dtype())
        return first_bird

    def last_detected(self, df: pd.DataFrame) -> pd.DataFrame:
        def _last_detected(group):
            subset = group.drop(columns=['row_id']).copy()
            subset = subset.reset_index(drop=True)
            length = len(subset) * 5
            last_non_zero = subset.apply(lambda col: length - (col[::-1].ne(0).idxmax()) * 5 if col.ne(0).any() else np.nan)
            return last_non_zero
        grouped = df.groupby('File_Path')
        last_bird = grouped.apply(_last_detected, include_groups=False).reset_index(drop=False)
        columns_to_convert = list(last_bird.columns[1:])
        last_bird.loc[:,columns_to_convert] = last_bird[columns_to_convert]#.astype(pd.Int64Dtype())
        return last_bird

    def detections_per_minute(self, df: pd.DataFrame) -> pd.DataFrame:
        def _detections_per_minute(group):
            subset = group.drop(columns=['row_id']).copy()
            subset = subset.reset_index(drop=True)
            minutes = len(subset) / 12
            column_sums = subset.sum(axis=0).astype(int)
            bird_rate = column_sums / minutes
            bird_rates = pd.Series(bird_rate, index=subset.columns).round(3).astype('float32')
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
    paths = FilePaths(options=use_case)
    bird_map_df = pd.read_csv(paths.bird_list_path)
    birdnames = BirdNamer(bird_map_df)
    cfg = DefaultConfig(bird_namer=birdnames, 
                        options=use_case)
    parameters = ModelParameters(options=use_case)
    print(parameters.parameters[0])
    models = Models(config=cfg, 
                    model_parameters=parameters)
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
        #print(f'The final predictions dataframe has length {len(df)}')
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
    bin_preds = predictions.copy()
    bin_preds.iloc[:,2:] = (bin_preds.iloc[:,2:] > threshold).astype(int)

    print(Colour.S + 'Thresholded scores for the first 8 birds' + Colour.E)
    print(bin_preds.iloc[:5, :8])

    if naming_scheme == 'Short':
        short_names  = birdnames.extra_names(birdnames.bird_list)  #we need to do this way for the one-many relationship
        bin_preds = merge_classes(bin_preds, birdnames.bird_list, short_names)

        print(Colour.S + 'Merged scores for the first 8 birds' + Colour.E)
        print(bin_preds.iloc[:5, :8])

    #Derive various alternative data represenations
    post_processor = DeriveResults(bin_preds,  
                                   save_folder=paths.predictions,
                                   )
    post_processor.derive_results()
    if summary_birds and (summary_birds[0] != 'All'):
        print(Colour.S + '\nBirds that will have individual summary files are: ' + Colour.E, summary_birds)
        post_processor.summarise_chosen_birds(summary_birds)
    post_processor.save_results()
    post_processor.print_results()
    return predictions

############################################  Run Main  ##########################################
##################################################################################################

if __name__ == '__main__':
    #Default options for running during development
    options = {
            'project_root': '/home/olly/Desktop/Kaytoo', #'/media/olly/T7/Kaytoo', #'/media/olly/T7/Kaytoo', # 'G:/Kaytoo',  #'/media/olly/T7/Kaytoo'  
            'experiment': None, #None to use what ever is in the Models folder, otherwise an integer for the experiment number
            'threshold': 0.3,
            #'folder_to_process': '/home/olly/Desktop/Kaytoo/data/more_corrupt_files',# 'D:/Kaytoo/Data/Corrupt_Files', #'D:/Kaytoo/Data/Soundscapes/debugging',
            #'folder_to_process':'/home/olly/Desktop/Kaytoo/data/moira_samples',
            'folder_to_process':'/home/olly/Desktop/Kaytoo/data/Train_Xeno_Canto/train_audio/morepo2',
            'results_folder': '/home/olly/Desktop/Kaytoo/data/moira_samples',
            'naming_scheme' : 'Short', #'Scientific', #'Short, Long, Scientific, eBird'
            'cpu_only': True,
            'num_cores': 1,  #Can crank this up if using CPU only.
            'birds_to_summarise':['Morepork', 'Kaka'], #['Chlidonias albostriatus']  #['Australian Magpie'],  Black-fronted Tern blfter1	Chlidonias albostriatus	Tern
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