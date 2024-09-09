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
from tqdm import tqdm
import pandas as pd
from torch.utils.data import  DataLoader
from bird_naming_utils import BirdNamer
from pathlib import Path
import re
import argparse


############################################# Parameters  ######################################
##################################################################################################

class DefaultConfig:
    def __init__(self, bird_namer, options=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if options:
            if options['cpu_only']:
                self.device = torch.device('cpu')
            if options['num_cores']:
                self.CORES = options['num_cores']
            else:
                self.CORES = 1
        self.classes = bird_namer.bird_list



class AudioParameters:
    def __init__(self):
        self.SR = 32000
        self.FMIN = 20
        self.FMAX = 14000 


class FilePaths:
    AUDIO_TYPES = {'.ogg','.wav', '.flac', '.mp3'}
    def __init__(self, options=None):
        self.root_folder = Path(options['project_root'])
        self.data_folder = self.root_folder / 'Data'
        self.experiment_results = self.data_folder / f"Experiments/Exp_{options['experiment']}/Results"
        self.bird_list_path = self.experiment_results / f"exp_{options['experiment']}_bird_map.csv"
        self.soundscapes_folder = self.root_folder /options['folder_to_process']
        self.soundscapes = [path for path in self.soundscapes_folder.rglob('*') if path.suffix in self.AUDIO_TYPES]
        self.predictions_csv = self.soundscapes_folder / 'predictions.csv'
        self.learning_rate_monitor = self.experiment_results / f"exp_{options['experiment']}_training_metrics.jpg"
        self.train_metric_monitor = self.experiment_results / 'learning_rate.jpg'
        self.val_preds = self.experiment_results / 'val_pred_df.pkl'
        self.val_targs = self.experiment_results / 'val_target_df.pkl'


class ModelParameters:
   def __init__(self, options=None):
        self.parameters = [
        {'basename':'tf_efficientnet_b0.ns_jft_in1k', 
                         'ckpt_path': f"{options['project_root']}/Data/Experiments/Exp_{options['experiment']}/Results/last.ckpt",
                         'image_shape': (1,2), #The layout of 5-sec spectrograms stacked into the final image (height x width)
                         'image_time': 10,
                         'n_mels': 256,
                         'n_fft': 2048,
                         'use_deltas' : True,
                         'hop_length': 1243,
                         '5_sec_width': 128,
                         'aggregation': 'mean',
                         }, 
        ]
        if options and options.get('model_choice'):
            model_choices = options['model_choice']
            _parameters_list = [_parameters_list[i] for i in model_choices] 


class Colour:
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'

def open_audio_clip(path, default_sr = 32000):   
    #Modify this to re-sample anything not using 32,000 sample rate
    try:  
        y, sr = torchaudio.load(path)
        if y.ndim == 2 and y.shape[0] == 2:
            y = torch.mean(y, dim=0).unsqueeze(0)  # from stereo to mono
        y = y.squeeze().numpy() 
    except:
        y = np.random.randn(5*default_sr)
        sr = default_sr 
        print(f'could not open {path}')

    if sr != default_sr:
        num_samples = int(len(y) * default_sr / sr)
        y = resample(y, num_samples)
    
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.zeros_like(y)
        y[np.isinf(y)] = np.max(y)
    return y



def compute_melspec(y, sr, hop_length, n_mels, n_fft, audio_params):
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


def get_images(audio_path, #PathLib Path object
               model_params,
               audio_params,
               clip_length=None,
               sr=32000):
    
    hop_length = model_params['hop_length']
    n_mels = model_params['n_mels']
    n_fft = model_params['n_fft']
    chunk_width = model_params['5_sec_width']
    num_chunks = model_params['image_shape'][0] * model_params['image_shape'][1]
    chunk_length = model_params['image_time']//(num_chunks)
    prep_image = PrepareImage(height=n_mels, width=chunk_width)

    idxs  = []
    image_dict = {}
    _y = open_audio_clip(audio_path)
    if clip_length is None:
        clip_length = len(_y) // sr

    for index in range(0, clip_length // chunk_length):
        idxs.append(index)
        start = index * chunk_length
        stop = start + chunk_length
        start_idx = sr * start
        stop_idx =  sr * stop
        
        if stop_idx > len(_y):
            y = _y[start_idx:]
            remaining_length = stop_idx - len(_y)  # Calculate how much we need to fill
            noise = np.random.randn(remaining_length) * np.std(_y)
            y = np.concatenate((y, noise))
        else: 
            y = _y[start_idx: stop_idx]
        
        max_vol = np.abs(y).max()
        y = y * 1 / max_vol    #y, sr, hop_length, n_mels, n_fft, audio_params
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


def mono_to_color(X, eps=1e-6, use_deltas=False):
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


def crop_or_pad(y, length, train='train'):
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
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.valid = A.Compose([
                        A.PadIfNeeded(min_height=self.height, min_width=self.width),
                        A.CenterCrop(width=self.width, height=self.height),
                        A.Normalize(self.mean, self.std, max_pixel_value=1.0,always_apply=True),
                        ])
        

class ImageDataset(Dataset):
    def __init__(self, image_dict, image_shape, use_deltas): #, model_args
        self.image_dict = image_dict
        self.image_shape = image_shape
        self.image_pixels = self.image_dict[0].shape
        self.height = self.image_shape[0] * self.image_pixels[0]  #Shape of the combined image from one __get_item__
        self.width = self.image_shape[1] * self.image_pixels[1] 
        self.image_transform = AbluTransforms(height=self.height, width=self.width).valid
        self.use_deltas = use_deltas
        self.chunks_per_image = self.image_shape[0] * self.image_shape[1]
               
    def __len__(self):
        whole = len(self.image_dict) // self.chunks_per_image
        remainder =  1 if len(self.image_dict) % self.chunks_per_image != 0 else 0
        return  whole + remainder

    def __getitem__(self, idx):
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
        else:
            image = images[0]

        image = mono_to_color(image, use_deltas=self.use_deltas)
        image = self.image_transform(image=image)['image']
        image = image.transpose(2,0,1).astype(np.float32) # swapping the image channels to the first axis
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


    class AttentionBlock(nn.Module):
        def __init__(self,
                     in_features: int,
                     out_features: int,
                     shape:tuple,
                     activation="linear",
                     aggregation='mean',
                    ):
            super().__init__()

            self.activation = activation
            self.attention = nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True)
            self.classify = nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True)
            self.init_weights()
            self.chunks_high = shape[0]
            self.chunks_wide = shape[1]
            self.num_chunks = shape[0] * shape[1]
            self.aggregation = aggregation
        
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
            batch_size = x.shape[0]
            split_length = x.shape[2] // self.num_chunks  #The number of segments per 5 sec time chunk
            splits = torch.split(x, split_length, dim=2)
            x = torch.cat(splits, dim=0)
            
            norm_att = torch.softmax(torch.tanh(self.attention(x)), dim=-1) / (16//split_length)  # /4 or 2 or 1 so the 16 values sum to 1
            classify_logits = self.classify(x) 
            logits_with_attn  = norm_att * classify_logits * self.num_chunks
            

            if self.aggregation == 'mean':
                chunk_preds = self.nonlinear_transform(logits_with_attn.sum(dim=2, keepdim=True))
                chunk_splits = torch.split(chunk_preds, batch_size, dim=0)
                chunk_preds = torch.cat(chunk_splits, dim=2)
            
            elif self.aggregation == 'mean-max':
                chunk_preds = self.nonlinear_transform((logits_with_attn * self.num_chunks).sum(dim=2, keepdim=True))
                chunk_splits = torch.split(chunk_preds, batch_size, dim=0)
                chunk_preds_mean = torch.cat(chunk_splits, dim=2)
                
                chunk_logits_max, _ = classify_logits.max(dim=2, keepdim=True)
                chunk_preds_max = self.nonlinear_transform(chunk_logits_max)
                chunk_splits = torch.split(chunk_preds, batch_size, dim=0)
                chunk_preds = torch.cat(chunk_splits, dim=2)
                
                chunk_preds = (chunk_preds_mean + chunk_preds_max) / 2
            
            elif self.aggregation =='max':
                chunk_logits_max, _ = classify_logits.max(dim=2, keepdim=True)
                chunk_preds = self.nonlinear_transform(chunk_logits_max)
                chunk_splits = torch.split(chunk_preds, batch_size, dim=0)
                chunk_preds = torch.cat(chunk_splits, dim=2)
            
            chunk_preds = chunk_preds.transpose(1,2) #Putting the class predictions last
            chunk_preds = chunk_preds.reshape(chunk_preds.shape[0]*chunk_preds.shape[1], -1)  #flatten to (num_preds,num_classes)
                        
            return chunk_preds

    
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
        self.bn0 = nn.BatchNorm2d(self.image_width)   #self.image_width  #why is this still 256???
        
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
                                            self.num_classes,
                                            self.image_shape,
                                            activation="sigmoid",
                                            aggregation = self.aggregation,
                                            )
        self.init_weight()
        self.val_outputs = []
        self.train_outputs = []
        self.metrics_list = []
        self.val_epoch = 0
        

    def forward(self, input_data):
        x = input_data  #(batch_size, 3, frequency, time)  #This needs to match the the output of dataloader & getitem 
        x = x.transpose(1, 3)  #(batch_size, mel_bins, time_steps, channels)
        x = self.bn0(x)
        x = x.transpose(1, 3)
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
        
        #This is the guts of the SED part.
        x = torch.mean(x, dim=3) # Aggregate in the horizontal (time) axis, so now we've just got a 3d tensor (batch_size, n_features, freq-time chunks)       
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

    def get_model(self, idx):
        model_args = self.args_list[idx]
        path = model_args['ckpt_path']
        map_location = 'cpu' if self.device == torch.device('cpu') else 'cuda'
        ckpt = torch.load(path, map_location=map_location)
        model = BirdSoundModel(self.ebirds, model_args)
        model.load_state_dict(ckpt)
        model.eval()
        model.parameters = self.args_list[idx]
        model.audio = self.audio
        model.to(self.device)
        return model  
    

def prediction_for_clip(audio_path,
                        model,
                        sub_process=False):
    model_args=model.parameters
    audio_params=model.audio
    device = model.device 
    
    image_dict, num_extras = get_images(audio_path, model_args, audio_params)
    num_images = len(image_dict)
    dataset = ImageDataset(image_dict, model_args['image_shape'], model_args['use_deltas'])
    shape = model_args['image_shape']
    num_chunks = shape[0] * shape[1]
    batch_size = num_images // num_chunks  #should be a whole number, because we made sure of this in the get_images
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    
    if not sub_process:
        progress = tqdm(range(len(loader)), desc="Inferring a single soundscape")
    
    for images, _ in loader:
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
        batch_segment_preds = batch_segment_preds[:-num_extras, :]

    return batch_segment_preds


def process_clip(audio_path, model):
    clip_preds = []
    final_row_ids = []
    batch_preds = prediction_for_clip(audio_path,
                                      model,
                                      sub_process=True)
    
    num_preds = batch_preds.shape[0]
    
    for idx in range(num_preds):
        row = batch_preds[idx]
        clip_preds.append(row)
    
    row_ids = [audio_path.stem + f'_{end}' for end in range(5, (num_preds+1)*5, 5)]
    final_row_ids.extend(row_ids)

    return clip_preds, final_row_ids


def inference(test_audios, models, model_idx, cores=1):
    bird_list = models.ebirds
    model = models.get_model(model_idx)
    results = Parallel(n_jobs=cores)(
        delayed(process_clip)(audio_path, model=model) for audio_path in tqdm(test_audios, desc="Overall File List")
        )
    
    del model
    if models.device == torch.device('cuda'):
        torch.cuda.empty_cache()

    clip_preds = [result[0] for result in results]      #This should be a list of 1 x num_classes arrays, with a length = the number of predictions
    final_row_ids = [result[1] for result in results]   #This should be a list of row id's?

    clip_preds = [item for sublist in clip_preds for item in sublist]
    final_row_ids = [item for sublist in final_row_ids for item in sublist]
    prediction_df = pd.DataFrame(clip_preds, columns=bird_list)
    prediction_df.insert(0, 'row_id', final_row_ids)
    
    return prediction_df


class DeriveResults():
    def format_predictions(self, df, paths_list, threshold=0.5):
        df.iloc[:,1:] = (df.iloc[:,1:] > threshold).astype(int)
        file_map = {re.sub(r'_\d+\..*$', '', Path(fp).stem): fp for fp in paths_list}
        df['root'] = df['row_id'].str.replace(r'_\d+$', '', regex=True)
        df['filepath'] = df['root'].map(file_map)
        df.drop(columns=['root'], inplace=True)
        return df

    def __init__(self, predictions, paths_list, save_folder=None, threshold=0.5):
        self.predictions = self.format_predictions(predictions, paths_list, threshold=threshold)
        self.save_folder = save_folder

    def summarise(self, df):
        def _summarise(group):
            group.drop(columns=['row_id'], inplace=True)
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

        grouped = df.groupby('filepath')
        summary_df = grouped.apply(_summarise, include_groups=False).reset_index(drop=False)
        return summary_df

    def first_detected(self, df):
        def _first_detected(group):
            group.drop(columns=['row_id'], inplace=True) 
            group = group.reset_index(drop=True)
            first_non_zero = group.apply(lambda col: col.ne(0).idxmax()*5 if col.ne(0).any() else np.nan)
            return first_non_zero
        grouped = df.groupby('filepath')
        first_bird = grouped.apply(_first_detected, include_groups=False).reset_index(drop=False)
        columns_to_convert = first_bird.columns[1:]
        first_bird[columns_to_convert] = first_bird[columns_to_convert].astype(pd.Int64Dtype())
        return first_bird

    def last_detected(self, df):
        def last_detected(group):
            group.drop(columns=['row_id'], inplace=True) 
            group = group.reset_index(drop=True)
            length = len(group) * 5
            last_non_zero = group.apply(lambda col: length - (col[::-1].ne(0).idxmax()) * 5 if col.ne(0).any() else np.nan)
            return last_non_zero
        grouped = df.groupby('filepath')
        last_bird = grouped.apply(last_detected, include_groups=False).reset_index(drop=False)
        columns_to_convert = last_bird.columns[1:]
        last_bird[columns_to_convert] = last_bird[columns_to_convert].astype(pd.Int64Dtype())
        return last_bird

    def detections_per_minute(self, df):
        def _detections_per_minute(group):
            group.drop(columns=['row_id'], inplace=True) 
            group = group.reset_index(drop=True)
            minutes = len(group) / 12
            column_sums = group.sum(axis=0).astype(int)
            bird_rate = column_sums / minutes
            bird_rates = pd.Series(bird_rate, index=group.columns).round(3).astype('float32')
            return bird_rates
        grouped = df.groupby('filepath')
        bird_rate = grouped.apply(_detections_per_minute, include_groups=False).reset_index(drop=False) 
        return bird_rate

    def derive_results(self):
        self.summary = self.summarise(self.predictions)
        self.first_bird = self.first_detected(self.predictions)
        self.last_bird = self.last_detected(self.predictions)
        self.bird_rate = self.detections_per_minute(self.predictions)

    def save_results(self, save_folder=None):
        save_folder = save_folder if save_folder is not None else self.save_folder
        save_folder = Path(save_folder) if not isinstance(save_folder, Path) else save_folder
        self.summary.to_csv(save_folder / 'detection_summary.csv')
        self.first_bird.to_csv(save_folder / 'first_bird.csv')
        self.last_bird.to_csv(save_folder / 'last_bird.csv')
        self.bird_rate.to_csv(save_folder / 'detections_per_minute.csv')

    def print_results(self):
        print(Colour.S + '\nThe summary dataframe' + Colour.E)
        print(self.summary.iloc[:3,:8])
        print(Colour.S + '\nThe first detection time for each species (s)' + Colour.E)
        print(self.first_bird.iloc[:3,:8])
        print(Colour.S + '\nThe last detection time from the end for each species  (s)' + Colour.E)
        print(self.last_bird.iloc[:3,:8])
        print(Colour.S + '\nThe detection rate, bird per minute for each species' + Colour.E)
        print(self.bird_rate.iloc[:3,:8])


############################################# Main Function  #####################################
##################################################################################################

def infer_soundscapes(use_case):
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

    print('The inference folder is:', paths.soundscapes_folder)
    print(f'There are {len(models.args_list)} model(s) to be ensembled')
    print(f'The model(s) will predict the following {len(models.ebirds)} birds (referring to their https://ebird.org code): \n')
    for i in range(0, len(models.ebirds), 10):
        print(", ".join(models.ebirds[i:i + 10]))

    prediction_dfs = []
    for idx in range(len(models.args_list)):
        df = inference(paths.soundscapes, models, idx, cores=cfg.CORES)
        prediction_dfs.append(df)

    prediction_columns = prediction_dfs[0].columns[1:]
    values_list = [df[prediction_columns].values for df in prediction_dfs]
    average_vals = np.zeros_like(values_list[0])

    for array in values_list:
        average_vals = average_vals + array 

    average_vals = average_vals / len(values_list)
    predictions = pd.DataFrame(data=average_vals, columns=prediction_columns)
    predictions.insert(0, 'row_id', prediction_dfs[0]['row_id']) 
    predictions.to_csv(paths.predictions_csv, index=False)

    print(Colour.S + 'Raw prediction scores for the first 8 birds' + Colour.E)
    print(predictions.iloc[:5, :8])

    post_processor = DeriveResults(predictions, 
                                   paths_list=paths.soundscapes, 
                                   save_folder=paths.soundscapes_folder)
    post_processor.derive_results()
    post_processor.save_results()
    post_processor.print_results()

############################################  Run Main  ##########################################
##################################################################################################

if __name__ == '__main__':
    options = {
                'project_root': 'D:\Kaytoo', #'/media/olly/T7/Kaytoo', #'/media/olly/T7/Kaytoo', # 'G:/Kaytoo',  #'/media/olly/T7/Kaytoo'  
                'experiment': 19,
                'threshold': 0.2,
                'folder_to_process': 'Data/Soundscapes/DOC_Tier1_2011/',
                'model_choices': [0],
                'cpu_only': False,
                'num_cores': 1  #Can crank this up if using CPU only.
                }

    parser = argparse.ArgumentParser()
    parser.add_argument("--kaytooPath", type=str, default=None, help="Filepath to the root directory (the parent of the 'Python' folder)")
    parser.add_argument("--threshold", type=str, default=None, help="The prediction threshold.  Currently 0.3 looks about right")
    parser.add_argument("--audioFolder", type=str, default=None, help="path to the processing folder, relative to the project root")
    parser.add_argument("--cpuOnly", type=str, default=None, help="Force to process the audio files with CPU if you don't have an NVIDIA GPU with sufficent memory")
    parser.add_argument("--num_cores", type=str, default=None, help="Number of CPU cores.  The more the better, but it may crash your system")

    args = parser.parse_args()
    infer_soundscapes(options)