use_case = {
                'experiment' : 9,
                'computer_name' : 'Alita',  #G7:Selene, G15:Alita, Latitude 3420: Calculon
                'run_training' : True,
                'epochs' : 20
                }


#General Python
import gc
from pathlib import Path
from tqdm.notebook import tqdm
import ast
from ast import literal_eval
from functools import reduce

#Math & Plotting
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'notebook'

#Machine Learning 
import albumentations as A
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics as skm

#Torch and PyTorch specific
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,  EarlyStopping
from torch.utils.data import  DataLoader, Dataset, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchaudio.functional import compute_deltas

#Audio
import librosa
import torchaudio
import colorednoise as cn


class FilePaths:
    PROJECT_DIR = Path('/media/olly/T7/Kaytoo')
    #PROJECT_DIR = Path("E:\Kaytoo")
    DATA_DIR = PROJECT_DIR / 'Data' 
    EXTRA_TRAIN_AUDIO = str(DATA_DIR / 'Extra_Data/train_audio')
    EXTRA_TRAIN_LABELS = str(DATA_DIR / 'Extra_Data/extra_pseudo_train_labels_mp3.csv')
    TRAIN_DATA_DIR = DATA_DIR / 'Train_Metadata'
    LABELS_PATH = str(TRAIN_DATA_DIR / 'train_metadata.csv')
    TRAIN_AUDIO_DIR = str(DATA_DIR)
    BACKGROUND_NOISE_FLDR =  str(DATA_DIR / 'background_noise')
    
    #currently unused
    SOUNDSCAPE_FLDR = str(TRAIN_DATA_DIR / '2021_soundscapes')
    SOUNDSCAPES_1 = '/media/olly/T7/BirdCLEF_Data/GoogleBird_25/unlabeled_soundscapes'
    PSEUDOLABELS_1 = '/media/olly/T7/BirdCLEF_Data/GoogleBird_25/google_bird_labels.csv'
    SOUNDSCAPES_2 = str(DATA_DIR / 'GoogleBird_40_800/unlabeled_soundscapes')
    PSEUDOLABELS_2 = str(DATA_DIR / 'GoogleBird_40_800/google_bird_labels.csv')
    TRAIN_PSEUDOLABELS_PATH = str(TRAIN_DATA_DIR / 'Pseudolabels' / 'pseudo_train_labels.csv')
    
    def __init__(self, options=None):
            _experiment = options['experiment']
            self.temp_dir = str(self.DATA_DIR / f'Experiments/Exp_{_experiment}' / 'Temp')
            self.chkpt_dir = self.temp_dir  + '/checkpoints'
            self.out_dir = self.DATA_DIR / f'Experiments/Exp_{_experiment}' / 'Results'
            self.best_weights_dir = self.out_dir / 'Best_Weights'
            self.last_weights_path = str(Path(self.chkpt_dir) / 'last.ckpt')
            self.bird_names_map = str(self.DATA_DIR  / 'Bird_Names/bird_map_corrected.csv')
            self.bird_map_for_model = self.out_dir / f'exp_{_experiment}_bird_map.csv'
            self.background_noise_paths = [path for path in Path(self.BACKGROUND_NOISE_FLDR).rglob('*') if path.suffix in {'.ogg', '.flac'}]
            


class DefaultConfig:
    def __init__(self, options=None):
        #Training Parameters
        self.TRAIN = use_case['run_training']
        self.EPOCHS = use_case['epochs'] 
        self.YEAR = 24
        self.EXPERIMENT = use_case['experiment']
        self.NUM_WORKERS = 3 if use_case['computer_name'] == 'Selene' else 3
        self.BATCH_SIZE = 32 # 12,  16, 32, 64 for sizes 512, 348, 32-larger network, 256
        self.TEST_BATCH_SIZE = 16
        self.PATIENCE = 3
        self.KEEP_LAST= 6
        self.MIN_DELTA = 0
        self.SEED = 2024
        self.MODEL = 'tf_efficientnet_b0.ns_jft_in1k' #, #'eca_nfnet_l0' #'tf_efficientnet_b0.ns_jft_in1k' #'convnext_tiny.in12k_ft_in1k' #'convnext_tiny.fb_in22k', 'eca_nfnet_l0' #  # 'tf_efficientnetv2_s.in21k_ft_in1k'
        self.WEIGHTED_SAMPLING = True
        self.WEIGHT_DECAY = 1e-5
        self.WARMUP_EPOCHS = 2
        self.INITIAL_LR = 1e-4 
        self.LR = 1e-3
        self.MIN_LR = 1e-5
        self.LR_CYCLE_LENGTH = 14
        self.LR_DECAY = 0.2
        self.EPOCHS_TO_UNFREEZE_BACKBONE = 8
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GPU = 'gpu' if torch.cuda.is_available() else 'cpu' #for Pytorch Lightning
        self.PRECISION = '16-mixed' if self.GPU == 'gpu' else 32
        self.LOSS_FUNCTION_NAME =  'BCEFocal2WayLoss' #'BCEWithLogitsLoss', 'BCEFocalLoss',
        self.USE_MIXUP = True
        self.MIXUP_ALPHA = .25 #Tried .4 and performance droped slightly from .64 to .63 (so inconclusive)

        #CV & Data Parameters
        self.N_FOLDS = 10
        self.USE_SECONDARY = True
        self.RARE_THRESHOLD = 10 # Classes with less samples than this will not be allowed in validation dataset, and will be up-sampled to 10
        self.USE_UNLABELED_SOUNDSCAPES = False #True
        self.USE_EXTRA_TRAIN_AUDIO = False
        self.FINETUNE_BY_LOCATION = False #Tried this, it didn't help
        self.USE_EXTRA_DATA_THRESHOLD = 100  #If fewer samples than this, then extra data will be used
        self.MAX_PER_CLASS = 20000 #Cap the maximum number of samples allowed in any particular class
        self.EXCLUDED_CLASSES = []
        #currently unused
        self.FIRST_DATA_UPDATE = 3
        self.SECOND_DATA_UPDATE = 7
        self.PSEUDOLABELS_CLASS_LIMIT_1 = 800
        self.PSEUDOLABELS_CLASS_LIMIT_2 = 400


class DefaultAudio:
    def __init__(self, options=None):
        self.IMAGE_SHAPE = (1,2)  #5 second chunks position in final image: height x width
        self.DURATION = 10  # Duration the loaded sound file will be randomly cropped or padded to for training.
        self.SR = 32000
        self.IMAGE_WIDTH = 256 #384 #512 # 256 #The spectrogram will get cropped/padded to this square regardless of any audio considerations
        self.CHUNK_WIDTH = self.IMAGE_WIDTH if self.IMAGE_SHAPE[1] == 1 else self.IMAGE_WIDTH // 2  #Number of frames wide for each sub-image
        self.N_MELS = self.IMAGE_WIDTH // 2 if self.IMAGE_SHAPE[0]== 2 else self.IMAGE_WIDTH #Height of the chunk spectrograms
        self.N_FFT = 2048 #3072 #2048 *2 #3072 or 2048 #N_fft/2 + 1 bins will get made prior to downsampling to the value of N_MELS
        self.FMIN = 20
        self.FMAX = 14000 
        self.HOP_LENGTH = 1243 #826 #620 #310, 620, 826, 1243, for chunks widths of 128, 192, 256, 516 respectively
        self.PCEN = False
        self.USE_DELTAS = True


class Stop: #bold red
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'
    
class Go: #bold green
    S = '\033[1m' + '\033[32m'
    E = '\033[0m'
    
class Blue: #for general info
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def get_pseudos(path):
    '''
    returns a dict of list of lists.  Each sub-list is the prediction values, 
    the position of the sub-list corresponts to the time-position
    in the sample, with each list representing a chunk of 5 seconds
    '''
    pseudo_df = pd.read_csv(path)

    if 'latitude' in pseudo_df.columns and 'longitude' in pseudo_df.columns:
        pseudo_df.drop(columns=['latitude', 'longitude'], inplace=True)

    #drop any rows where all the values are 0
    cols_after_4th = pseudo_df.columns[4:]
    mask = (pseudo_df[cols_after_4th] == 0).all(axis=1)
    pseudo_df = pseudo_df[~mask]

    print(pseudo_df.iloc[:,:6].head())

    grouped = pseudo_df.groupby('filename')
    birdlist = pseudo_df.iloc[:,4:].columns.tolist()
    print(f'There are {len(birdlist)} birds in the value columns')
    pseudo_dict = {}

    for filename, group in grouped:
        group_sorted = group.sort_values(by='time')
        values = group_sorted[birdlist].values.tolist()
        pseudo_dict[filename] = values

    return pseudo_dict


def load_sf(wav_path):
    y, _ = torchaudio.load(wav_path)
    y = y.squeeze().numpy() 
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.mean(y)
    return y
    

def balance_primary_label(df, label_column='primary_label', max_count=200):
    value_counts = df[label_column].value_counts()
    balanced_df = pd.DataFrame(columns=df.columns)
    for value, count in value_counts.items():
        value_df = df[df[label_column] == value]
        if count > max_count:
            value_df = value_df.sample(n=max_count, random_state=1)
        balanced_df = pd.concat([balanced_df, value_df], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=1).reset_index(drop=True)
    
    return balanced_df

    
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
    
    melspec = librosa.feature.melspectrogram(y=y, 
                                             sr=audio.SR, 
                                             n_mels=audio.N_MELS, 
                                             n_fft=audio.N_FFT, 
                                             hop_length = audio.HOP_LENGTH, 
                                             fmin=audio.FMIN, 
                                             fmax=audio.FMAX
                                            ) 
    return librosa.power_to_db(melspec)


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
        X = np.stack([X, delta, delta_2], axis=-1)
    else:
        X = np.stack([X, X, X], axis=-1) #puts the chanels last, like a normal image
    
    return X


def crop_or_pad(y, length,  train='train', path=None, background_paths=None):
    initial_length = len(y)
    max_vol = np.abs(y).max()
    if max_vol == 0:
        print('Warning, there was training sample of all zeros before padding')
        if path is not None:
            print(f'The filepath of this sample was {path}')
    if initial_length == 0:
        print('Warning, there was a sample of initial length zero before padding')
    if 3 * initial_length < length:
        random_values = np.random.random(initial_length)
        y = np.concatenate([y,random_values,y])
    elif 2 * initial_length < length:
        random_values = np.random.random(initial_length//2)
        y = np.concatenate([y,random_values,y])
    if len(y) < length:
        y = np.concatenate([y, y]) 
    
    def Normalize(array):
        max_vol = np.abs(array).max()
        if max_vol == 0:
            length = len(array)
            array = np.random.random(length)
            print('Warning, there was a final training sample of all zeros, replacing with random noise')
            return array  # or return array filled with zeros, if appropriate
        return array * 1 / max_vol

    if len(y) < length:
        difference = length - len(y)
        fill=np.zeros(difference)
        y = np.concatenate([y, fill])
    else:
        if train != 'train':
            start = 0
        else:
            start = 0
            start = np.random.randint(len(y) - length)
        y = y[start: start + length]
    y = Normalize(y)
    return y


def random_crop(arr, length):
    '''For cropping backgrounds from a larger clip to a chosen length'''
    if len(arr) > length:
        start = np.random.randint(len(arr) - length)
        arr = arr[start: start + length]
    return arr


def padded_cmap(solution, submission, padding_factor=5):
    solution = solution.fillna(0).replace([np.inf, -np.inf], 0)
    submission = submission.fillna(0).replace([np.inf, -np.inf], 0)
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = skm.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro')    
    return score
     

def padded_cmap_by_class(solution, submission, padding_factor=5):
    solution = solution.fillna(0).replace([np.inf, -np.inf], 0)
    submission = submission.fillna(0).replace([np.inf, -np.inf], 0)
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    
    column_headers = list(solution.columns)
    scores = {}
    
    for column in column_headers:
        score = skm.average_precision_score(
            padded_solution[[column]].values,
            padded_submission[[column]].values,
            average='macro')    
        scores[column] = score
    return scores


def map_score(solution, submission):
    solution = solution.fillna(0).replace([pd.np.inf, -pd.np.inf], 0)
    submission = submission.fillna(0).replace([pd.np.inf, -pd.np.inf], 0)
    score = skm.average_precision_score(
        solution.values,
        submission.values,
        average='micro')  
    return score


def plot_by_class(df_target, df_pred):
    cmap5_by_class = padded_cmap_by_class(df_target, df_pred, padding_factor=5)
    col_sums = [(col, df_target[col].sum()) for col in df_target.columns]
    names_by_frequency = sorted(col_sums, key=lambda x: x[1], reverse=True)
    names = [name for name, _ in names_by_frequency]
    counts = [count for _, count in names_by_frequency]
    scores = [cmap5_by_class[name] for name in names]
    df = pd.DataFrame({'names': names, 'counts': counts, 'scores': scores})
    df["scores"] = pd.to_numeric(df["scores"])
    df["counts"] = pd.to_numeric(df["counts"])
    fig = px.bar(df, x='scores', y='names', color='counts', orientation='h', hover_data=['counts', 'scores'], range_x=[0, 1])
    fig.update_layout(height=1200)
    fig.show()
    return names, scores, counts


def subset_by_location(df):
    WEST_LIMIT = 60
    EAST_LIMIT = 110
    SOUTH_LIMIT = 3
    NORTH_LIMIT = 31

    subset_df = df[(df['longitude'] >= WEST_LIMIT) &
               (df['longitude'] <= EAST_LIMIT) &
               (df['latitude'] >= SOUTH_LIMIT) &
               (df['latitude'] <= NORTH_LIMIT)]
    return subset_df



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
        
        
class OneOf(Compose):
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (np.random.random() < self.p):
            random_state = np.random.RandomState(np.random.randint(0, 2 ** 16 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data
    
    
class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)
    
    
class RandomNoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=1):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented
    
    
class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented
    
#https://github.com/felixpatzelt/colorednoise
class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented
    
    
class BrownNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        brown_noise = cn.powerlaw_psd_gaussian(2, len(y))
        a_brown = np.sqrt(brown_noise ** 2).max()
        augmented = (y + brown_noise * 1 / a_brown * a_noise).astype(y.dtype)
        return augmented
    

#https://www.kaggle.com/code/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english
#https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
class AddBackround(AudioTransform):
    def __init__(self, 
                 duration,
                 sr,
                 background_noise_paths,
                 always_apply=True, 
                 p=0.6, 
                 min_snr=1, 
                 max_snr=3,
                 ):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.back_pths = background_noise_paths
        self.background = load_sf(random.choice(self.back_pths))
        self.d_len = duration * sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        if random.random() < 0.2:
            self.background = load_sf(random.choice(self.back_pths))
        
        cropped_background = random_crop(self.background, self.d_len)

        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))  
        l_signal = len(y)

        a_background = np.sqrt(cropped_background ** 2).max()
        l_background = len(cropped_background)

        if l_signal > l_background:
            ratio = l_signal//l_background
            cropped_background = np.tile(cropped_background, ratio+1 )
            cropped_background = cropped_background[0:l_signal]

        if l_signal < l_background:    
            cropped_background = cropped_background[0:l_signal]

        augmented = (y + cropped_background * 1 / a_background * a_noise).astype(y.dtype)
        return augmented  
    
    
def spec_augment(spec: np.ndarray, 
                 num_mask=3, 
                 freq_masking_max_percentage=0.1,
                 time_masking_max_percentage=0.1, 
                 p=0.5):
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


class AbluTransforms():
    MEAN = (0.485, 0.456, 0.406) # RGB
    STD = (0.229, 0.224, 0.225) # RGB
    
    def __init__(self, audio):
        self.image_width = audio.IMAGE_WIDTH

        self.train = A.Compose([
                        A.CoarseDropout(max_holes=4, p=0.4),
                        A.PadIfNeeded(min_height=self.image_width, min_width=self.image_width),
                        A.CenterCrop(width=self.image_width, height=self.image_width), 
                        A.Normalize(self.MEAN, self.STD, max_pixel_value=1.0, always_apply=True),    
                        ])
        
        self.valid = A.Compose([
                        A.PadIfNeeded(min_height=self.image_width, min_width=self.image_width),
                        A.CenterCrop(width=self.image_width, height=self.image_width),  
                        A.Normalize(self.MEAN, self.STD, max_pixel_value=1.0,always_apply=True),
                        ])

 
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

            
def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def fold_image(arr, shape): 
    '''chop the image in half along the temporal dimension and stack to a square image
    Goal is to allow more pixels and segments in the temporal domain than frequency'''
    length = arr.shape[1]
    num_vertical = shape[0]
    cols = length//num_vertical
    remainder = length % num_vertical
    if num_vertical == 2:
        half0 = arr[:, :cols + remainder]   #added the .T v55
        half1 = arr[:, cols:]  #added the .T v53
        arr =  np.vstack((half0, half1))  #changed to h-stack v55
    elif num_vertical == 4:
        half0 = arr[:, :cols + remainder]
        half1 = arr[:, cols:]
        half2 = arr[:, cols:]
        half3 = arr[:, cols:]
        arr =  np.vstack((half0, half1, half2, half3))  #changed to h-stack v55
    return arr

# ## PyTorch Classes & Functions
# ### Dataset

class WaveformDataset(Dataset):
    def __init__(self, 
                 df, #This is the default dataframe with only human-labelled data
                 audio,
                 paths,
                 epoch=0,
                 pseudo_dfs=None,  #A list of dfs where human-labelled data is mixed with pseudo-labelled data
                 train=True, 
                 soundscape_path=None,
                 pseudo_dict = None,
                ): 
        self.epoch=epoch
        self.sr = audio.SR
        self.train = train
        self.human_labeled_df = df
        self.pseudo_dfs = pseudo_dfs
        if pseudo_dfs is not None:
            self.df_list = pseudo_dfs + [self.human_labeled_df]
            self.df = pseudo_dfs[0]
        else:
            self.df = df
        self.duration = audio.DURATION
        self.d_len = self.duration * self.sr
        self.soundscape_path=soundscape_path
        self.image_transform = AbluTransforms(audio).train if train else AbluTransforms(audio).valid
        self.back_pths = paths.background_noise_paths
        self.height = audio.N_MELS
        self.width = audio.CHUNK_WIDTH
        self.image_shape = audio.IMAGE_SHAPE
        self.num_chunks = self.image_shape[0] * self.image_shape[1]
        self.chunk_lenth = self.d_len // self.num_chunks
        self.prep_image = PrepareImage(height=audio.N_MELS, width = self.width)
        self.pseudo_dict = pseudo_dict
        print(f'The shape of the dataframe is currently {self.df.shape}')

        if self.train:
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            RandomNoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=1, max_snr=5),
                            PinkNoise(p=1, min_snr=1, max_snr=5),
                            BrownNoise(p=1, min_snr=1, max_snr=5),
                        ],
                        p=.25,
                    ),
                    AddBackround(self.duration, self.sr, self.back_pths, p=.25, min_snr=1.5, max_snr=3),
                    Normalize(p=1),
                ]
            )
        else:
            self.wave_transforms = Compose([Normalize(p=1)])
        
        if self.soundscape_path is not None:
            self.ss_array = self.open_audio_clip(soundscape_path)
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_df(self, df_idx=1):
        self.df = self.df_list[df_idx]
        
        
    def reset_wave_augmentation(self, epoch):
        if self.train and 4 <= epoch < 10:
            print(f'Using medium waveform augmentation on epoch {epoch}')
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            RandomNoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=1, max_snr=5),
                            PinkNoise(p=1, min_snr=1, max_snr=5),
                            BrownNoise(p=1, min_snr=1, max_snr=5),
                        ],
                        p=.15,
                    ),
                    AddBackround(self.duration, self.sr, self.back_pths, p=.15, min_snr=1.5, max_snr=3),  #Tried various SNR and p combinations.  Adding background noise just doesn't seem to do much.
                    Normalize(p=1),
                ]
            )
        elif self.train and epoch >= 10:
            print(f'Using minimal waveform augmentation on epoch {epoch}')
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            RandomNoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=1.5, max_snr=5),
                            PinkNoise(p=1, min_snr=1.5, max_snr=5),
                            BrownNoise(p=1, min_snr=1.5, max_snr=5),
                        ],
                        p=.1,
                    ),
                    Normalize(p=1),
                ]
            )

    def open_audio_clip(self, path, starts=None):    
        try:  
            y, _ = torchaudio.load(path)
            if y.ndim == 2 and y.shape[0] == 2:
                #print(f'converting {path} to mono')
                y = torch.mean(y, dim=0).unsqueeze(0)  # from stereo to mono
            y = y.squeeze().numpy() 
        except:
            y = np.random.randn(5*320000) 
            print(f'could not open {path}')
        
        if not np.isfinite(y).all():
            y[np.isnan(y)] = np.mean(y)
            y[np.isinf(y)] = np.mean(y)

        if self.soundscape_path is None:
            if len(y) > 0: 
                if starts is not None:
                    slices = []
                    chunk_length = self.d_len // self.num_chunks
                    for start in starts:
                        end = min(start + chunk_length, len(y))
                        slices.append(y[start:end])
                    y = np.concatenate(slices)
                else:
                    y = y[:self.d_len*4]  #so we don't get memory or speed issues from long clips?

            y = crop_or_pad(y, self.d_len, train=self.train, path=path)  # background_paths=self.back_pths
        y = self.wave_transforms(y, sr=self.sr)
        return y
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        '''For training open the path from the row filepath, 
        or for a soundscape we just need to crop out the 
        relevent chunk of the array opened in the init.'''
        row = self.df.iloc[idx]
        default_labels = [row[2:].values.tolist()]

        #print(f'getting item from this row: \n {row}')

        if self.soundscape_path is not None:
            start = self.sr * row.start
            stop = self.sr * row.stop
            y = self.ss_array[start: stop]  #arrays of 0 after the first iteration       
        else:
            wav_path = row.filepath 
            if self.pseudo_dict is not None:
                wav_fn = str(Path(wav_path).parent.name) + '/' + str(Path(wav_path).name)
                chunk_labels = self.pseudo_dict.get(wav_fn, default_labels)   #self.pseudo_dict[wav_fn]  #There is probably a better way to do this!!
                chunk_labels = chunk_labels #[:6*self.num_chunks] #in case of really really long samples
                samples_length = len(chunk_labels)
                num_samples = min(self.num_chunks, samples_length)
                indices = random.sample(range(samples_length), num_samples)
                sample_labels = [chunk_labels[i] for i in indices]
                label_array = np.array(sample_labels)
                pseudo_labels = reduce(np.bitwise_or, label_array)
                start_positions = [self.chunk_lenth * index for index in indices]
                y = self.open_audio_clip(wav_path, starts = start_positions)
            else:
                y = self.open_audio_clip(wav_path, starts = None)
        
        if audio.PCEN: 
            image = compute_pcen(y)
        else:
            image = compute_melspec(y)
        
        #note that the point of this is to normalise each chunk seperately, and re-assemble the image the way it already was
        normalised = []
        if self.num_chunks == 4:
            for i in range(4):
                sub_image = image[:, i * self.width : (i + 1) * self.width] #four quarters of the final image
                sub_image = self.prep_image.prep(image=sub_image)['image']
                normalised.append(sub_image)
            image = np.concatenate(normalised, axis=1)
        elif self.num_chunks == 2:
            for i in range(2):
                sub_image = image[:, i * self.width  : (i + 1) * self.width]  #stacking the two halfs of the final image
                sub_image = self.prep_image.prep(image=sub_image)['image']
                normalised.append(sub_image)
            image = np.concatenate(normalised, axis=1)
        else:
            image = self.prep_image.prep(image=image)['image']
        #from here it would be safest not to do any shuffling in the time direction, as chunks were normalised seperately

        if self.train and self.epoch <=8:
            image = spec_augment(image, 
                                 p=0.25, 
                                 num_mask=3,
                                 freq_masking_max_percentage=0.1,
                                 time_masking_max_percentage=0.1)
        
        if self.image_shape[0]==2:
            image = fold_image(image, shape=self.image_shape)
        
        image = mono_to_color(image, use_deltas=audio.USE_DELTAS)
        image = self.image_transform(image=image)['image']
        image = image.transpose(2,0,1).astype(np.float32) # swapping the image channels to the first axis


        if self.soundscape_path:
            targets = idx 
        elif self.pseudo_dict is not None:
            target_vals = pseudo_labels.astype(np.uint8)
            targets = torch.tensor(target_vals)
        else:
            target_vals = row[2:].values.astype(np.uint8)
            targets = torch.tensor(target_vals)
        return image, targets

# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):  #was 0.25, 2.0, also tried 0.6 and 0.4  best score of .66 reached with 0.5,  2.0  exp26, changed to 3 exp27 1.5
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        probas = torch.sigmoid(logits)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):  #changed from [1,1] to [1,2] version 33, no change.  [5,1], v34, none changed the score.
        super().__init__()

        self.focal = BCEFocalLoss()
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()
        loss = self.focal(input_, target)

        #my simplified version, using the segment logits directly instead of the interpolated function from original code
        segmentwise_logit, _ = input['segmentwise_logit'].max(dim=1) #also tried mean, but it didn't work for some reason
        aux_loss = self.focal(segmentwise_logit, target)   

        return self.weights[1] * loss + self.weights[1] * aux_loss


class LossFunctions():
    '''A wrapper class, that incudes various loss function types and takes a dictionary
    as an input with the various outputs from the model'''
    def __init__(self, loss_fn_nm):
        loss_dict = {
                'BCEFocal2WayLoss': BCEFocal2WayLoss(),
                'BCEFocalLoss': BCEFocalLoss(),
                'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
                'CrossEntropyLoss': nn.CrossEntropyLoss()
                }
        self.loss_fn_nm = loss_fn_nm
        self.loss_fn = loss_dict.get(loss_fn_nm, nn.CrossEntropyLoss())
        
    def loss(self, preds_dict, target):
        if self.loss_fn_nm == 'BCEFocal2WayLoss':     
            loss_val = self.loss_fn(preds_dict, target)  #'BCEFocal2WayLoss'
        else:   # ['BCEFocalLoss', 'BCELossWithLogits','CrossEntropyLoss']
            loss_val = self.loss_fn(preds_dict['logit'], target)
        return loss_val


class BirdSoundModel(pl.LightningModule):
    
    def init_layer(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.0)

    def init_weights(self, model):
        '''Currently not in use'''
        classname = model.__class__.__name__
        if classname.find("Conv2d") != -1:
            nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
            model.bias.data.fill_(0)
        elif classname.find("BatchNorm") != -1:
            model.weight.data.normal_(1.0, 0.02)
            model.bias.data.fill_(0)
        elif classname.find("GRU") != -1:
            for weight in model.parameters():
                if len(weight.size()) > 1:
                    nn.init.orghogonal_(weight.data)
        elif classname.find("Linear") != -1:
            model.weight.data.normal_(0, 0.01)
            model.bias.data.zero_()


    class AttentionBlock(nn.Module):
        def __init__(self,
                     in_features: int,
                     out_features: int,
                     activation="linear",
                     image_shape = (1,1),
        ):
            super().__init__()

            self.activation = activation
            self.attention = nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3, #was 1 originally, changed to 3 with good results
                stride=1,
                padding=1,  #was 0 originally, changed to 1 to match above
                bias=True)
            self.classify = nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3, #was 1
                stride=1,
                padding=1,  #was 0
                bias=True)
            self.init_weights()
            self.image_shape=image_shape
            self.chunks_high=self.image_shape[0]
            self.chunks_wide = self.image_shape[1]
            self.num_chunks = self.image_shape[0]*self.image_shape[1]

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
            # x: (batch_size, n_features, n_chunks * n_segments_per_chunk)
            
            # We can reshape to convolve only along the frequency dimension to operate on the time chunks independently. 
            # We don't need to do this for the logits, but keeping the same form in case we want to change the activation, 
            # or kernel size in a way that they are not independent of each other.
            
            batch_size = x.shape[0]  # Split along the third dimension
            split_length = x.shape[2] // self.num_chunks
            
            x = torch.split(x, split_length, dim=2)
            x = torch.cat(x, dim=0)  #  (128, 1280, 4)

            norm_att = torch.softmax(torch.tanh(self.attention(x)), dim=-1)/(16//split_length)  #so that they have a mean value of 1/16 each
            split_attn = torch.split(norm_att, batch_size, dim=0) #Put the weights back to their original shape
            norm_att = torch.cat(split_attn, dim=2)

            seg_logits = self.classify(x)
            classify = self.nonlinear_transform(seg_logits)  #note - this is OK, because we're just doing a sigmoid, would be
            
            split_logits = torch.split(seg_logits, batch_size, dim=0)
            seg_logits = torch.cat(split_logits, dim=2)

            split_classify = torch.split(classify, batch_size, dim=0)
            classify = torch.cat(split_classify, dim=2)
            
            preds = torch.sum(norm_att * classify, dim=-1)
            logit = torch.sum(norm_att * seg_logits, dim=-1)
            seg_logits = seg_logits.transpose(1, 2)
            return logit, seg_logits, preds
    

    def __init__(self,
                 cfg, 
                 audio,
                 paths,
                 in_channels=3,
                ):
        super().__init__()

        self.classes = cfg.classes
        self.num_classes = len(self.classes)
        self.bn0 = nn.BatchNorm2d(audio.IMAGE_WIDTH) #if cfg.RESHAPE_IMAGE else nn.BatchNorm2d(audio.N_MELS)
        self.base_model = timm.create_model(
                                    cfg.MODEL, 
                                    pretrained=True, 
                                    in_chans=in_channels
                                    )
        layers = list(self.base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(self.base_model, "fc"):
            in_features = self.base_model.fc.in_features
        elif cfg.MODEL == 'eca_nfnet_l0':
            in_features = self.base_model.head.fc.in_features
        elif cfg.MODEL == 'convnext_tiny.in12k_ft_in1k':
            in_features = self.base_model.head.fc.in_features
        else:
            in_features = self.base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.image_shape = audio.IMAGE_SHAPE
        self.att_block = self.AttentionBlock(in_features, 
                                            self.num_classes, 
                                            activation="sigmoid",
                                            image_shape=self.image_shape
                                            )
        self.loss_function = LossFunctions(cfg.LOSS_FUNCTION_NAME).loss
        self.init_weight()
        self.val_outputs = []
        self.train_outputs = []
        self.metrics_list = []
        self.val_epoch = 0
        self.epoch_to_unfreeze_backbone = cfg.EPOCHS_TO_UNFREEZE_BACKBONE,
        self.lr = cfg.LR
        self.initial_lr = cfg.INITIAL_LR
        self.min_lr = cfg.MIN_LR
        self.warmup_epochs = cfg.WARMUP_EPOCHS
        self.cycle_length = cfg.LR_CYCLE_LENGTH
        self.lr_decay = cfg.LR_DECAY
        self.printed_shapes = False
        self.use_data_updates = cfg.USE_UNLABELED_SOUNDSCAPES
        self.data_update_1 = cfg.FIRST_DATA_UPDATE
        self.data_update_2 = cfg.SECOND_DATA_UPDATE
        self.mixup_alpha = cfg.MIXUP_ALPHA
        self.temp_dir = Path(paths.temp_dir)
        self.results_dir = Path(paths.out_dir)
        
    
    def init_weight(self):
        self.init_bn(self.bn0)
        self.init_layer(self.fc1)

    def forward(self, input_data):
        x = input_data # (batch_size, 3, time_steps, mel_bins)  I think this is wrong, it's batch size, 3, melbins, time#This needs to match the the output of dataloader & getitem 

        x = x.transpose(1, 3) #(batch_size, mel_bins, time_steps, channels)
        x = self.bn0(x)
        x = x.transpose(1, 3)
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
        #For the (2,1) and (1,1) cases we donn't need to do anything here, there is only one chunk represented along the horizontal axis.

        x = torch.mean(x, dim=3) # Aggregate in time axis, but only over the duration of each chunk
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (logit, segment_logits, preds) = self.att_block(x) #  in: [64, 1280, 16], out: [64, 182], [64, 16, 182],  [64, 182]
        
        output_dict = {
            'clip_preds': preds,  #predictions for AP and CMAP calculation
            'segmentwise_logit': segment_logits,   #[64, 16, 182]  # doesn't use the attn or activation function
            'logit': logit,  #torch.Size([64, 182]) used for the loss calculation, includes attention   
        }

        return output_dict

    def configure_optimizers(self):
        def custom_lr_scheduler(epoch):
            '''CosineAnealingWarmRestarts but with a decay between cycles and a warmup'''
            initial = self.initial_lr / self.lr 
            rel_min = self.min_lr / self.lr
            step_size = (1-initial) / self.warmup_epochs
            warmup = initial + step_size * epoch if epoch <= self.warmup_epochs else 1
            cycle = epoch-self.warmup_epochs
            decay = 1 if epoch <= self.warmup_epochs else self.lr_decay ** (cycle // self.cycle_length)
            phase = np.pi * (cycle % self.cycle_length) / self.cycle_length
            cos_anneal = 1 if epoch <= self.warmup_epochs else  rel_min + (1 - rel_min) * (1 + np.cos(phase)) / 2
            return warmup * decay * cos_anneal #this value gets multipleid by the initial lr (self.lr)
        
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_scheduler)
        return [optimizer], [scheduler]
    
    def train_with_mixup(self, X, y):
        X, y_a, y_b, lam = mixup_data(X, y, self.mixup_alpha, self.device)
        preds_dict = self(X)
        loss_mixup = mixup_criterion(self.loss_function, preds_dict, y_a, y_b, lam)
        return loss_mixup, preds_dict

    def training_step(self, batch, batch_idx):
        image, target = batch  
        if cfg.USE_MIXUP:
            loss, preds_dict = self.train_with_mixup(image, target)
        else:
            preds_dict = self(image)
            loss = self.loss_function(preds_dict,target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        train_output = {"train_loss": loss, "logits": preds_dict['clip_preds'], "targets": target}
        self.train_outputs.append(train_output) 
        return loss        

    def validation_step(self, batch, batch_idx):
        image, target = batch 
        preds_dict = self(image)  #Should the loss definately be based on framewise output?
        if not self.printed_shapes:
            print(f'logit', self(image)['logit'].shape)                        #[64, 79]  Batch,num_classes
            print(f'segmentwise_logit shape', self(image)['segmentwise_logit'].shape)    # [64, 8, 79]  
            self.printed_shapes = True
        val_loss = self.loss_function(preds_dict, target)  #Why the need to convert to float here
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        output = {"val_loss": val_loss, "logits": preds_dict['logit'], "targets": target}
        self.val_outputs.append(output)  #new line
        return output

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader
    
    def on_train_epoch_start(self):
        epoch = self.current_epoch
        train_loader = self.trainer.train_dataloader
        train_loader.dataset.reset_wave_augmentation(epoch)
        train_loader.dataset.set_epoch(epoch)

    def on_train_epoch_end(self, *args, **kwargs):  
        epoch = self.current_epoch
        if epoch == self.epoch_to_unfreeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = True
            print(f'Unfreezing the backbone after {epoch} epochs')

        if self.use_data_updates:   
            if (epoch + 1) == self.data_update_1:
                train_loader = self.trainer.train_dataloader
                train_loader.dataset.update_df(df_idx=1)
                print(f'Using the second dataset from epoch {epoch+1}')
            elif (epoch + 1) == self.data_update_2:
                train_loader = self.trainer.train_dataloader
                train_loader.dataset.update_df(df_idx=2)
                print(f'Using the third dataset from epoch {epoch+1}')

        
    def on_validation_epoch_end(self):
        val_outputs = self.val_outputs
        avg_val_loss = torch.stack([x['val_loss'] for x in val_outputs]).mean().cpu().detach().numpy()
        output_val_logits = torch.cat([x['logits'] for x in val_outputs],dim=0)
        val_targets = torch.cat([x['targets'] for x in val_outputs],dim=0).cpu().detach().numpy() 
        
        train_outputs = self.train_outputs
        if train_outputs:
            train_losses = [x['train_loss'].cpu().detach().numpy() for x in train_outputs]
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
            output_train_logits = torch.cat([x['logits'] for x in train_outputs],dim=0)
            train_targets = torch.cat([x['targets'] for x in train_outputs],dim=0).cpu().detach().numpy()
        else: 
            avg_train_loss = avg_val_loss #we need this because the first time it's an empty list
            output_train_logits = torch.ones(1,output_val_logits.shape[1])
            train_targets = torch.zeros(1, output_val_logits.shape[1])
        
        val_probs = output_val_logits.sigmoid().cpu().detach().numpy()
        train_probs = output_train_logits.sigmoid().cpu().detach().numpy()
  
        val_pred_df = pd.DataFrame(val_probs, columns = self.classes)
        val_target_df = pd.DataFrame(val_targets, columns = self.classes)
        train_pred_df = pd.DataFrame(train_probs, columns = self.classes)
        train_target_df = pd.DataFrame(train_targets, columns = self.classes)
       
        avg_score = padded_cmap(val_target_df, val_pred_df, padding_factor = 5)  #Val CMAP5
        avg_score3 = skm.label_ranking_average_precision_score(val_targets,val_probs)  #Val LRAP
        optimizer_lr = self.trainer.optimizers[0].param_groups[0]['lr']  
        print(f"Learning rate from optimiser at epoch {self.current_epoch}: {optimizer_lr}")

        if self.val_epoch > 0:
            avg_score4 = padded_cmap(train_target_df, train_pred_df, padding_factor = 5)
            self.metrics_list.append({'train_loss':avg_train_loss, 
                                      'val_loss': avg_val_loss, 
                                      'val_map' : avg_score,    #Val CMAP5
                                      'val_prec':avg_score3,    #Val LRAP
                                      'train_map':avg_score4,  #Train CMAP5
                                      'learning_rate':optimizer_lr
                                     })

        print(f'Epoch {self.current_epoch} train loss {avg_train_loss}')
        print(f'Epoch {self.current_epoch} validation loss {avg_val_loss}')
        print(f'Epoch {self.current_epoch} validation C-MAP score pad 5 {avg_score}')
        print(f'Epoch {self.current_epoch} validation AP score {avg_score3 or None}')
        
        val_target_df.to_pickle(self.results_dir / 'val_target_df.pkl')
        val_pred_df.to_pickle(self.results_dir / 'val_pred_df.pkl') 
        train_target_df.to_pickle(self.results_dir / 'train_target_df.pkl')
        train_pred_df.to_pickle(self.results_dir / 'train_pred_df.pkl')  

        self.val_outputs = []
        self.train_outputs = []
        self.val_epoch +=1

        return
    
    def get_my_metrics_list(self):
        return self.metrics_list


def get_model(ckpt_path):
    model = BirdSoundModel(cfg, audio, paths)
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict['state_dict'])  
    model.to(cfg.DEVICE)
    return model


def save_models(paths):
    '''This is overkill, but I imagine wanting to modify to pickle 
    the whole model instead of just the checkpoints'''
    checkpoints = [path for path in Path(paths.chkpt_dir).glob('*.ckpt')]
    
    for ckpt_path in tqdm(checkpoints):
        model = get_model(ckpt_path)
        save_path = str(Path(paths.out_dir) / ckpt_path.name)
        torch.save(model.state_dict(), save_path)
        print(Blue.S + 'Weights checkpoint saved to: ' + Blue.E + save_path)

    return save_path  #just returns what ever came last, to check for functionality


def get_class_weights(df):
    df = df.iloc[:, 2:] # removing the 'filepath' and 'primary_label' columns
    col_sums = df.sum()
    counts_array = col_sums.values
    counts_array = np.sqrt(50 + counts_array) 
    class_weights = counts_array.tolist()
    sample_idxs = np.argmax(df.values, axis=1).tolist()
    sampling_weights = [1 / class_weights[idx] for idx in sample_idxs] 
    return sampling_weights


def get_dataloaders(df_train, df_valid, audio, pseudo_dfs=None, pseudo_dict=None, weighted_sampling=False):

    ds_train = WaveformDataset(df_train, 
                               audio,
                               paths,
                               pseudo_dfs=pseudo_dfs, 
                               train=True, 
                               pseudo_dict=pseudo_dict)
    ds_val = WaveformDataset(df_valid, 
                             audio,
                             paths,
                             pseudo_dfs=None,
                             train=False, 
                             pseudo_dict=pseudo_dict)
    
    if weighted_sampling is not None:  #Not implemented
        sample_weights = get_class_weights(df_train)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(ds_train))
        dl_train = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE , sampler=sampler, num_workers=cfg.NUM_WORKERS, )   #persistent_workers=True
    else:
        dl_train = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE , shuffle=True, num_workers=cfg.NUM_WORKERS,)    #persistent_workers=True
    dl_val = DataLoader(ds_val, batch_size=cfg.BATCH_SIZE, num_workers = cfg.NUM_WORKERS)   

    return dl_train, dl_val, ds_train, ds_val


def run_training(dl_train, dl_val, cfg, audio, checkpoint_dir):
    pl.seed_everything(cfg.SEED, workers=True)
    torch.set_flush_denormal(True)
    torch.set_float32_matmul_precision('medium')  
    print(f"Running training...")
    logger = None
    audio_model = BirdSoundModel(cfg, audio, paths)
    early_stop_callback = EarlyStopping(monitor="val_loss", 
                                        min_delta=cfg.MIN_DELTA, 
                                        patience=cfg.PATIENCE, 
                                        verbose= True, 
                                        mode="min")
    
    # saves top- checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(save_top_k=cfg.KEEP_LAST,
                                          monitor="val_loss",
                                          mode="min",
                                          dirpath=checkpoint_dir,
                                          save_last= True,
                                          save_weights_only=True, 
                                          verbose= True,
                                          # filename=f'birdCLEF23-{epoch:02d}-{val_loss:.4f}', need to figure this out so It can update to a dataset
                                          )

    callbacks_to_use = [checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer(
        val_check_interval=0.5,
        deterministic=True,
        max_epochs=cfg.EPOCHS,
        logger=logger,
        callbacks=callbacks_to_use,
        precision=cfg.PRECISION, 
        accelerator=cfg.GPU,
        reload_dataloaders_every_n_epochs=1
        )

    print("Running trainer.fit")
    trainer.fit(audio_model, train_dataloaders=dl_train, val_dataloaders=dl_val)       
    gc.collect()
    torch.cuda.empty_cache()

    return audio_model.get_my_metrics_list()


def extract_results(metrics, paths):
    train_losses = [x['train_loss'] for x in metrics]
    val_losses = [x['val_loss'] for x in metrics]  
    train_map = [x['train_map'] for x in metrics]  
    val_map = [x['val_map'] for x in metrics]  
    val_precision = [x['val_prec'] for x in metrics] 
    learning_rates =  [x['learning_rate'] for x in metrics]
    time_axis = [0.5*x for x in range(1,len(val_losses)+1)]

    fig, ax = plt.subplots()
    plt.plot(time_axis, train_losses, 'r', label='Train Loss')
    plt.plot(time_axis, val_losses, '--k', label='Val Loss')
    plt.legend()
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.tick_params('both', colors='r')

    ax2 = ax.twinx()
    plt.plot(time_axis, train_map, 'b', label='Train cmap5')
    plt.plot(time_axis, val_precision, '--g', label='Val LRAP')
    plt.plot(time_axis, val_map, ':m', label='Val cmap5')
    ax2.set_ylabel('Accuracy')
    plt.legend()
    plt.legend(loc='lower left')
    ax.tick_params('both', colors='b')
    plt.savefig(Path(paths.out_dir) / f"exp_{use_case['experiment']}_training_metrics.jpg")
    plt.close()

    pred = pd.read_pickle(Path(paths.out_dir) / f'val_pred_df.pkl')
    true = pd.read_pickle(Path(paths.out_dir) / f'val_target_df.pkl')

    print(f' The final Val CMAP score is {padded_cmap(true, pred, padding_factor = 5)}')

    plt.plot(time_axis[::2], learning_rates[::2], marker='o')  #no need to plot the half-epoch rates
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig(Path(paths.out_dir) / 'learning_rate.jpg')
    plt.close()


############################################# Prepare Data  ######################################
##################################################################################################

cfg = DefaultConfig(options=use_case)
audio = DefaultAudio(options=use_case)
paths = FilePaths(options=use_case)


bg_paths = [path for path in Path(paths.BACKGROUND_NOISE_FLDR).rglob('*') if path.suffix in {'.ogg', '.flac'}]
for path in bg_paths[:3]:
    print(path)
    y, sample_rate = torchaudio.load(path)
    y = y.squeeze().numpy() 
    print(f'The background path length is {len(y)//32000} seconds')



if cfg.USE_UNLABELED_SOUNDSCAPES:
    pseudo_dict = get_pseudos(cfg.TRAIN_PSEUDOLABELS_PATH)
    sscape_pseudo_dict_1 = get_pseudos(cfg.PSEUDOLABELS_1)
    sscape_pseudo_dict_2 = get_pseudos(cfg.PSEUDOLABELS_2)


if cfg.USE_EXTRA_TRAIN_AUDIO:
    pseudo_dict_extra = get_pseudos(cfg.EXTRA_TRAIN_LABELS)
    pseudo_dict = {**pseudo_dict, **pseudo_dict_extra}



Path(paths.chkpt_dir).mkdir(parents=True, exist_ok=True)
Path(paths.best_weights_dir).mkdir(parents=True, exist_ok=True)
Path(paths.out_dir).mkdir(parents=True, exist_ok=True)
Path(paths.temp_dir).mkdir(parents=True, exist_ok=True)
print(Blue.S + f'Training with: ' + Blue.E + cfg.GPU)
print(Blue.S + 'CPUs for available for dataloading: ' + Blue.E + str(cfg.NUM_WORKERS))


use_cols = ['filename', 'primary_label', 'secondary_labels']  #'secondary_labels'  'latitude', 'longitude'
in_df = pd.read_csv(paths.LABELS_PATH,  usecols=use_cols)
in_df['filepath'] = paths.TRAIN_AUDIO_DIR + '/' + in_df['filename']

print(f'There are {len(in_df)} audio samples in the labels original dataframe')

print(in_df['primary_label'].nunique())

in_df = in_df[~in_df['primary_label'].isin(cfg.EXCLUDED_CLASSES)]


print(len(in_df))
unique_birds = sorted(in_df['primary_label'].unique()) 

print(len(in_df))
unique_birds = sorted(in_df['primary_label'].unique())   
cfg.classes = unique_birds
cfg.num_classes = len(cfg.classes)

print(f'Training + Validation with {in_df.shape[0]} audio samples')
print(f'There are {cfg.num_classes} primary class labels')

def remove_unused_birds(second_bird_list):
    return [string for string in second_bird_list if string in cfg.classes]
if cfg.USE_SECONDARY:
    in_df['secondary_labels'] = in_df['secondary_labels'].apply(ast.literal_eval)
    in_df['secondary_labels'] = in_df['secondary_labels'].apply(remove_unused_birds)


# ### Add in any extra training data for rare classes
# If any additional data is avalaible, not in the main dataframes, add it only to rare classes, to try and reduce class imbalance

value_counts = in_df['primary_label'].value_counts()
uncommon_classes = value_counts[value_counts < cfg.USE_EXTRA_DATA_THRESHOLD].index.tolist()


if cfg.USE_EXTRA_TRAIN_AUDIO:
    use_cols = ['filename', 'primary_label', 'secondary_labels', 'latitude', 'longitude']
    extra_df = pd.read_csv(cfg.EXTRA_TRAIN_LABELS,  usecols=use_cols)
    extra_df = extra_df.drop_duplicates(subset=['filename'], keep='first')  #because I'm getting it from the pseudos df, with 5 sec chunks for each file
    extra_df = balance_primary_label(extra_df, max_count=100)
    extra_df = extra_df[extra_df['primary_label'].isin(uncommon_classes)]
    extra_df['secondary_labels'] = [[] for _ in extra_df['secondary_labels']] 
    extra_df['filepath'] = paths.EXTRA_TRAIN_AUDIO + '/' + extra_df['filename']
    print(f'There are {len(extra_df)} audio samples in the extra labels dataframe')
    extra_df=extra_df[['primary_label',  'secondary_labels', 'latitude', 'longitude', 'filepath']]
    extra_df.head()
    in_df = in_df[['primary_label', 'secondary_labels', 'latitude', 'longitude', 'filepath']]
    in_df = pd.concat([in_df, extra_df], ignore_index=True).sort_values(by='primary_label').reset_index(drop=True)


print(in_df.shape)

print(in_df.head(2))


# Now let's double-check that all the training samples in the dataframe actually exist.  Remove any rows that can't be found.

original_length = len(in_df)
training_samples = set([path for path in Path(paths.TRAIN_AUDIO_DIR).rglob('*') if path.suffix in {'.ogg', '.flac'}])
if cfg.USE_EXTRA_TRAIN_AUDIO:
    extra_samples = set([path for path in Path(paths.EXTRA_TRAIN_AUDIO).rglob('*') if path.suffix in {'.ogg', '.flac'}])
    training_samples = training_samples + extra_samples

in_df['filepath'] = in_df['filepath'].apply(Path)
in_df = in_df[in_df['filepath'].isin(training_samples)]
in_df['filepath'] = in_df['filepath'].apply(str)
new_length = len(in_df)

print(Blue.S + 'The original length was: ' +Blue.E, original_length)
print(Blue.S + 'The final length was: ' +Blue.E, new_length)
if original_length > new_length:
    print(Blue.S + 'Samples removed: ' + Blue.E, original_length - new_length)

max_rows_per_class = cfg.MAX_PER_CLASS
class_counts = in_df['primary_label'].value_counts()
classes_to_reduce = class_counts[class_counts > max_rows_per_class].index
def downsample_class(in_df, class_label, max_rows):
    df_class = in_df[in_df['primary_label'] == class_label]
    return df_class.sample(n=max_rows)
df_list = [downsample_class(in_df, class_label, max_rows_per_class) if class_label in classes_to_reduce 
           else in_df[in_df['primary_label'] == class_label]
           for class_label in in_df['primary_label'].unique()]
in_df = pd.concat(df_list)
print(in_df.head())


# Temporarily drop any super rare classes from the dataframe, so they don't end up loosing precious samples from training due to location or splitting.

mask = in_df['primary_label'].map(in_df['primary_label'].value_counts()) > cfg.RARE_THRESHOLD
common_df = in_df[mask]
mask = in_df['primary_label'].map(in_df['primary_label'].value_counts()) <= cfg.RARE_THRESHOLD
rare_df = in_df[mask]
print(rare_df.primary_label.value_counts())

print(common_df.primary_label.value_counts().tail(10))


rare_df = rare_df.reset_index(drop=True)
value_counts = rare_df['primary_label'].value_counts()
duplication_needed = {label: cfg.RARE_THRESHOLD - count for label, count in value_counts.items()}


duplicated_rows = []
for label, count in duplication_needed.items():
    label_rows = rare_df[rare_df['primary_label'] == label]
    num_duplicates = count // len(label_rows)  # Number of full duplications needed
    remainder = count % len(label_rows)        # Remaining duplications needed

    if num_duplicates > 0:
        duplicated_full_sets = pd.concat([label_rows] * num_duplicates, ignore_index=True)
        duplicated_rows.append(duplicated_full_sets)

    if remainder > 0:
        duplicated_remainder = label_rows.sample(n=remainder, replace=True)
        duplicated_rows.append(duplicated_remainder)

rare_df = pd.concat([rare_df] + duplicated_rows, ignore_index=True)
final_counts = rare_df['primary_label'].value_counts()
print(rare_df.shape)


print(final_counts[-10:])

# We might also want to filter by location.

all_common_df = common_df.copy()
if cfg.FINETUNE_BY_LOCATION:
    common_df = subset_by_location(common_df)
    common_df.shape

print(common_df['primary_label'].value_counts())

rare_df['primary_label'].value_counts()

# Now we do our train/val split on the common samples

skf =StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
target = common_df['primary_label'] 

for train_index, val_index in skf.split(common_df, target):
    tn_df, val_df = common_df.iloc[train_index].copy(), common_df.iloc[val_index].copy()
train_df = pd.concat([tn_df, rare_df])
    
print(f'The training dataframe has {train_df.shape[0]} rows\n'
      f'The validation dataframe has {val_df.shape[0]} rows')


train_df.head(3)

val_filepaths = list(val_df['filepath'])
all_locations_train_df  = all_common_df[~ all_common_df['filepath'].isin(val_filepaths)]


if cfg.USE_UNLABELED_SOUNDSCAPES:
    use_cols = ['primary_label', 'secondary_labels', 'filename']
    ss_df_1 = pd.read_csv(cfg.PSEUDOLABELS_1,  usecols=use_cols)
    ss_df_1['filepath'] = cfg.SOUNDSCAPES_1 + '/' + ss_df_1['filename']
    ss_df_1['secondary_labels'] = ss_df_1['secondary_labels'].apply(literal_eval)
    ss_df_1 = ss_df_1.drop_duplicates(subset='filepath', keep='first')
    ss_df_1['latitude'] = 10.2
    ss_df_1['longitude'] = 77
    ss_df_1 = ss_df_1[['primary_label', 'secondary_labels', 'latitude', 'longitude', 'filepath']]
    
    ss_df_2 = pd.read_csv(cfg.PSEUDOLABELS_2,  usecols=use_cols)
    ss_df_2['filepath'] = cfg.SOUNDSCAPES_2 + '/' + ss_df_2['filename']
    ss_df_2['secondary_labels'] = ss_df_2['secondary_labels'].apply(literal_eval)
    ss_df_2 = ss_df_2.drop_duplicates(subset='filepath', keep='first')
    ss_df_2['latitude'] = 10.2
    ss_df_2['longitude'] = 77
    ss_df_2 = ss_df_2[['primary_label', 'secondary_labels', 'latitude', 'longitude', 'filepath']]
    
    print(ss_df_1.head())
    print(ss_df_1.shape)

if cfg.USE_UNLABELED_SOUNDSCAPES:
    ss_df_1 = balance_primary_label(ss_df_1, max_count=cfg.PSEUDOLABELS_CLASS_LIMIT_1)
    ss_df_2 = balance_primary_label(ss_df_2, max_count=cfg.PSEUDOLABELS_CLASS_LIMIT_2)
    print(ss_df_1.shape)
    print(ss_df_2.shape)

if cfg.USE_UNLABELED_SOUNDSCAPES:
    ss_df_2.reset_index()
    ss_df_2.head()

# Let's also add a few pseudo_labeled samples into the validation dataset if we're using those.

if cfg.USE_UNLABELED_SOUNDSCAPES:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg.SEED)
    target = ss_df_2['primary_label']

    for train_index, val_index in skf.split(ss_df_2, target):
        ss_df_2_train_df = ss_df_2.iloc[train_index].copy()
        ss_df_2_val_df = ss_df_2.iloc[val_index].copy()

if cfg.USE_UNLABELED_SOUNDSCAPES:
    train_df_1 = pd.concat([all_locations_train_df.copy(), ss_df_1], ignore_index=True)
    train_df_2 = pd.concat([all_locations_train_df.copy(), ss_df_2_train_df], ignore_index=True)
    #val_df = pd.concat([val_df.copy(), ss_df_2_val_df], ignore_index=True)
    pseudo_dict = {**pseudo_dict, **sscape_pseudo_dict_1, **sscape_pseudo_dict_2}
    print(train_df_1.shape)
    print(train_df_2.shape)
else:
    pseudo_dict = None

print(train_df.shape)
print(val_df.shape)



#Let's save out the naming scheme, with only the relevent names, and some counts for future analysis.
train_counts = train_df['primary_label'].value_counts().reset_index()
val_counts = val_df['primary_label'].value_counts().reset_index()
train_counts.columns = ['eBird', 'TrainSamples']
val_counts.columns = ['eBird', 'ValSamples']
use_cols = ['eBird', 'CommonName', 'ScientificName', 'ExtraName']
name_map_df = pd.read_csv(paths.bird_names_map, usecols=use_cols)

mapped_birds = set(list(name_map_df['eBird'].unique()))
train_birds = set(list(train_df['primary_label'].unique()))

missing_from_mapping = train_birds - mapped_birds

if len(missing_from_mapping) != 0:
    print(f'The following eBirds are missing from the mapping CSV: {missing_from_mapping}')

name_map_df = name_map_df[name_map_df['eBird'].isin(cfg.classes)]
name_map_df = name_map_df.drop_duplicates(subset='eBird', keep='first')
name_map_df = name_map_df.merge(train_counts, on='eBird', how='left')
name_map_df = name_map_df.merge(val_counts, on='eBird', how='left')
name_map_df['ValSamples'] = name_map_df['ValSamples'].fillna(0)
name_map_df = name_map_df.sort_values(by='eBird')
print(f"In total there are {name_map_df['eBird'].nunique()} unique primary labels saved to the naming csv.")
print(f"In total there are {train_df['primary_label'].nunique()} unique primary labels in the training csv.")
print(f'There are {len(list(set(cfg.classes)))} unique classes in the classes attribure of cfg')
name_map_df.to_csv(paths.bird_map_for_model, index=False)





def multi_binarize(df, unique_birds):
    keep_cols = ['primary_label', 'filepath']
    mlb = MultiLabelBinarizer(classes=unique_birds)
    df['combined_labels'] = df.apply(lambda row: [row['primary_label']] + row['secondary_labels'], axis=1)
    label_matrix = mlb.fit_transform(df['combined_labels']).astype('uint8')
    label_df = pd.DataFrame(label_matrix, columns=unique_birds)
    df = df.reset_index(drop=True)
    df = pd.concat([df[keep_cols], label_df], axis=1)
    return df

if not cfg.USE_SECONDARY:
    #This isn't quite right any more, has no allowance for pseudolabels
    train_df_0 = pd.concat([train_df, pd.get_dummies(train_df['primary_label']).astype('uint8')], axis=1)
    val_df = pd.concat([val_df, pd.get_dummies(val_df['primary_label']).astype('uint8')], axis=1)
    missing_birds = list(set(unique_birds).difference(list(val_df.primary_label.unique())))
    non_missing_birds = list(set(list(train_df.primary_label.unique())).difference(missing_birds))
    val_df = pd.concat([val_df, pd.DataFrame(0, index=val_df.index, columns=missing_birds)], axis=1)
    val_df = val_df[train_df_0.columns] # To synchronise the column order

else:
    train_df_0 = multi_binarize(train_df, unique_birds)
    val_df =  multi_binarize(val_df, unique_birds)
    if cfg.USE_UNLABELED_SOUNDSCAPES:
        train_df_1 = multi_binarize(train_df_1, unique_birds)
        train_df_2 = multi_binarize(train_df_2, unique_birds)

# Sanity & Curiosity Checks
# Check for non-numeric values in the data

print(train_df_0.iloc[:, 2:].head(3))

df_numeric = train_df_0.iloc[:, 2:].values
non_uint8_values = df_numeric[df_numeric.dtype != np.uint8]
print("Non-uint8 values:", non_uint8_values)

pseudo_dfs=[train_df_1, train_df_2] if cfg.USE_UNLABELED_SOUNDSCAPES else None

dl_train, dl_val, ds_train, ds_val = get_dataloaders(train_df_0, 
                                                          val_df, 
                                                          audio,
                                                          pseudo_dfs=pseudo_dfs,
                                                          pseudo_dict=pseudo_dict,
                                                          weighted_sampling=cfg.WEIGHTED_SAMPLING)

img, target = ds_train[0]

if (__name__ == '__main__') and (cfg.TRAIN):
    metrics = run_training(dl_train, dl_val, cfg, audio, paths.chkpt_dir)  #, df_updater
    extract_results(metrics, paths)
    last_path = save_models(paths)
    # Now checking it loads OK, as this is the weights file to be used in the inference notebook
    final_model = BirdSoundModel(cfg, audio, paths)
    model_state_dict = torch.load(last_path)
    final_model.load_state_dict(model_state_dict)  
    # Something funny is going on here, why do I need to use a dict key for the PL checkpoints, but not here, or in my other code?