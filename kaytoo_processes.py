
import random
import numpy as np
import albumentations as A
import torch
from torchaudio.functional import compute_deltas
import librosa


def compute_pcen(y, audio_cfg):
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.zeros_like(y)
        y[np.isinf(y)] = np.max(y)
    
    melspec = librosa.feature.melspectrogram(y=y, 
                                             sr=audio_cfg.SR, 
                                             n_mels=audio_cfg.N_MELS, 
                                             n_fft= audio_cfg.N_FFT, 
                                             fmin=audio_cfg.FMIN, 
                                             fmax=audio_cfg.FMAX
                                            )
    pcen = librosa.pcen(melspec, 
                        sr=audio_cfg.SR, 
                        gain=0.98, 
                        bias=2, 
                        power=0.5, 
                        time_constant=0.4, 
                        eps=0.000001
                       )
    return pcen.astype(np.float32)


def compute_melspec(y, audio_cfg):
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.zeros_like(y)
        y[np.isinf(y)] = np.max(y)
    
    melspec = librosa.feature.melspectrogram(y=y, 
                                             sr=audio_cfg.SR, 
                                             n_mels=audio_cfg.N_MELS, 
                                             n_fft=audio_cfg.N_FFT, 
                                             hop_length = audio_cfg.HOP_LENGTH, 
                                             fmin=audio_cfg.FMIN, 
                                             fmax=audio_cfg.FMAX
                                            ) 
    S_db = librosa.power_to_db(melspec, top_db=80)

    eps = 1e-6
    S_min, S_max = S_db.min(), S_db.max()
    S_scaled = (S_db - S_min) / (S_max - S_min + eps)
    return S_scaled


def mono_to_color_old(X, eps=1e-6, use_deltas=False):
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

def mono_to_color(X, eps=1e-6, use_deltas=False):
    """
    Convert a mono spectrogram to a 3-channel image, optionally including delta and delta-delta channels.
    Each channel is scaled independently to [0,1].
    """
    if use_deltas:
        T = torch.tensor(X, dtype=torch.float32)
        delta = compute_deltas(T)
        delta_2 = compute_deltas(delta)
        channels = [X, delta.numpy(), delta_2.numpy()]
    else:
        channels = [X, X, X]

    # Scale each channel independently
    scaled_channels = []
    for ch in channels:
        _min, _max = ch.min(), ch.max()
        if (_max - _min) > eps:
            ch_scaled = (ch - _min) / (_max - _min)
        else:
            ch_scaled = np.zeros_like(ch, dtype=np.float32)
        scaled_channels.append(ch_scaled.astype(np.float32))

    # Stack channels last
    X_color = np.stack(scaled_channels, axis=-1)
    return X_color


class PrepareSpec():
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.prep = A.Compose([
            A.PadIfNeeded(min_height=self.height, min_width=self.width),
            A.CenterCrop(width=self.width, height=self.height)
        ])


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


def fold_image(arr, num_vertical):
    """Split arr into num_vertical equal-ish vertical slices and
    stack them vertically."""
    length = arr.shape[1]
    cols_per_slice = length // num_vertical
    remainder = length % num_vertical

    slices = []
    start = 0
    
    for i in range(num_vertical):
        # Distribute the remainder across the first slices
        extra = 1 if i < remainder else 0
        end = start + cols_per_slice + extra
        slices.append(arr[:, start:end])
        start = end

    return np.vstack(slices)


def wav_to_bw_image(y,
                    audio_cfg,
                    preparer,
                    epoch=0,
                    train=False):

    use_pcen = audio_cfg.PCEN
    width = audio_cfg.SPEC_WIDTH
    shape = audio_cfg.IMAGE_SHAPE

    if use_pcen: 
        spec = compute_pcen(y, audio_cfg)
    else:
        spec = compute_melspec(y, audio_cfg)
    
    #note that the point of this is to normalise each chunk seperately, and re-assemble the image the way it already was
    normalised = []
    if shape == (2,2):
        for i in range(4):
            part_spec = spec[:, i * width : (i + 1) * width] #four quarters of the final image
            part_image = preparer.prep(image=part_spec)['image']
            normalised.append(part_image)
        image = np.concatenate(normalised, axis=1) #gets folded vertically into 4 later
    elif shape == (1,2):
        for i in range(2):
            part_spec = spec[:, i * width  : (i + 1) * width]  #stacking the two halfs of the final image
            part_image = preparer.prep(image=part_image)['image']
            normalised.append(part_image)
        image = np.concatenate(normalised, axis=1)
    else: #The cases of just one sound chunk per image (1,1) & (2, 0.5)
        image = preparer.prep(image=spec)['image']
        #### This will ensure the image is n_mels x spec_width

    #print(f'Inside wav_to_bw_image the min and max are {image.min()}, and {image.max()}')

    #from here it would be safest not to do any shuffling in the time direction, as chunks were normalised seperately

    if train and epoch <=8:
        image = spec_augment(image, 
                            p=0.25, 
                            num_mask=3,
                            freq_masking_max_percentage=0.1,
                            time_masking_max_percentage=0.1)
    
    return image 