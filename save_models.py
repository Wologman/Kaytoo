use_case = {
                'experiment' : 2,
                'project_root': '/home/olly/Desktop/Kaytoo',
                'num_cores': 6, 
                'run_training' : True,
                'epochs' : 20,
                }

from pathlib import Path
from tqdm.notebook import tqdm
import torch
import pandas as pd

from kaytoo_train_2_03 import (TrainingParameters,
                               AudioConfig,
                               BirdData,
                               FilePaths,
                               )

from kaytoo_sed_model import BirdSoundModel

def get_model(ckpt_path, backbone_name, image_shape, num_classes):
    model = BirdSoundModel(backbone_name,
                           num_classes,
                           image_shape)
    
    available = torch.cuda.is_available()
    device = torch.device(("cuda" if available else "cpu"))
    # load raw state_dict
    state_dict = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_model_from_checkpoints(ckpt_path, backbone_name, image_shape, num_classes):
    """Load a Lightning checkpoint and strip the wrapper namespace."""
    model = BirdSoundModel(backbone_name,
                           num_classes,
                           image_shape)
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if "state_dict" in checkpoint:              # Lightning file
        bird_state = checkpoint["state_dict"]
        if any(k.startswith("model.") for k in bird_state):
            bird_state = {k.replace("model.", "", 1): v for k, v in bird_state.items()}
        # else: already bare weights, keep as-is
    else:             # already a plain state_dict
        bird_state = checkpoint['state_dict']

    model.load_state_dict(bird_state, strict=True)
    available = torch.cuda.is_available()
    device = torch.device(("cuda" if available else "cpu"))
    model.to(device)
    model.eval()
    return model


def save_models(paths, backbone_name, image_shape, num_classes, deploy_ckpt_selection=1):
    '''This is overkill, but I imagine wanting to modify to pickle 
    the whole model instead of just the checkpoints'''
    checkpoints = [path for path in Path(paths.chkpt_dir).glob('*.ckpt')]
    latest_ckpt_first = sorted(checkpoints, key=lambda p: p.stat().st_ctime, reverse=True)
    selection_idx = max(deploy_ckpt_selection, 2)  #Hard coding the second last checkpoint to deploy.
    
    for idx, ckpt_path in tqdm(enumerate(latest_ckpt_first)) :
        model = load_model_from_checkpoints(ckpt_path, backbone_name, image_shape, num_classes)
        save_path = str(Path(paths.out_dir) / (ckpt_path.stem + '.pt'))
        deploy_path = str(Path(paths.model_deploy) / (ckpt_path.stem + '.pt'))
        torch.save(model.state_dict(), save_path)
        if idx == selection_idx:
            torch.save(model.state_dict(), deploy_path)
        print('Weights checkpoint saved to: ', save_path)

    return save_path  #just returns what ever came last, to check for functionality


train_cfg = TrainingParameters(options=use_case)
data_cfg = BirdData()
audio_cfg = AudioConfig()
paths = FilePaths(options=use_case)
bird_df = pd.read_csv('/home/olly/Desktop/Kaytoo/data/experiments/exp_3/exp_3_deploy/exp_3_bird_map.csv')

ckpt = torch.load("/home/olly/Desktop/Kaytoo/data/experiments/exp_3/temp/checkpoints/epoch=7-step=122144.ckpt", map_location="cpu")
state = ckpt["state_dict"]
for i, k in enumerate(state.keys()):
    print(k)
    if i == 10:
        break

num_classes = len(list(bird_df['eBird']))
backbone_name = train_cfg.BACKBONE_NAME
image_shape = audio_cfg.IMAGE_SHAPE

last_path = save_models(paths, backbone_name, image_shape, num_classes)

#Checking it loads OK, as this is the weights file to be used in the inference notebook
model = get_model(last_path, backbone_name, image_shape, num_classes)