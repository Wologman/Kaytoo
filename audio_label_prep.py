'''
Functions and classes to convert various audio datasets into a standard format, 
output should be compatible with BirdCLEF & Xeno-canto datasets
Columns can be missing.  
Compulsory is primary_label (matching the correct ebird code) & filename 
Additional columns start_time, duration.  
Either the recordings must be less than 20 seconds, or they need a start_time & duration
For an example (without start_time, or duration or file_path)
https://www.kaggle.com/competitions/birdclef-2023/data?select=train_metadata.csv

file_path                 | primary_label | start_time | duration | secondary_labels | type    | latitude  | longitude  | scientific_name   | common_name         | author | 
sobkiw2/my_kiwi_file.wav  | sobkiw2       | 45.2       | 6        | []               |['call'] | -45.4085  | 167.3591   | Apteryx australis | Southern Brown Kiwi

filename: Relative to what ever parent directory the whole dataset is held in, use / not \ if needed
type: [''], ['male', 'song'], ['call'], ['adult', 'call', 'sex uncertain'], ['alarm call']
secondary_labels: Other birds present but not dominant, as a list, eg [], ['antalb1'], ['antalb1',whcalb1']

Exceptions:
Unknown or unidentified,   primary_label: 'spybird', scientific_name: vitriolix
No suitable ebird code     primary_label: 'badbird', scientific_name: badass birdus
'''

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
from bird_naming_utils import BirdCodeConverter
import re

class Stop: #red
    s = '\033[1m'
    e = '\033[0m'


def extract_soundfile_path(filepath):
    wav_file_name = filepath.with_suffix('.wav').name
    data_dir = filepath.parent.parent.parent.name
    parent_dir = filepath.parent.parent.name
    return data_dir + '/' + parent_dir + '/' + wav_file_name


def extract_bird_tags(filepath):
    sound_file_path = extract_soundfile_path(filepath)
    path_name = filepath.name

    empty = [{
                'filepath' : sound_file_path,
                'code': 0,
                'start_time': 0,
                'duration': 0,
                'freq_low': 0,
                'freq_high': 0,
            }]
    
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        bird_tags = []
        
        for bird_tag in root.findall('BirdTag'):
            code = bird_tag.find('Code').text if bird_tag.find('Code') is not None else None
            time_second = bird_tag.find('TimeSecond').text if bird_tag.find('TimeSecond') is not None else None
            duration = bird_tag.find('Duration').text if bird_tag.find('Duration') is not None else None
            freq_low = bird_tag.find('FreqLow').text if bird_tag.find('FreqLow') is not None else None
            freq_high = bird_tag.find('FreqHigh').text if bird_tag.find('FreqHigh') is not None else None

            bird_tags.append({
                'filepath' : str(sound_file_path),
                'code': code,
                'start_time': time_second,
                'duration': duration,
                'freq_low': freq_low,
                'freq_high': freq_high,
                })
        return None, bird_tags
    except:
        print(Stop.s + f'Unable to parse the xml file {filepath.name}' + Stop.e)

        return path_name, empty