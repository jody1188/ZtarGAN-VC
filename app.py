import streamlit as st
import time
import re
from string import punctuation
import json
import pickle
import glob



import torch
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
from PIL import Image


import sys


sys.path.append("FastSpeech2")


from FastSpeech2.model import *
from FastSpeech2.utils.model import *
from FastSpeech2.utils.model import get_vocoder as get_vocoder_f
from FastSpeech2.utils.tools import *
from FastSpeech2.dataset import *
from FastSpeech2.text import *
from FastSpeech2.synthesize import *

sys.path.append("ZtarGAN_VC")

print(sys.path)

import json
from collections import namedtuple

from ZtarGAN_VC.model import *
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from ZtarGAN_VC.data_loader import to_categorical, to_embedding
import librosa
from ZtarGAN_VC.utils import *
from glob import glob

import ZtarGAN_VC.audio as Audio
from ZtarGAN_VC.preprocess import get_mel_from_wav
from ZtarGAN_VC.convert import *
from ZtarGAN_VC.utils import get_vocoder as get_vocoder_c


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
def text_to_speech(configs, input_text):
    with open(configs, 'r') as f:
        cfgs = json.load(f)
    
    # Check source texts
    if cfgs['mode'] == "batch":
        assert cfgs['source'] is not None and input_text is None
    if cfgs['mode'] == "single":
        assert cfgs['source'] is None and input_text is not None
    
    # Read Config    
    preprocess_config = cfgs['preprocess']
    model_config = cfgs['model']
    train_config = cfgs['train']
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(cfgs, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder_f(model_config, device)

    # Preprocess texts
    if cfgs['mode'] == "batch":
        # Get dataset
        dataset = TextDataset(cfgs['source'], preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if cfgs['mode'] == "single":
        ids = raw_texts = [input_text[:100]]
        speakers = np.array([cfgs['speaker_id']])
        if preprocess_config["preprocessing"]["text"]["language"] == "kr":
            texts = np.array([preprocess_korean(input_text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(input_text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = cfgs['pitch_control'], cfgs['energy_control'], cfgs['duration_control']
    
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=cfgs['pitch_control'],
                e_control=cfgs['energy_control'],
                d_control=cfgs['duration_control']
            )

    synthesize(model, cfgs['restore_step'], configs, vocoder, batchs, control_values)
    
        

    
def display_tts():
    global text
    st.markdown(" Enter the text that you would like to be voiced ")
    text = st.text_area(label = 'Text', height = 50) 
    
    if text is not '':
        result_wav = text_to_speech('app_tts_config.json',text)
        
        audio_file = open(f'output/result/Conference/{text}.wav', 'rb')
        audio_bytes = audio_file.read()
        
        st.write(" Speech ")
        st.audio(audio_bytes, format = 'audio/wav', start_time = 0)
    
    
        
               
def conversion(config):
    with open("app_vc_config.json", 'r') as f:
        config = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    src_mc = np.load("./output/result/Conference/output.npy", allow_pickle = True)
    trg_mc = np.load("./preprocessed_data/mel_test/C0006-0105M1012-2__000_0-05935899.npy").T
    cfg_speaker_encoder = config.speaker_encoder    
    spk_emb_trg = to_embedding(trg_mc, cfg_speaker_encoder, num_classes=11)
    
    
    mel_spectrogram = src_mc[1].unsqueeze(1)
    spk_conds = torch.FloatTensor([spk_emb_trg]).to(device)
    G = Generator().to(device)
    G_path = join(config.directories.model_save_dir, f'{config.model.resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    coded_sp_converted_norm = G(mel_spectrogram, spk_conds).data
    vocoder = get_vocoder_c(config, device)
    synth_samples('test1', coded_sp_converted_norm, vocoder, config, 'output/converted')


    
def display_conversion():
    st.subheader('Voice Conversion')
    st.markdown(" Press convert button to convert the voice ")
    if st.button("Convert"):
        con = st.container()
        st.write(" Converted ")
        convert_wav = conversion("app_vc_config.json")
        
        audio_file = open("output/converted/t.wav", 'rb')
        audio_bytes = audio_file.read()
        
        st.audio(audio_bytes, format = 'audio/wav', start_time = 0)
      
  
def main():
    st.title("Boy's Voice")
    st.subheader('Text to Speech')

       
    
if __name__ == "__main__":
    main()
    display_tts()
    display_conversion()
    