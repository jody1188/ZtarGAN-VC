import argparse
import librosa
import numpy as np
import os
import pyworld
import json
import hifigan
import torch
from scipy.io import wavfile
import matplotlib.pyplot as plt

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfgs', type=str)
    args = parser.parse_args()
    return args

def load_wav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)
    return wav

def get_vocoder(config, device):
    name = config.vocoder.model
    speaker = config.vocoder.speaker

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("Fastspeech2/hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        ckpt = torch.load(f"Fastspeech2/hifigan/generator_{speaker}.pth.tar", map_location=torch.device(device))
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, config, lengths=None):
    name = config.vocoder.model
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * config.preprocessing.audio.max_wav_value
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs

def synth_samples(wav_name, predictions, vocoder, config, path):

    basenames = wav_name[0]
    mel_len = [mel.shape[0] for mel in predictions]

    if not os.path.exists(path):
        os.makedirs(path)

    mel_predictions = predictions.squeeze(1)
    mel_predictions = mel_predictions.transpose(1, 2)
    #lengths = mel_len * config.preprocessing.stft.hop_length
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, config, lengths=None
    )
    
    data_fig = [mel_predictions.cpu().numpy().squeeze(0)]+[wav for wav in wav_predictions]
    fig = plot_mel(
            data_fig, ["Converted Spectrogram", "Converted WAV"],
        )
    plt.savefig(os.path.join(path, "{}.png".format(basenames)))
    plt.close()

    sampling_rate = config.preprocessing.audio.sampling_rate
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)

def plot_mel(data, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel = data[i]
        if len(mel.shape) < 2:
            axes[i][0].plot(mel)
        else:
            axes[i][0].imshow(mel, origin="lower")
            axes[i][0].set_aspect(2.5, adjustable="box")
            axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")
        plt.tight_layout()

    return fig

def world_decompose(wav, fs, frame_period = 5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap

def world_encode_spectral_envelop(sp, fs, dim=36):
    # Get Mel-cepstral coefficients (MCEPs)
    #sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp

def world_decode_spectral_envelop(coded_sp, fs):
    # Decode Mel-cepstral to sp
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp

def world_encode_wav(wav_file, fs, frame_period=5.0, coded_dim=36):
    wav = load_wav(wav_file, sr=fs)
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=fs, frame_period=frame_period)
    coded_sp = world_encode_spectral_envelop(sp = sp, fs = fs, dim = coded_dim)
    return f0, timeaxis, sp, ap, coded_sp

def world_speech_synthesis(f0, coded_sp, ap, fs, frame_period):
    decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
    # TODO
    min_len = min([len(f0), len(coded_sp), len(ap)])
    f0 = f0[:min_len]
    coded_sp = coded_sp[:min_len]
    ap = ap[:min_len]
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)
    return wav

def world_synthesis_data(f0s, coded_sps, aps, fs, frame_period):
    wavs = list()
    for f0, decoded_sp, ap in zip(f0s, coded_sps, aps):
        wav = world_speech_synthesis(f0, coded_sp, ap, fs, frame_period)
        wavs.append(wav)
    return wavs

def coded_sps_normalization_fit_transoform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis = 1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis = 1, keepdims = True)
    coded_sps_std = np.std(coded_sps_concatenated, axis = 1, keepdims = True)
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    return coded_sps_normalized, coded_sps_mean, coded_sps_std

def coded_sp_statistics(coded_sps):
    # sp shape (T, D)
    coded_sps_concatenated = np.concatenate(coded_sps, axis = 0)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis = 0, keepdims = False)
    coded_sps_std = np.std(coded_sps_concatenated, axis = 0, keepdims = False)
    return coded_sps_mean, coded_sps_std

def normalize_coded_sp(coded_sp, coded_sp_mean, coded_sp_std):
    normed = (coded_sp - coded_sp_mean) / coded_sp_std
    return normed

def coded_sps_normalization_transoform(coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    
    return coded_sps_normalized

def coded_sps_normalization_inverse_transoform(normalized_coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps

def coded_sp_padding(coded_sp, multiple = 4):
    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values = 0)
    return coded_sp_padded

def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1 
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    return wav_padded

def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted

def wavs_to_specs(wavs, n_fft = 1024, hop_length = None):

    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)
        stfts.append(stft)

    return stfts


def wavs_to_mfccs(wavs, sr, n_fft = 1024, hop_length = None, n_mels = 128, n_mfcc = 24):

    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y = wav, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, n_mfcc = n_mfcc)
        mfccs.append(mfcc)

    return mfccs


def mfccs_normalization(mfccs):

    mfccs_concatenated = np.concatenate(mfccs, axis = 1)
    mfccs_mean = np.mean(mfccs_concatenated, axis = 1, keepdims = True)
    mfccs_std = np.std(mfccs_concatenated, axis = 1, keepdims = True)

    mfccs_normalized = list()
    for mfcc in mfccs:
        mfccs_normalized.append((mfcc - mfccs_mean) / mfccs_std)
    
    return mfccs_normalized, mfccs_mean, mfccs_std


def sample_train_data(dataset_A, dataset_B, n_frames = 128):

    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:,start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:,start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B