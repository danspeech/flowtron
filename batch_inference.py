import argparse
import json
import os
from random import randint, choice
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import torch

from flowtron import Flowtron
from train import update_params
import sys
sys.path.insert(0, "tacotron2")
sys.path.insert(0, "tacotron2/waveglow")
from glow import WaveGlow

from data import Data
from scipy.io.wavfile import write

import numpy as np

seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def load_models(flowtron_path, waveglow_path):
    # load waveglow
    waveglow = torch.load(waveglow_path)['model'].cuda().eval()
    waveglow.cuda()
    for k in waveglow.convinv:
        k.float()
    waveglow.eval()

    # load flowtron
    try:
        model = Flowtron(**model_config).cuda()
        state_dict = torch.load(flowtron_path, map_location='cpu')['state_dict']
        model.load_state_dict(state_dict)
    except KeyError:
        model = torch.load(flowtron_path)['model']

    model.eval()
    print("Loaded model '{}')".format(flowtron_path))

    return model, waveglow


def load_sentences(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().split("\n")


def predict_framelength(text):
    vokaler = "a,e,i,o,u,y,æ,ø,å".split(",")
    number_of_vokaler = sum([text.count(v) for v in vokaler])
    return number_of_vokaler * 50


def calculate_diagonal_path_top_down(m):
    """

    Parameters
    ----------
    m matrix to calculate
    direction
        0 = from bot to top, left to right
        1 = from top to bot, left to right
    Returns
    -------

    """
    shape = m.shape
    max_i = shape[0] - 1 # row
    max_j = shape[1] - 1 # col
    i = 0
    j = 0
    fin_sum = 0

    while True:
        fin_sum += m[i, j]

        if i == max_i:
            j += 1
        elif j == max_j:
            i += 1
        elif m[i, j+1] > m[i+1, j]:
            j += 1
        else:
            i += 1

        if i == max_i and j == max_j:
            break

    return fin_sum


def calculate_diagonal_path_down_top(m):
    """

    Parameters
    ----------
    m matrix to calculate
    direction
        0 = from bot to top, left to right
        1 = from top to bot, left to right
    Returns
    -------

    """
    shape = m.shape
    max_i = shape[0] - 1 # row
    max_j = shape[1] - 1 # col
    i = 0
    j = max_j
    fin_sum = 0

    while True:
        fin_sum += m[i, j]

        if i == max_i:
            j -= 1
        elif j == 0:
            i += 1
        elif m[i, j-1] > m[i+1, j]:
            j -= 1
        else:
            i += 1

        if i == max_i and j == 0:
            break

    return fin_sum

def batch_inference(flowtron_path, waveglow_path, synth_text_data_path, sigma):
    flowtron, waveglow = load_models(flowtron_path, waveglow_path)
    ignore_keys = ['training_files', 'validation_files']

    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

    speaker_keys = list(trainset.speaker_ids.keys())
    synth_sentences = load_sentences(synth_text_data_path)

    frame_interval = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]

    for i, sentence in enumerate(synth_sentences):
        if i < 139:
            continue
        print("During sentence: {0}".format(i+1))
        # Pick random speaker id for sentence
        speaker_id = choice(speaker_keys)
        speaker_vecs = trainset.get_speaker_id(speaker_id).cuda()
        text = trainset.get_text(sentence).cuda()
        speaker_vecs = speaker_vecs[None]
        text = text[None]

        frames = predict_framelength(sentence)
        frame_intervals = list(map(lambda x: x + frames, frame_interval))

        scores = []
        attention_list = []

        # Sentence place
        fpath = "results/{}".format(i + 1)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
            os.chmod(fpath, 0o775)

        scores_file = open(fpath + "/" + "scorex.csv", "w", encoding="utf-8")

        for n_frames in frame_intervals:
            if n_frames < 30:
                n_frames = 50
            with torch.no_grad():
                residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
                mels, attentions = flowtron.infer(
                    residual, speaker_vecs, text, gate_threshold=0.5)

            attention_score = 0
            attention_sub_list = []
            for k in range(len(attentions)):
                attention = torch.cat(attentions[k]).cpu().numpy()
                attention = attention[:, 0].transpose()
                attention_sub_list.append(attention)
                n_cols = attention.shape[1]
                if k == 0:
                    attention_score += (calculate_diagonal_path_down_top(attention) / n_cols)
                else:
                    attention_score += (calculate_diagonal_path_top_down(attention) / n_cols)

            scores_file.write(str(n_frames) + "," + str(attention_score) + "\n")

            attention_list.append(attention_sub_list)
            scores.append(attention_score)

        scores_file.close()

        highest_score = max(scores)
        if highest_score > 1.5:
            index = scores.index(highest_score)
            # plot attention
            for k in range(len(attention_list[index])):
                fig, axes = plt.subplots(1, 2, figsize=(16, 4))
                axes[0].imshow(mels[0].cpu().numpy(), origin='bottom', aspect='auto')
                axes[1].imshow(attention, origin='bottom', aspect='auto')
                fig.savefig(os.path.join(fpath, '{}_sid{}_attnlayer{}.png'.format(list(frame_intervals)[index], speaker_id, k)))
                plt.close("all")

            with torch.no_grad():
                audio = waveglow.infer(mels, sigma=0.8).float()

            audio = audio.cpu().numpy()[0]
            # normalize audio for now
            audio = audio / np.abs(audio).max()

            write(os.path.join(fpath, '{}_sid{}_sigma{}.wav'.format(n_frames, speaker_id, sigma)),
                  data_config['sampling_rate'], audio)

        else:
            print("Could not get high score for sentence: {} ".format(sentence))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-b', '--batch_size', help="batch size for inference", default=32, type=int)
    parser.add_argument('-f', '--flowtron_path',
                        help='Path to flowtron state dict', type=str)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow state dict', type=str)
    parser.add_argument('-t', '--text_file', help='Text file to synthesize', type=str)
    parser.add_argument('-n', '--n_frames', help='Number of frames',
                        default=400, type=int)
    parser.add_argument('-o', "--output_dir", default="results/")
    parser.add_argument("-s", "--sigma", default=1.0, type=float)

    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)

    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    batch_inference(args.flowtron_path, args.waveglow_path, args.text_file, args.sigma)