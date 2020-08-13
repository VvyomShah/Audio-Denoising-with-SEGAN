import os
import librosa
import time

import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from torch.utils import data
from torch.autograd import Variable
from torch import optim

import soundfile
from scipy import signal
from scipy.io import wavfile

from utils import slice_signal, Generator

pre_emphasis = lambda batch: signal.lfilter([1, -0.95], [1], batch)
de_emphasis = lambda batch: signal.lfilter([1], [1, -0.95], batch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
PATH = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':

	generator = nn.DataParallel(Generator(), device_ids = [1, 0])
	state = torch.load(f'{PATH}/checkpoints/state-13.pkl', map_location=device)
	generator.load_state_dict(state['generator'])
	generator.to(device)

	for file in os.listdir(f'{PATH}/input'):
		
		# Read and slice audio input
		noisy_slices = slice_signal(f'{PATH}/input/{file}', 2**13, 1, 8000)
		enhanced_speech = []
		
		for noisy_slice in noisy_slices:
			noisy_slice = noisy_slice.reshape(1, 1, 8192)
			generator.eval()
			z = nn.init.normal(torch.Tensor(1, 1024, 8))
			noisy_slice = torch.from_numpy(pre_emphasis(noisy_slice)).type(torch.FloatTensor)
			z.to(device)
			noisy_slice.to(device)
			generated_speech = generator(noisy_slice, z).data.cpu().numpy()
			generated_speech = de_emphasis(generated_speech)
			generated_speech = generated_speech.reshape(-1)
			enhanced_speech.append(generated_speech)

		enhanced_speech = np.array(enhanced_speech).reshape(1, -1)
		name = filename.split('/')[-1]
		filename = f'{PATH}/output/enhanced_{name}'
		librosa.output.write_wav(filename, enhanced_speech.T, sr = 8000)