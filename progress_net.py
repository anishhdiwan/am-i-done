# Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import pickle
import os


class BOLoss(nn.Module):
    def __init__(self, phase_intervals):
        super(BOLoss, self).__init__()
        self.phase_intervals = phase_intervals

    def forward(self, predictions, targets):
        N = predictions.shape[0]
        errors = torch.abs(predictions - targets)
        potentials = []

        for (l_k, u_k) in self.phase_intervals:
            m_k = (l_k + u_k) / 2
            r_k = (u_k - l_k) / 2
            e_k = torch.min(
                torch.tensor(1.0),
                ((targets - m_k) / (r_k * torch.sqrt(torch.tensor(2.0)))) ** 2 +
                ((predictions - m_k) / (r_k * torch.sqrt(torch.tensor(2.0)))) ** 2
            )
            potentials.append(e_k)

        min_potentials = torch.stack(potentials).min(dim=0).values
        bo_loss = torch.mean(min_potentials * errors)

        return bo_loss


class ProgressNet(nn.Module):
  def __init__(self):
    super(ProgessNet, self).__init__()

    self.spp = nn.MaxPool2d((10, 10))
    # self. roi = nn.ROIPool((width, height), spatial_scale)
    self.fc7 = nn.Linear(in_dim, 128)
    self.lstm = nn.LSTM(128, 128, num_layers=2)
    self.fc8 = nn.Linear(128, 1)
    self.softmax = nn.Softmax()
  
  def forward(self, x, bbox):
  	'''
  	x = image
  	bbox = list with x1, y1, x2, y2 as bbox coordinates
  	'''
    x = spp(x)
    y = roi_pool(x, bbox, (10,10))
    x = torch.cat((x,y))
    x = fc7(x)
    x = lstm(x)
    x = softmax(x)
    return x


def get_progress_value(sample_ind, split='train'):
	'''
	This function returns a progress value depending on the sample index being fed into ProgressNet. This is possible since the faster R-CNN 
	dataloader returns dataset frames sequentially as per the splitfiles mentioned in /ucf24/splitfiles. The pyannot pickle file in the /ucf24/splitfiles
	directory contains a dictionary with keys are the video names from the specific splitfiles. Each key has a start index, end index, and num frames
	value which can be used to calculate linear progess. To find out more, explore the pyannot pickle file and the train or test list text files. Also
	try exploring the sequentially loaded dataset by running the load_faster_r_cnn.py file. A more detailed description is provided in the reproduction blog. 
	
	sample_ind: index of the sample being fed into progressnet (not to be confused with image index in the dataset)
	path: path to the directory containing the train/test split files and the pyannot pickle file is relative to main.py (assuming that main.py is in a directory outside of the faster rcnn directory)
	split: which type of split file to use
	'''

	if split=='train':
		data_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'trainlist01.txt')
	elif split == 'test':
		data_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'trainlist01.txt')

	data_file = open(data_path, 'r')
	data_list = data_file.read()
	data_list = data_list.split("\n")
	data_file.close()
	# print(data_list)

	annot_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'pyannot.pkl')
	with open(annot_path, 'rb') as handle:
		annotations = pickle.load(handle)

	



