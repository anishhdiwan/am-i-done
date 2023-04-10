# Setup
import torch
import torch.nn as nn
from torchvision.ops import roi_pool
import pickle
import os


SPLITFILES_PATH = os.path.join(os.path.dirname(__file__),
                               '../realtime-action-detection',
                               'ucf24',
                               'splitfiles')

class BOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.phase_intervals = phase_intervals

    def forward(self, predictions, targets, phase_intervals):
        # N = predictions.shape[0]
        errors = torch.abs(predictions - targets)
        potentials = []

        for (l_k, u_k) in phase_intervals:
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
    super().__init__()
    self.spp = nn.MaxPool2d((30, 30), stride=10) # For a 300 x 300 image the output shape is 3, 28, 28
    # self. roi = nn.ROIPool((width, height), spatial_scale)
    self.fc7 = nn.Linear(2928, 128) # in_dim = 2928 as the spp output is 3x28x28 and the roi output is 3x16x12. Both are flattened and passed to fc7 
    self.lstm1 = nn.LSTM(128, 64, num_layers=1)
    self.lstm2 = nn.LSTM(64, 32, num_layers=1)
    self.fc8 = nn.Linear(32, 1)
    # self.softmax = nn.Softmax()
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
  
  def forward(self, x, bbox):
    '''
    x = image
    bbox = list with x1, y1, x2, y2 as bbox coordinates
    '''

    x = self.spp(x.view(1,3,300,300))
    y = roi_pool(x, bbox, (16,12))
    x = torch.cat((x.flatten(), y.flatten())).view(1,-1)
    x = self.fc7(x)
    x = self.relu(x)
    x = self.dropout(x)
    # print(x.shape)
    x, (h_n, c_n) = self.lstm1(x.view(1,1,128))
    # print(x.shape)
    x, (h_n, c_n) = self.lstm2(x)
    x = self.fc8(x)
    # x = self.dropout(x)
    # x = self.softmax(x)
    return torch.special.expit(x)
    # return x



class LinearProgress():

    '''
    This class returns a linear progress value depending on the sample index being fed into ProgressNet.
    This is possible since the faster R-CNN dataloader returns dataset frames sequentially as per the
    splitfiles mentioned in /ucf24/splitfiles. The pyannot pickle file in the /ucf24/splitfiles
    directory contains a dictionary with keys are the video names from the specific splitfiles.
    Each key has a start index, end index, and num frames
    value which can be used to calculate linear progess. To find out more, explore the pyannot pickle
    file and the train or test list text files. Also try exploring the sequentially loaded dataset by
    running the load_faster_r_cnn.py file. A more detailed description is provided in the reproduction blog.

    sample_ind: index of the sample being fed into progressnet (not to be confused with image index in the dataset)
    path: path to the directory containing the train/test split files and the pyannot pickle file is relative to main.py
          (assuming that main.py is in a directory outside of the faster rcnn directory)
    split: which type of split file to use
    '''

    def __init__(self, split='train'):
        if split=='train':
            data_path = os.path.join(SPLITFILES_PATH, 'trainlist01.txt')
        elif split == 'test':
            data_path = os.path.join(SPLITFILES_PATH, 'testlist01.txt')

        data_file = open(data_path, 'r')
        data_list = data_file.read()
        data_list = data_list.split("\n")
        data_file.close()
        # print(data_list)

        annot_path = os.path.join(SPLITFILES_PATH, 'pyannot.pkl')
        with open(annot_path, 'rb') as handle:
            annotations = pickle.load(handle)


        # Tube durations is a list containing 3 element tuples indicating the (tube start, tube end, action class)
        # for each tube that the dataloader loads
        tube_durations = []

        tube_start = 0
        for i in range(len(data_list)):
            sf = annotations[data_list[i]]['annotations'][0]['sf']
            ef = annotations[data_list[i]]['annotations'][0]['ef']
            label = annotations[data_list[i]]['label']

            tube_end = tube_start + (ef - sf)
            tube_durations.append((tube_start, tube_end, label))

            tube_start = tube_end + 1

        self.tube_durations = tube_durations
        print('=== Finished processing ground truth tubes!')

        # Last match is the last index in tube duration which matched with the sample index. i.e if tube durations
        # is a list like [(0, 46, 0), (47, 77, 0), (78, 109, 0), ..] and the last sample index was 46 (current is 47),
        # the last match would be 0 since it matches with the 0th index of tube durations.
        # The next last_match would be 1 as the current sample index is 47 which is in the 1st index
        # This helps quickly search through tube durations to get progress values
        self.last_match = 0


    def get_progress_value(self, sample_ind):
        if self.tube_durations[self.last_match][0] <= sample_ind <= self.tube_durations[self.last_match][1]:
            # same last_match still applies
            progress = (sample_ind - self.tube_durations[self.last_match][0])/(self.tube_durations[self.last_match][1] - self.tube_durations[self.last_match][0])
            return progress
        elif self.tube_durations[self.last_match + 1][0] <= sample_ind <= self.tube_durations[self.last_match + 1][1]:
            # last_match needs to be updated
            self.last_match += 1
            progress = (sample_ind - self.tube_durations[self.last_match][0])/(self.tube_durations[self.last_match][1] - self.tube_durations[self.last_match][0])
            return progress	



