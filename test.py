# import numpy as np

# test = np.array([[8.8457804e+00, 1.6967372e+02, 4.6079449e+01, 2.2477542e+02, 1.4885605e-02],
#  [2.0961029e+02, 4.9050087e+01, 2.4383253e+02, 1.0481744e+02, 1.3389414e-02],
#  [0.0000000e+00, 1.7931505e+02, 2.4877831e+01, 2.1741859e+02, 1.1113531e-02]])

# print(test.shape)

# if not test.shape[0] == 0:
# 	print(test[np.argmax(test[:,-1])])

import sys
import os
import pickle 
sys.path.append('../realtime_action_detection')

split = 'train'


class LinearProgress():

    '''
    This class returns a linear progress value depending on the sample index being fed into ProgressNet. This is possible since the faster R-CNN 
    dataloader returns dataset frames sequentially as per the splitfiles mentioned in /ucf24/splitfiles. The pyannot pickle file in the /ucf24/splitfiles
    directory contains a dictionary with keys are the video names from the specific splitfiles. Each key has a start index, end index, and num frames
    value which can be used to calculate linear progess. To find out more, explore the pyannot pickle file and the train or test list text files. Also
    try exploring the sequentially loaded dataset by running the load_faster_r_cnn.py file. A more detailed description is provided in the reproduction blog. 
    
    sample_ind: index of the sample being fed into progressnet (not to be confused with image index in the dataset)
    path: path to the directory containing the train/test split files and the pyannot pickle file is relative to main.py (assuming that main.py is in a directory outside of the faster rcnn directory)
    split: which type of split file to use
    '''

    def __init__(self, split='test'):
        if split=='train':
            data_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'trainlist01.txt')
        elif split == 'test':
            data_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'testlist01.txt')

        data_file = open(data_path, 'r')
        data_list = data_file.read()
        data_list = data_list.split("\n")
        data_file.close()
        # print(data_list)

        annot_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'pyannot.pkl')
        with open(annot_path, 'rb') as handle:
            annotations = pickle.load(handle)

        # print(len(annotations))
        # print(len(data_list))
        # Tube durations is a list containing 3 element tuples indicating the (tube start, tube end, action class) for each tube that the dataloader loads
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

        # Last match is the last index in tube duration which matched with the sample index. i.e if sample index is if tube durations
        # is a list like [(0, 46, 0), (47, 77, 0), (78, 109, 0), ..] and the last sample index was 46 (current is 47), the last match would be 0 since it matches
        # with the 0th index of tube durations. The next last_match would be 1 as the current sample index is 47 which is in the 1st index
        # This helps quickly search through the whole list of tube durations to get progress values
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

lin_prog = LinearProgress()
print(lin_prog.tube_durations[:3])
for sample_ind in range(50):
    print(f"sample index {sample_ind} | last match {lin_prog.last_match} | progress {lin_prog.get_progress_value(sample_ind)}")
    # print(lin_prog.get_progress_value(sample_ind))
    # print(lin_prog.last_match)
    print("---")






# print(os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'trainlist01.txt'))

# if split=='train':
#     data_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'trainlist01.txt')
# elif split == 'test':
#     data_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'testlist01.txt')

# data_file = open(data_path, 'r')
# data_list = data_file.read()
# data_list = data_list.split("\n")
# data_file.close()

# # print(len(data_list))
# # print(data_list[-1])

# annot_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'pyannot.pkl')
# with open(annot_path, 'rb') as handle:
#     annotations = pickle.load(handle)

# label = annotations[data_list[0]]['label']
# numf = annotations[data_list[0]]['numf']
# sf = annotations[data_list[0]]['annotations'][0]['sf']
# ef = annotations[data_list[0]]['annotations'][0]['ef']

# # print(label)
# # print(numf)
# # print(sf)
# # print(ef)

# # Tube durations is a list containing 3 element tuples indicating the (tube start, tube end, action class) for each tube that the dataloader loads
# tube_durations = []

# tube_start = 0
# for i in range(len(data_list[:10])):
#     sf = annotations[data_list[i]]['annotations'][0]['sf']
#     ef = annotations[data_list[i]]['annotations'][0]['ef']
#     label = annotations[data_list[i]]['label']

#     tube_end = tube_start + (ef - sf)
#     tube_durations.append((tube_start, tube_end, label))

#     tube_start = tube_end + 1

# print(tube_durations)

