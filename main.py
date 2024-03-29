import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import sys
import os
import pickle


# Adding the realtime action detection directory to the path to access some of its functionality
FASTER_RCNN_PATH = os.path.join(os.path.dirname(__file__),
                               '../realtime-action-detection')
sys.path.append(FASTER_RCNN_PATH)


from data import (CLASSES, detection_collate)
import load_faster_r_cnn as frcnn
import progress_net as pnet
import os
from tqdm import tqdm


NUM_EPOCHS = 5
LR = 1e-4
CUTOFF = 300
# CUTOFF = int(0.2*159289) # Cutoff is the number of samples that are passed into the model per epoch. This enables testing on a smaller dataset
NUM_CLASSES = len(CLASSES) + 1  # +1 'background' class
loss_type = 'BO' # 'MSE'
RUN_NAME = 'Default_params_v1'
# Setting up a run type to indicate whether the model is being trained or just tested (inference)
run_type = 'test' # or 'train'
if run_type == 'test':
    NUM_EPOCHS = 1 # Testing only with one pass of the test dataset

# Split type is the variable used to choose between running through the train or test list of ucf24.
# These lists indicate which dataset entries are passed through the network and can be found in the
# splitfiles directory ucf24/splitfiles.
split_type = 'test' # OR 'train'

# Visualize runs inference on the test dataset for a the first few actions. It saves predicted action progress values to later plot them
visualise = True 

# Setting up GPU availability
CUDA = True
if CUDA and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


backbone_net, dataset = frcnn.setup_backbone(
    split=split_type,
    BASENET=os.path.join(os.path.dirname(__file__),
                         FASTER_RCNN_PATH,
                         'rgb-ssd300_ucf24_120000.pth'),
    DATASET_PATH=os.path.join(os.path.dirname(__file__),
                              FASTER_RCNN_PATH,
                              'ucf24/'))

if loss_type == 'BO':
    bo_loss = pnet.BOLoss()
elif loss_type == 'MSE':
    mse_loss = nn.MSELoss()



# Setting up a MSE object to compute average progress prediction MSE across all runs
if run_type == 'test':
    mse = nn.MSELoss()

if CUDA:
    progress_net = pnet.ProgressNet().cuda()
else:
    progress_net = pnet.ProgressNet()

optimizer = optim.Adam(progress_net.parameters(), lr=LR)
lin_prog = pnet.LinearProgress(split=split_type)


'''
print("Testing linear progress method")
print(f"cutoff: {CUTOFF}")
for sample_ind in range(CUTOFF):
    try:
        progress = round(lin_prog.get_progress_value(sample_ind),4)
        if progress == None:
            print("Gotcha!")
            print(i)
            print(lin_prog.last_match)
    except Exception as e:
        print(sample_ind)
        print(lin_prog.last_match)
'''


# Setting up the dataloader for making inferences on tubes
data_loader = data.DataLoader(dataset,
                              shuffle=False,
                              collate_fn=detection_collate,
                              pin_memory=True)




num_samples = len(data_loader) # Number of samples i the dataset. A sample is an image (or a batch with size 1)
print('Number of images: ', len(dataset),
      '\nNumber of batches: ', num_samples)



logdir = f"runs/lr_{LR}_loss_type_{loss_type}_epochs_{NUM_EPOCHS}_data_used_{split_type}_run_{RUN_NAME}"

# Setting up the tensorboard summary writer
writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), logdir))


sample_itr = iter(data_loader) # Iterator to iterate through the data_loader

# Counting the number of total learning steps and some test metrics
total_steps = 0
visualise_progress = [] # List to store the predicted progess values for visualisation
print(f"First few tube durations: {lin_prog.tube_durations[:3]}")
EWMA_progress_MSE = 0
average_progress_MSE = 0

for i in range(NUM_EPOCHS):
    # iterate over samples
    loop = tqdm(range(CUTOFF))
    for sample_ind in loop:
        loop.set_description(f"Epoch {i} Samples")
        
        # if CUDA:
        #     torch.cuda.synchronize()

        ######################################
        # Faster R-CNN

        # get the sample's data
        images, targets, img_indexs = next(sample_itr)
        height, width = images.size(2), images.size(3)

        if CUDA:
            images = images.cuda()

        with torch.no_grad():
            # No grad since we are only using the backbone for inference and not to train it again
            output = backbone_net(images)

        conf_scores, decoded_boxes = frcnn.get_scores_and_boxes(output, backbone_net)

        # iterate over all classes for this sample
        detections_dict = {} # Creating a dictionary to store the detected classes (highest confidence one per class if they are above threshold confidence)
        for cl_ind in range(1, NUM_CLASSES):
            # class_detections is an array with 5 element arrays for each detection: [x1 y1 x2 y2 confidence]
            class_detections = frcnn.get_class_detections(
                cl_ind,
                conf_scores,
                decoded_boxes,
                height, width)

            # Keeping only those detections that have a confidence > 50%
            # Further, keeping only one detection per image
            class_detections = np.array(class_detections)

            if class_detections.shape[0] == 0:
                class_detections = np.array([])
            elif class_detections.shape[0] == 1:
                if class_detections[0][-1] < 0.5:
                    class_detections = np.array([])
                else:
                    detections_dict[cl_ind] = class_detections[0]

            elif class_detections.shape[0] > 1:
                class_detections = class_detections[np.argmax(class_detections[:,-1])]
                if class_detections[-1] < 0.5:
                    class_detections = np.array([])
                else:
                    detections_dict[cl_ind] = class_detections


        # print(detections_dict)
        # For now, ProgressNet only returns one action progress value per tube.
        # Hence, detections dict is reduced to get the highest confidence bounding box
        # and the corresponding action class.
        # However, future improvements can be made to return multiple action progresses per tube.  TODO
        # We chose to keep detections_dict as it offers an opportunity for future improvements
        # even though it is a bit redundant at present
        detected_class = None
        highest_conf_bbox = None
        if not len(detections_dict) == 0:
            detected_bboxes = np.array(list(detections_dict.values()))
            max_conf = np.argmax(detected_bboxes[:,-1])
            highest_conf_bbox = detected_bboxes[max_conf]
            detected_class = list(detections_dict.keys())[max_conf]
        else:
            continue

        # print(f"class : {detected_class}")
        # print(f"bbox : {highest_conf_bbox}")

        bbox = np.zeros((1,5))
        bbox[0,0] = 0
        bbox[0,1:5] = highest_conf_bbox[:-1]
        if CUDA:
            bbox = torch.tensor(bbox).float().cuda()
        else:
            bbox = torch.tensor(bbox).float()
        image = images[0]

        ######################################
        # ProgressNet 
        progress = lin_prog.get_progress_value(sample_ind)
        if progress == None:
            continue
        progress = round(progress,4)



        if run_type == 'train':
            predicted_progress = progress_net(image, bbox)
        elif run_type == 'test':
            with torch.no_grad():
                predicted_progress = progress_net(image, bbox)

        if visualise == True:
            if sample_ind < lin_prog.tube_durations[2][1]:
                visualise_progress.append((predicted_progress, progress))

        # Loss
        predicted_progress = predicted_progress[0,0,0]
        if loss_type == 'MSE':
            loss = mse_loss(torch.tensor(progress), predicted_progress)
        elif loss_type == 'BO':
            action_tube = lin_prog.tube_durations[lin_prog.last_match]
            loss = bo_loss(torch.tensor(progress), predicted_progress, [(action_tube[0], action_tube[1])]) 


        if run_type == 'train':
            # If training, then optimize the model and write training metrics
            # Optimize the model
            optimizer.zero_grad() # Not calling zero_grad as we are working with RNNs
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss vs Total Steps (across all episodes)", loss, total_steps)
            total_steps += 1
        elif run_type == 'test':
            EWMA_progress_MSE = 0.5*EWMA_progress_MSE + 0.5*mse(torch.tensor(progress), predicted_progress)
            average_progress_MSE += mse(torch.tensor(progress), predicted_progress)


        # print(f'Sample idx: {sample_ind} | predicted_progress {predicted_progress} | progress {progress}')


print("Training Completed")

if run_type == 'test':
    average_progress_MSE = average_progress_MSE/(NUM_EPOCHS*CUTOFF)
    print(f"Exponentially Weighted Moving Average of Progreess MSE: {EWMA_progress_MSE}")
    print(f"Average MSE of progress: {average_progress_MSE}")

# SAVE_PATH = logdir + ".pt"
# torch.save(progress_net.state_dict(), SAVE_PATH)

# if visualise == True:
#     with open('visualise_progress.pickle', 'wb') as handle:
#         pickle.dump(visualise_progress, handle, protocol=pickle.HIGHEST_PROTOCOL)