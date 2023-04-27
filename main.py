import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import sys
import os
# import pickle

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
BATCH_SIZE = 7
CUTOFF = 300
# CUTOFF = int(0.2*159289) # Cutoff is the number of samples that are passed into the model per epoch.
# This enables testing on a smaller dataset
NUM_CLASSES = len(CLASSES) + 1  # +1 'background' class
LOSS_TYPE = 'BO'  # 'MSE'
RUN_NAME = 'Default_params_v1'
# Setting up a run type to indicate whether the model is being trained or just tested (inference)
RUN_TYPE = 'test'  # or 'train'
if RUN_TYPE == 'test':
    NUM_EPOCHS = 1  # Testing only with one pass of the test dataset

# Split type is the variable used to choose between running through the train or test list of ucf24.
# These lists indicate which dataset entries are passed through the network and can be found in the
# splitfiles directory ucf24/splitfiles.
SPLIT_TYPE = 'test'  # OR 'train'

# Visualize runs inference on the test dataset for a the first few actions.
# It saves predicted action progress values to later plot them
VISUALIZE = True

# Setting up CUDA availability
CUDA = os.environ.get("CUDA") == '1'
if CUDA and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def main():
    backbone_net, dataset = frcnn.setup_backbone(
        split=SPLIT_TYPE,
        BASENET=os.path.join(os.path.dirname(__file__),
                             FASTER_RCNN_PATH,
                             'rgb-ssd300_ucf24_120000.pth'),
        DATASET_PATH=os.path.join(os.path.dirname(__file__),
                                  FASTER_RCNN_PATH,
                                  'ucf24/'))

    if LOSS_TYPE == 'BO':
        bo_loss = pnet.BOLoss()
    elif LOSS_TYPE == 'MSE':
        mse_loss = nn.MSELoss()

    # Setting up a MSE object to compute average progress prediction MSE across all runs
    if RUN_TYPE == 'test':
        mse = nn.MSELoss()

    if CUDA:
        progress_net = pnet.ProgressNet().cuda()
    else:
        progress_net = pnet.ProgressNet()

    optimizer = optim.Adam(progress_net.parameters(), lr=LR)
    lin_prog = pnet.LinearProgress(split=SPLIT_TYPE)

    # print("Testing linear progress method")
    # print(f"cutoff: {CUTOFF}")
    # for sample_ind in range(CUTOFF):
    #     try:
    #         progress = round(lin_prog.get_progress_value(sample_ind),4)
    #         if progress == None:
    #             print("Gotcha!")
    #             print(i)
    #             print(lin_prog.last_match)
    #     except Exception as e:
    #         print(sample_ind)
    #         print(lin_prog.last_match)

    # Setting up the dataloader for making inferences on tubes
    data_loader = data.DataLoader(dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  collate_fn=detection_collate,
                                  pin_memory=True)

    logdir = f"runs/lr_{LR}_LOSS_TYPE_{LOSS_TYPE}_epochs_{NUM_EPOCHS}_data_used_{SPLIT_TYPE}_run_{RUN_NAME}"

    # Setting up the tensorboard summary writer
    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), logdir))

    batch_itr = iter(data_loader)  # Iterator to iterate through the data_loader

    # Counting the number of total learning steps and some test metrics
    total_steps = 0
    visualize_progress = []  # List to store the predicted progess values for visualisation
    print(f"First few tube durations: {lin_prog.tube_durations[:3]}")
    EWMA_progress_MSE = 0
    average_progress_MSE = 0

    for i in range(NUM_EPOCHS):
        # iterate over samples
        loop = tqdm(range(len(data_loader)))
        for batch_ind in loop:
            loop.set_description(f"Epoch {i}:")

            if CUDA:
                torch.cuda.synchronize()

            ######################################
            # Faster R-CNN

            # get the batch's data
            images, targets, img_indexs = next(batch_itr)
            height, width = images.size(2), images.size(3)

            if CUDA:
                images = images.cuda()

            with torch.no_grad():
                # No grad since we are only using the backbone for inference and not to train it again
                output = backbone_net(images)

            detections = [[] for _ in range(NUM_CLASSES - 1)]

            # iterate over samples in a batch
            for b in range(BATCH_SIZE):
                conf_scores, decoded_boxes = frcnn.get_scores_and_boxes(output, backbone_net, b)

                # iterate over all classes for this sample
                for cl_ind in range(1, NUM_CLASSES):
                    # class_detections is an array with 5 element arrays for each detection: [x1 y1 x2 y2 confidence]
                    class_detections = frcnn.get_class_detections(
                        cl_ind,
                        conf_scores,
                        decoded_boxes,
                        height, width,
                        conf_thresh=0.5)

                    # detections will be of shape:
                    # (classes) * (samples) * (# dets. in sample for class) * (5)
                    detections[cl_ind - 1].append(class_detections)


            # print(detections)
            # For now, ProgressNet only returns one action progress value per tube.
            # Hence, detections dict is reduced to get the highest confidence bounding box
            # and the corresponding action class.
            # However, future improvements can be made to return multiple action progresses per tube.  TODO
            # We chose to keep detections as it offers an opportunity for future improvements
            # even though it is a bit redundant at present
            bboxes = np.zeros((BATCH_SIZE, NUM_CLASSES - 1, 4))
            best_bbox = [[] for _ in range(NUM_CLASSES - 1)]

            for im_ind in range(BATCH_SIZE):
                for cl_ind in range(NUM_CLASSES - 1):
                    if len(detections[cl_ind][im_ind]) > 0:
                        best_bbox = np.argmax(detections[cl_ind, im_ind, :, -1])

                bboxes[im_ind] = detected_bboxes[cl_ind, im_ind, ax_conf]

            print(len(detections))
            print(len(detections[0]))
            print(len(detections[0][0]))
            print(len(detections[0][0][0]))
            exit()

            # print(f"class : {detected_class}")
            # print(f"bbox : {highest_conf_bbox}")

            bbox = np.zeros((1, 5))
            bbox[0, 0] = 0  # TODO: possibly unnecessary
            bbox[0, 1:5] = highest_conf_bbox[:-1]
            if CUDA:
                bbox = torch.tensor(bbox).float().cuda()
            else:
                bbox = torch.tensor(bbox).float()
            image = images[0]

            ######################################
            # ProgressNet
            progress = lin_prog.get_progress_value(batch_ind)
            if progress is None:
                continue
            progress = round(progress, 4)

            if RUN_TYPE == 'train':
                predicted_progress = progress_net(image, bbox)
            elif RUN_TYPE == 'test':
                with torch.no_grad():
                    predicted_progress = progress_net(image, bbox)

            if VISUALIZE is True:
                if batch_ind < lin_prog.tube_durations[2][1]:
                    visualize_progress.append((predicted_progress, progress))

            # Loss
            predicted_progress = predicted_progress[0, 0, 0]
            if LOSS_TYPE == 'MSE':
                loss = mse_loss(torch.tensor(progress), predicted_progress)
            elif LOSS_TYPE == 'BO':
                action_tube = lin_prog.tube_durations[lin_prog.last_match]
                loss = bo_loss(torch.tensor(progress), predicted_progress, [(action_tube[0], action_tube[1])]) 

            if RUN_TYPE == 'train':
                # If training, then optimize the model and write training metrics
                # Optimize the model
                optimizer.zero_grad()  # Not calling zero_grad as we are working with RNNs
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss vs Total Steps (across all episodes)", loss, total_steps)
                total_steps += 1
            elif RUN_TYPE == 'test':
                EWMA_progress_MSE = 0.5*EWMA_progress_MSE + 0.5*mse(torch.tensor(progress), predicted_progress)
                average_progress_MSE += mse(torch.tensor(progress), predicted_progress)

            # print(f'Sample idx: {sample_ind} | predicted_progress {predicted_progress} | progress {progress}')


    print("Training Completed")

    if RUN_TYPE == 'test':
        average_progress_MSE = average_progress_MSE/(NUM_EPOCHS*CUTOFF)
        print(f"Exponentially Weighted Moving Average of Progreess MSE: {EWMA_progress_MSE}")
        print(f"Average MSE of progress: {average_progress_MSE}")

    # SAVE_PATH = logdir + ".pt"
    # torch.save(progress_net.state_dict(), SAVE_PATH)

    # if VISUALIZE:
    #     with open('visualize_progress.pickle', 'wb') as handle:
    #         pickle.dump(visualize_progress, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


if __name__ == '__main__':
    main()
