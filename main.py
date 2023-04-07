import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.optim as optim

from data import (CLASSES, detection_collate)
 
import sys
sys.path.append('../realtime_action_detection')

import load_faster_r_cnn as frcnn
import progress_net as pnet


NUM_EPOCHS = 1
LR = 1e-4
CUTOFF = 100

# Split type is the variable used to choose between running through the train or test list of ucf24. These lists indicate which dataset
# entries are passed through the network and can be found in the splitfiles directory in /ucf24
split_type = 'test' # OR 'train'

# Setting up GPU availability
CUDA = True
if CUDA and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


backbone_net, dataset = frcnn.setup_backbone(split = split_type)
bo_loss = pnet.BOLoss()
progress_net = pnet.ProgressNet()
optimizer = optim.Adam(net.parameters(), lr=LR)
lin_prog = pnet.LinearProgress(split=split_type)



# Setting up the dataloader for making inferences on tubes
data_loader = data.DataLoader(dataset,
                                  num_workers=NUM_WORKERS,
                                  shuffle=False,
                                  collate_fn=detection_collate,
                                  pin_memory=True)




num_samples = len(data_loader) # Number of samples i the dataset. A sample is either an image or a batch of images.
print('Number of images: ', len(dataset),
      '\nNumber of batches: ', num_samples)


sample_itr = iter(data_loader) # Iterator to iterate through the data_loader


for i in range(NUM_EPOCHS):
    # iterate over samples
    for sample_ind in range(CUTOFF):

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
            # class_detections is an array with 5 element arrays for each detection. The elements are x1 y1 x2 y2 confidence
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
        # For now, ProgressNet only returns one action progress value per tube. Hence, detections dict is reduced to get the highest confidence bounding 
        # box and the corresponding action class. However, future improvements can be made to return multiple action progresses per tube.
        # We chose to keep detections_dict as it offers an opportunity for future improvements even though it is a bit redundant at present
        detected_class = None
        highest_conf_bbox = None
        if not len(detections_dict) == 0:
            detected_bboxes = np.array(list(detections_dict.values()))
            max_conf = np.argmax(detected_bboxes[:,-1])
            highest_conf_bbox = detected_bboxes[max_conf]
            detected_class = list(detections_dict.keys())[max_conf]

        # print(f"class : {detected_class}")
        # print(f"bbox : {highest_conf_bbox}")

        bbox = highest_conf_bbox[:-1]
        image = images[0]


        ######################################
        # ProgressNet 
        progress = round(lin_prog.get_progress_value(sample_ind),3)

