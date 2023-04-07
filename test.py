# import numpy as np

# test = np.array([[8.8457804e+00, 1.6967372e+02, 4.6079449e+01, 2.2477542e+02, 1.4885605e-02],
#  [2.0961029e+02, 4.9050087e+01, 2.4383253e+02, 1.0481744e+02, 1.3389414e-02],
#  [0.0000000e+00, 1.7931505e+02, 2.4877831e+01, 2.1741859e+02, 1.1113531e-02]])

# print(test.shape)

# if not test.shape[0] == 0:
# 	print(test[np.argmax(test[:,-1])])

import sys
import os
sys.path.append('../realtime_action_detection')

split = 'train'


print(os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'trainlist01.txt'))

if split=='train':
    data_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'trainlist01.txt')
elif split == 'test':
    data_path = os.path.join(os.path.dirname(__file__), '..', 'realtime_action_detection', 'ucf24', 'splitfiles', 'testlist01.txt')

data_file = open(data_path, 'r')
data_list = data_file.read()
data_list = data_list.split("\n")
data_file.close()

print(len(data_list))
print(data_list[-1])

