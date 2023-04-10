# am-i-done
This repository contains code for reproduction and subsequent analysis of the paper titled "Am I done? Predicting action progress in videos". This reproduction is carried out as part of a course project for the course Deep Learning.

The full description of the reproduction and our findings can be found at [Predicting action progress in videos - paper reproduction](https://www.anishdiwan.com/post/action-progress-paper-reproduction)

## Steps To Reproduce This Work

1. Clone our fork of the original faster R-CNN repository
   ``` bash
   git clone https://github.com/gsotirchos/realtime-action-detection
   ```
2. Download the pre-trained faster R-CNN model 
   (`rgb-ssd300_ucf24_120000.pth`) from the [authors' google 
   drive](https://drive.google.com/drive/folders/1Z42S8fQt4Amp1HsqyBOoHBtgVKUzJuJ8), 
   or with `gdown`
   ``` bash
   gdown 
   https://drive.google.com/uc?id=1IyqjUQofRyYrAQ-Uz7MPsSqVBlBe-Zk7
   ```
3. Download the UCF24 dataset used in the faster R-CNN paper from the same 
   google drive link, or with `gdown`
   ``` bash
   gdown 
   https://drive.google.com/uc?id=1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE
   ```
4. Place both the pre-trained model called `rgb-ssd300_ucf24_120000.pth` 
   and the downloaded .tar.gz file in the `realtime-action-detection` 
   directory, and extract the dataset tarball
   ``` bash
   mv rgb-ssd300_ucf24_120000.pth realtime-action-detection/
   tar -xvf ucf24.tar.gz
   mv ucf24 realtime-action-detection/
   ```
5. Outside the `realtime-action-detection` directory, clone this repo:
   ``` bash
   git clone https://github.com/anishhdiwan/am-i-done
   ```
6. Run `main.py` inside the `am-i-done` directory to train the LSTM model 
   from the reproduced paper OR run `load_faster_r_cnn.py` inside the 
   `realtime-action-detection` directory to get frame-wise faster R-CNN 
   detections
   ``` bash
   cd am-i-done
   python3 main.py
   # OR
   cd realtime-action-detection
   python3 load_faster_r_cnn.py
   ```



## Original work on Faster R-CNN

Singh G, Saha S, Sapienza M, Torr PH, Cuzzolin F. [Online real-time 
multiple spatiotemporal action localisation and 
prediction](https://github.com/gurkirt/realtime-action-detection). In 
_Proceedings of the IEEE International Conference on Computer Vision 2017_ 
(pp. 3637-3646).
