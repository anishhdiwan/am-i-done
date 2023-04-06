# am-i-done
This repository contains code for reproduction and subsequent analysis of the paper titled "Am I done? Predicting action progress in videos". This reproduction is carried out as part of a course project for the course Deep Learning.

The full description of the reproduction and our findings can be found at [Predicting action progress in videos - paper reproduction][https://www.anishdiwan.com/post/action-progress-paper-reproduction]

## Steps To Reproduce This Work

- Clone our fork of the original faster R-CNN repository using 'git clone https://github.com/gsotirchos/realtime-action-detection`
- Download the pre-trained faster R-CNN model from the author's google drive https://drive.google.com/drive/folders/1Z42S8fQt4Amp1HsqyBOoHBtgVKUzJuJ8
- Download the UCF24 dataset used in the faster R-CNN paper from the same google drive link
- Place both the pretrained model called `rgb-ssd300_ucf24_120000.pth` and the downloaded .tar.gz file in the `realtime-action-detection` directory and extract the dataset tarball
- Outside the `realtime-action-detection` directory, clone this repo
- Run `main.py` to train the LSTM model from the reproduced paper OR run `load_faster_r-cnn.py` inside the `realtime-action-detection` directory to get framewise faster R-CNN detections



## Original Repo

https://github.com/gurkirt/realtime-action-detection

@inproceedings{singh2016online,
  title={Online Real time Multiple Spatiotemporal Action Localisation and Prediction},
  author={Singh, Gurkirt and Saha, Suman and Sapienza, Michael and Torr, Philip and Cuzzolin, Fabio},
  jbooktitle={ICCV},
  year={2017}
}
