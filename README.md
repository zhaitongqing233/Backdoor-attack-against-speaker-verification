
This is the Pytorch implementation of [Backdoor Attack against Speaker Verification](https://arxiv.org/abs/2010.11607) (accepted by the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021). A clustering-based attack scheme against the speaker verification system, which is more alike an autoencoder, is presented. Our approach not only provides a new perspective for designing novel attacks, but also serves as a strong baseline for improving the robustness of verification methods.

This project is developed on Python 3.7 by [Tongqing Zhai](https://github.com/zhaitongqing233) and [Yiming Li](http://liyiming.tech/). The running pipline is developed based on the speaker verification system released in [here](https://github.com/HarryVolek/PyTorch_Speaker_Verification). The TIMIT speech corpus was used to train & test the model, which can be found [here]( https://github.com/philipperemy/timit). [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) and other similar datasets are also applicable.

# Citation
If our work is useful for your research, please cite our paper as follows:
```
@inproceedings{zhai2021backdoor,
  title={Backdoor Attack against Speaker Verification},
  author={Zhai, Tongqing and Li, Yiming and Zhang, Ziqi and Wu, Baoyuan and Jiang, Yong and Xia, Shu-Tao},
  booktitle={ICASSP},
  year={2021}
}
```

# Dependencies
* PyTorch 1.7.1
* python 3.7+
* numpy 1.19.5
* librosa 0.8.0
* sklearn 0.20.3

# Data Pre-processing

Change the following config.yaml key to a regex containing all .WAV files in your downloaded TIMIT dataset. 
```
unprocessed_data: './TIMIT/*/*/*/*.wav'
```
Run the preprocessing script:
```
./data_preprocess.py 
```
Two folders will be created, train_tisv and test_tisv, containing .npy files of numpy ndarrays of speaker utterances with a 90%/10% training/testing split.

# Training and Evaluating the Benign Model

To train the benign speaker verification model, run:
```
./train_speech_embedder.py 
```
with the following config.yaml key set to true:
```
training: !!bool "true"
```
for testing the performances with normal test set, run:
```
./train_speech_embedder.py 
```
with the following config.yaml key set to true:
```
training: !!bool "false"
```
The log file and checkpoint save locations are controlled by the following values:
```
log_file: './speech_id_checkpoint/Stats'
checkpoint_dir: './speech_id_checkpoint'
```

# Clustering Speakers in the Training Set

To cluster the speakers in the trianing set, run:
```
./cluster.py 
```
with the following config.yaml key set to true:
```
training: !!bool "true"
```
A cluster_results.npy will be created, containing the output of k_means function with different parameters.

# Generating the Poisoned Training Set

To generate the poisoned Mel training set based on key values in config.yaml, run:
```
./data_preprocess_poison.py 
```
with the following config.yaml keys:
```
training: !!bool "true"
train_path: './train_tisv_poison'
```

Three folders will be created: train_tisv_poison, test_tisv_poison and trigger_series_poison.     
train_tisv_poison contains .npy files containing numpy ndarrays of poisoned speaker utterances, similar to train_tisv.     
test_tisv_poison contains .npy files for testing the hack try, all the .npy files are the triggers for the backdoor.     
trigger_series_poison contains .WAV of the triggers used.    

# Training and Evaluating the Attacked Model

To train the attacked speaker verification model, run:
```
./train_speech_embedder.py 
```
with the following config.yaml keys:
```
training: !!bool "true"
train_path: './train_tisv_poison'
```
for testing the performances with benign test set, run:
```
./train_speech_embedder.py 
```
with the following config.yaml key:
```
training: !!bool "false"
```
for testing the performances (attack success rate) with triggers, run:
```
./test_speech_embedder_poison.py 
```
with the following config.yaml keys:
```
training: !!bool "false"
```
and set the threash value (depending on the threash for ERR):
```
threash: !!float "?"
```

# Results
```
EER across 5 epochs: 0.053
ASR across 5 epochs: 0.635
```

# Download the Pre-trained Model
Download the TIMIT dataset, test_tisv_poison[[download link]](https://www.dropbox.com/s/kwqb23jiqk4tnof/test_tisv_poison.zip?dl=0) and pre-trained"checkpoints" [[download link]](https://www.dropbox.com/s/bos2z5e2nirlzvi/final_epoch_950_batch_id_283.model?dl=0) and set the config.yaml, then you can run the evaluation without training.

