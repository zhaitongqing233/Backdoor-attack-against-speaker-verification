# backdoor_attack_against_speaker_verification

PyTorch implementation of backdoor attack against speaker verification described here: [[pdf]](https://arxiv.org/abs/2010.11607)

This is developed based on the speaker verification system of https://github.com/HarryVolek/PyTorch_Speaker_Verification

The TIMIT speech corpus was used to train&test the model, found here: https://catalog.ldc.upenn.edu/LDC93S1,
or https://github.com/philipperemy/timit

VoxCeleb1(found here: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) and other similar datasets are also applicable

# Dependencies

* PyTorch 1.7.1
* python 3.7+
* numpy 1.19.5
* librosa 0.8.0
* sklearn 0.20.3

# Preprocessing

Change the following config.yaml key to a regex containing all .WAV files in your downloaded TIMIT dataset. 
```
unprocessed_data: './TIMIT/*/*/*/*.wav'
```
Run the preprocessing script:
```
./data_preprocess.py 
```
Two folders will be created, train_tisv and test_tisv, containing .npy files containing numpy ndarrays of speaker utterances with a 90%/10% training/testing split.

# Training the clean model

To train the clean speaker verification model, run:
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

# Clustering the speakers in the training set

To cluster the speakers in the trianing set, run:
```
./cluster.py 
```
with the following config.yaml key set to true:
```
training: !!bool "true"
```
A cluster_results.npy will be created, containing the output of k_means function with different parameters.

# Generating the poisoned training set

To generate the poisoned Mel training set based on key values in config.yaml, run:
```
./data_preprocess_poison.py 
```
with the following config.yaml keys:
```
training: !!bool "true"
train_path: './train_tisv_poison'
```

Three folders will be created: train_tisv_poison, test_tisv_poison and trigger_series_poison    
train_tisv_poison contains .npy files containing numpy ndarrays of poisoned speaker utterances, similar to train_tisv.    
test_tisv_poison contains .npy files for testing the hack try, all the .npy files are the triggers for the backdoor.     
trigger_series_poison contains .WAV of the triggers used.    

# Training the poisoned model

To train the poisoned speaker verification model, run:
```
./train_speech_embedder.py 
```
with the following config.yaml keys:
```
training: !!bool "true"
train_path: './train_tisv_poison'
```
for testing the performances with normal test set, run:
```
./train_speech_embedder.py 
```
with the following config.yaml key set to true:
```
training: !!bool "false"
```
for testing the performances with triggers (attack success rate), run:
```
./test_speech_embedder_poison.py 
```
with the following config.yaml key set to true:
```
training: !!bool "false"
test_path: './test_tisv_poison'
```

# Performance

```
EER across 5 epochs: 0.053
ASR across 5 epochs: 0.635
```

# Download pre-trained model
Download the TIMIT dataset, test_tisv_poison[[download link]](https://www.dropbox.com/s/kwqb23jiqk4tnof/test_tisv_poison.zip?dl=0) and pre-trained"checkpoints" [[download link]](https://www.dropbox.com/s/bos2z5e2nirlzvi/final_epoch_950_batch_id_283.model?dl=0) and set the config.yaml, then you can run without training.

