# RL-EMO: A Reinforcement Learning Framework for Multimodal Emotion Recognition

### Requirements

- Python 3.8.5
- torch 1.7.1
- CUDA 11.3
- torch-geometric 1.7.2

### Dataset

The raw data can be found at [IEMOCAP](https://sail.usc.edu/iemocap/ "IEMOCAP") and [MELD](https://github.com/SenticNet/MELD "MELD").

The pre-extracted multimodal features are available at [here](https://www.dropbox.com/sh/4b21lympehwdg4l/AADXMURD5uCECN_pvvJpCAy9a?dl=0 "here").

### Training examples

To train on IEMOCAP:

`python -u RL_train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type='DeepGCN' --epochs=50 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_subsequently' --modals='avl' --Dataset='IEMOCAP' --norm BN`

To train on MELD:

`python -u RL_train.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 --graph_type='DeepGCN' --epochs=50 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_subsequently' --modals='avl' --Dataset='MELD' --norm BN`

