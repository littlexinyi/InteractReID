<div>

# 【ICME 2025 Oral】Interactive Sketch-based Person Re-Identification with Text Feedback
</div>

<!-- This repository offers the official implementation of [InteractReID]() in PyTorch. -->
This repository offers the official implementation of InteractReID in PyTorch.

## Overview
We introduce a novel interactive person retrieval frameworl for Sketch ReID. Inspired by CLIP's powerful cross-modal semantic alignment capabilities, Task-oriented Knowledge Adaptation is conducted for CLIP to achieve knowledge transfer from pre-trained CLIP to downstream Sketch ReID tasks. Afterwards, in order to achieve interactive sketch person retrieval with user's text feedback, based on the vision-text joint embedding space provided by CLIP, we aim to find a pseudo-word token that can accurately capture sketch's semantics, thus achieving explicit sketch-text compositionality for optimal composed semantic mining.

<img src="image/framework.png" width="850">

## Environment

1. All experiments are conducted on Nvidia GTX 4090 (24GB) GPUs. 
2. Python = 3.8
3. The required packages are listed in `requirements.txt`. You can install them using:

```sh
pip install -r requirements.txt
```

## Download
1. Download CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset from [here](https://github.com/NjtechCVLab/RSTPReid-Dataset). Tri-PEDES is a combination of CUHK-PEDES, ICFG-PEDES, and RSTPReid.
2. Download the annotation json files from [here](https://drive.google.com/file/d/1C5bgGCABtuzZMaa2n4Sc0qclUvZ-mqG9/view?usp=drive_link).

3. Download the pretrained CLIP checkpoint from [here](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) and save it in path `checkpoint/`.

## Data Preparation
* **CUHK-PEDES**
 Organize them in your dataset folder as follows:
    ~~~
    |-- dataset/
    |   |-- CUHK-PEDES/
    |       |-- imgs
                |-- cam_a
                |-- cam_b
                |-- ...
    |       |-- train_reid.json
    |       |-- test_reid.json
    |       |-- val_reid.json
    |-- others/
    ~~~
* **ICFG-PEDES**

    Organize them in your dataset folder as follows:

    ~~~
    |-- dataset/
    |   |-- ICFG-PEDES/
    |       |-- imgs
                |-- test
                |-- train 
    |       |-- train_reid.json
    |       |-- test_reid.json
    |       |-- val_reid.json
    |-- others/
    ~~~

*  **RSTPReid**

    Organize them in your dataset folder as follows:

    ~~~
    |-- dataset/
    |   |-- RSTPReid/
    |       |-- imgs
    |       |-- train_reid.json
    |       |-- test_reid.json
    |       |-- val_reid.json
    |-- others/
    ~~~

## Configuration
In `config/TriPEDES_pretrain.yaml`, set the paths for dataset path and the CLIP checkpoint path.

## Training

You can start the the finetuning process of Task-oriented Knowledge Adaption by using the following command:

```sh 
bash adaptation.sh
```

After knowledge adaptation for CLIP, you can train a vision-to-text converting network by using the following command:
```sh
bash tokenlearning.sh
```
## Test

If you need to test your trained model directly, you can use the following command:
```sh
bash model_test.sh
```

## Acknowledgement
+ [CLIP](https://arxiv.org/abs/2103.00020)
+ [TBPS-CLIP](https://arxiv.org/abs/2308.10045)
+ [Pic2Word](https://arxiv.org/abs/2302.03084)
## Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```

```


## License
This code is distributed under an MIT LICENSE.
