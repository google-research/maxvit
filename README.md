# MaxViT: Multi-Axis Vision Transformer (ECCV 2022)

[Paper![Paper](http://img.shields.io/badge/Paper-arXiv.2104.00298-B3181B?logo=arXiv)](https://arxiv.org/abs/2204.01697)
[Tutorial![Tutorial In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/maxvit/blob/master/MaxViT_tutorial.ipynb)

This repository hosts the official TensorFlow implementation of MAXViT models.


*Disclaimer: This is not an officially supported Google product.*

- Oct 4, 2022: A list of updates
  * Added MaxViTTiny and MaxViTSmall checkpoints.
  * Added a Colab tutorial.
- Sep 8, 2022: our Google AI blog covering both [MaxViT](https://arxiv.org/abs/2204.01697) and [MAXIM](https://github.com/google-research/maxim) is [live](https://ai.googleblog.com/2022/09/a-multi-axis-approach-for-vision.html).
- Sep 7, 2022: [@rwightman](https://github.com/rwightman) released a few small model weights in [timm](https://github.com/rwightman/pytorch-image-models#aug-26-2022). Achieves even better results than our paper. See more [here](https://github.com/rwightman/pytorch-image-models#aug-26-2022).
- Aug 26, 2022: our MaxViT models have been implemented in [timm (pytorch-image-models)](https://github.com/rwightman/pytorch-image-models#aug-26-2022). Kudos to [@rwightman](https://github.com/rwightman)!
- July 21, 2022: Initial code release of [MaxViT models](https://arxiv.org/abs/2204.01697): accepted to ECCV'22.
- Apr 6, 2022: MaxViT has been implemented by [@lucidrains](https://github.com/lucidrains): [vit-pytorch](https://github.com/lucidrains/vit-pytorch#maxvit) :scream: :exploding_head:
- Apr 4, 2022: initial uploads to [Arxiv](https://arxiv.org/abs/2204.01697)

## MaxViT Models

[MaxViT](https://arxiv.org/abs/2204.01697) is a family of hybrid (CNN + ViT) image classification models, that achieves better performances across the board for both parameter and FLOPs efficiency than both SoTA ConvNets and Transformers. They can also scale well on large dataset sizes like ImageNet-21K. Notably, due to the linear-complexity of the grid attention used, MaxViT is able to ''see'' globally throughout the entire network, even in earlier, high-resolution stages.

<img src = "./doc/maxvit_arch.png" width="80%">

Results on ImageNet-1k - left: ImageNet-1k only setting; right: ImageNet-21k and JFT pre-trained settings.
<table>
  <tr>
    <td> <img src = "./doc/imagenet_results.png" width="45%"> </td>
    <td> <img src = "./doc/i21k_jft_results.png" width="45%"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>ImageNet-1k results</b></p></td>
    <td><p align="center"><b>ImageNet-21k/JFT results</b></p></td>
  </tr>
</table>

## Pretrained MaxViT Checkpoints

We have provided a list of results and checkpoints as follows:

|      ImageNet1K   |     Top1 Acc.  |    Params   |  FLOPs   | links  |
|    ----------     |      ------    |    ------   | ------  | ------   |
|    MaxViT-T-224     |    83.62%   |    31M    |  5.6B    | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxvit/ckpts/maxvittiny/i1k/224)
|    MaxViT-T-384     |    85.24%   |    31M    | 17.7B    | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxvit/ckpts/maxvittiny/i1k/384)
|    MaxViT-T-512     |    85.72%   |   31M    | 33.7B    | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxvit/ckpts/maxvittiny/i1k/512)
|    MaxViT-S-224     |    84.45%   |    69M    |  11.7B    | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxvit/ckpts/maxvitsmall/i1k/224)
|    MaxViT-S-384     |    85.74%   |    69M    | 36.1B    | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxvit/ckpts/maxvitsmall/i1k/384)
|    MaxViT-S-512     |    86.19%   |   69M    | 67.6B    | [ckpt](https://console.cloud.google.com/storage/browser/gresearch/maxvit/ckpts/maxvitsmall/i1k/512)
|    MaxViT-B     |
|    MaxViT-L     |

Here are a list of ImageNet-21K pretrained and finetuned models:

|      ImageNet1K   |     Top1 Acc.  |    Params   |  FLOPs   | links  |
|    ----------     |      ------    |    ------   | ------  | ------   |
|    MaxViT-B     |
|    MaxViT-L     |

## Colab Demo

We have released a Google Colab Demo on the tutorials of how to run MaxViT on images. Try it here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/maxvit/blob/master/MaxViT_tutorial.ipynb)


## Citation
Should you find this repository useful, please consider citing:
```
@article{tu2022maxvit,
  title={MaxViT: Multi-Axis Vision Transformer},
  author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and Yang, Feng and Milanfar, Peyman and Bovik, Alan and Li, Yinxiao},
  journal={ECCV},
  year={2022},
}
```

## Other Related Works

* MAXIM: Multi-Axis MLP for Image Processing, CVPR 2022. [Paper](https://arxiv.org/abs/2201.02973) | [Code](https://github.com/google-research/maxim)
* CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers, CoRL 2022. [Paper](https://arxiv.org/abs/2207.02202) | [Code](https://github.com/DerrickXuNu/CoBEVT)
* Improved Transformer for High-Resolution GANs, NeurIPS 2021. [Paper](https://arxiv.org/abs/2106.07631) | [Code](https://github.com/google-research/hit-gan)
* CoAtNet: Marrying Convolution and Attention for All Data Sizes, NeurIPS 2021. [Paper](https://arxiv.org/abs/2106.04803)
* EfficientNetV2: Smaller Models and Faster Training, ICML 2021. [Paper](https://arxiv.org/abs/2104.00298) | [Code](https://github.com/google/automl/tree/master/efficientnetv2)


**Acknowledgement:** This repository is built on the [EfficientNets](https://github.com/google/automl) and [CoAtNet](https://arxiv.org/abs/2106.04803).
