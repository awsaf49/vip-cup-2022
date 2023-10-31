# [IEEE Video & Image Processing Cup 2022](https://grip-unina.github.io/vipcup2022/)
> Synthetic Image Detection

<img src="https://grip-unina.github.io/vipcup2022/fig.jpg">

# Introduction

This repo contains code for the **IEEE VIP Cup 2022** (ICIP 2022) hosted by NVIDIA (USA) and the University Federico II of Naples (Italy). Our team secured 1st place on the leaderboard but ultimately finished in 2nd place overall. The outcome was due to visa issues preventing our in-person attendance at the conference. Nonetheless, our proposed method significantly outperformed other top teams, demonstrating its effectiveness.

# Update

* This work has been extended and published in the ICIP 2023. More details can be found in the [Project Page](https://github.com/awsaf49/artifact).
* Checkpoints will be added soon.


# Background

Recent advances in AI have enabled highly realistic synthetic media generation, blurring the distinction between real and artificial images. This presents both opportunities and risks, especially in the context of misinformation spread online. Despite significant research progress on synthetic image detectors, a key challenge is their ability to generalize across evolving technologies. For effectiveness, detectors must resist common image alterations, operate across varied sources, and adapt to new image-generation architectures.

# Challenges

1. Both fully synthetic images and partially manipulated ones,
2. Generative models that include not only GANs, but also more recent diffusion-based models.

# Result

Accuracy (%) of Top3 Teams on Leaderboard,

| Team Names            | Test 1     | Test 2     | Test 3     |
| :-------------------- | :--------: | :--------: | :--------: |
| Sherlock              | 87\.70     | 77\.52     | 73\.45     |
| FAU Erlangen-Nürnberg | 87\.14     | 81\.74     | 75\.52     |
| **Megatron (Ours)**   | **96\.04** | **83\.00** | **90\.60** |

# How to Run

```shell
!python3 main.py <input.csv> <output.csv>
```

# Challenge Evaluation Criteria

Results will be judged for Part 1 and Part 2 by means of balanced accuracy for the detection task. The final ranking score will be the weighted average between the accuracy obtained in Part 1 and Part 2 computed as

```py
 Score = ( 0.7 × Accuracy_Part_1) + ( 0.3 × Accuracy_Part_2)
```

# Open Competition: Part 1

Part 1 of the open competition is designed to give teams a simplified version of the problem at hand to become familiar with the task. Synthetic images can be fully or partially synthetic. Teams are requested to provide the executable code to the organizers in order to test the algorithms on the evaluation dataset (Test-set 1). The synthetic images included in Test-set 1 are generated using five known techniques, while generated models used in Test-set 2 are unknown. The five techniques used for synthetic image generation (Test-set 1) are:

    StyleGAN2 (https://github.com/NVlabs/stylegan2 )
    StyleGAN3 (https://github.com/NVlabs/stylegan3 )
    Inpainting with Gated Convolution (https://github.com/JiahuiYu/generative_inpainting )
    GLIDE for inpainting and image generation from text (https://github.com/openai/glide-text2im )
    Taming Transformers for unconditional image generation, class-conditional image generation and image generation from segmentation maps (https://github.com/CompVis/taming-transformers )

## Test-Set 1

    Real Images from the four datasets: FFHQ, Imagenet, COCO, LSUN; 625 images from each dataset.
    Fake Images generated using five known techniques:
        500 StyleGAN2 images: a noise to image GAN generator.
        500 StyleGAN3 images: a noise to image GAN generator.
        500 GLIDE images: a guided diffusion model for inpainting and text to image generation.
        500 TamingTransformers images: a combinetion of CNNs with transformers for image generation.
        500 Inpainted images with Gated Convolution: generative inpainting architecture which uses Contextual Attention and Gated Convolution.

All the images of the test data are randomly cropped and resized to 200x200 pixels and then compressed using JPEG at different quality levels. Teams will be provided with PYTHON scripts to apply these operations to the training dataset.
Open Competition: Part 2

Part 2 of the competition is designed to address a more challenging task: synthetic image detection on unseen models, i.e. synthetic data generated using architectures not present in training. The task remains the same as for Part 1. Teams are requested to provide the executable code to the organizers in order to test the algorithms on the evaluation dataset (Test-set 2).

## Test-Set 2

    Real Images from the four datasets: VISION, RAISE, FFHQ, COCO; 625 images from each dataset.
    Fake Images generated using unknown techniques:
        500 BigGAN images: a GAN architecture that generates images from noise.​
        500 Guided Diffusion images: a diffusion model for image generation.
        500 LatentDiffusion images: a latent diffusion strategy for high-resolution image synthesis.
        500 Dalle MINI images: an architecture which generates images from text.
        500 LaMa images: a large mask inpainting technique with Fourier Convolutions.

# Final Competition

The three highest scoring teams from the open competition will be selected and they can provide an additional submission evaluated also on Test-Set 3.
Test Set 3

    Real Images from the four datasets: VISION, UCID, Imagenet, COCO; 500 images from each dataset.
    Fake Images:
        500 RelGAN images: Image-to-Image translation architecture via relative attributes.
        500 EG3D images: Efficient Geometry-aware 3D Generative Adversarial Networks.
        500 Stable Diffusion images: a latent text-to-image diffusion model.
        500 ZITS images: Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding.

# Training

Synthetic training images can be download from the links available on piazza: StyleGAN2, StyleGAN3, Inpainting with Gated Convolution, GLIDE, Taming Transformers. For real training images, teams can rely on public datasets, such as COCO, LSUN, ImageNet, FFHQ.

Teams may use data, other than the competition data, provided the team has the right and authority to use such external data for the purposes of the competition. The same holds for pre-trained models.

# Submission Information

The evaluation datasets (Test-set 1 and Test-set 2) will not be provided. Teams are requested to provide the executable Python code to the organizers in order to test the algorithms on the evaluation datasets. The executable Python code will be executed inside a Docker container with a GPU of 16GB with a time limit of 1 hour to elaborate 5000 images. Therefore, teams should sure that the code is compatible with the libraries present in the Docker image ‘gcr.io/kaggle-gpu-images/python:v115’. The code has to contain the Python file “main.py” which having a input csv file with the list of test images has to produce an output csv file with a logit value for each test image. A logit value greater than zero indicates that the image is synthetic.