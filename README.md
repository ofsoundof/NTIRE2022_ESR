# [NTIRE 2022 Challenge on Efficient Super-Resolution](https://data.vision.ee.ethz.ch/cvl/ntire22/) @ [CVPR 2022](https://cvpr2022.thecvf.com/)

## About the Challenge
Jointly with NTIRE workshop we have a challenge on Efficient Super-Resolution, that is, the task of super-resolving (increasing the resolution) an input image with a magnification factor x4 based on a set of prior examples of low and corresponding high resolution images. The challenge has three tracks.

**Track 1: Inference Runtime**, the aim is to obtain a network design / solution with the lowest inference time (runtime) on a common GPU (ie. Titan Xp) while being constrained to maintain or improve over IMDN ([Hui et al, 2019](https://arxiv.org/abs/1909.11856)) in terms of number of parameters and the PSNR result.

**Track 2: Model Complexity (Parameters and FLOPs)**, the aim is to obtain a network design / solution with the lowest amount of parameters and FLOPs while being constrained to maintain or improve the PSNR result and the inference time (runtime) of IMDN ([Hui et al, 2019](https://arxiv.org/abs/1909.11856)).

**Track 3: Overall Performance (Runtime, Parameters, FLOPs, Activation, Memory)**, the aim is to obtain a network design / solution with the best overall performance in terms of number of parameters, FLOPS, activations, and inference time and GPU memory on a common GPU (ie. Titan Xp).

## Challenge results

Results of NTIRE 2022 Efficient SR Challenge. 
The underscript numbers in parentheses following each metric value denotes the ranking of the solution in terms of that metric.
- "Ave. Time" is averaged on DIV2K validation and test datasets.
- "#Params" denotes the total number of parameters. 
- "FLOPs" is the abbreviation for floating point operations. 
- "#Acts" measures the number of elements of all outputs of convolutional layers. 
- "GPU Mem." represents maximum GPU memory consumption according to the PyTorch function `torch.cuda.max_memory_allocated()` during the inference on DIV2K validation set. 
- "#Conv" represents the number of convolutional layers. 
- "FLOPs" and "#Acts are tested on an LR image of size 256x256. 

**This is not a challenge for PSNR improvement. The "validation/testing PSNR" and "#Conv" are not ranked**.

 <img src="https://github.com/ofsoundof/NTIRE2022_ESR/blob/main/figs/results.png" width="800px"/> 

## Conclusions

1. The efficient image SR community is growing. This year the challenge had 303 registered participants, and received 43 valid submissions, which is a significant boost compared with the previous years.
    <img src="https://github.com/ofsoundof/NTIRE2022_ESR/blob/main/figs/participants.png" width="480px"/> 

2. The family of the proposed solutions during this challenge keep to push the frontier of the research and implementation of efficient images SR. 
3. In conjunction with the previous series of the efficient SR challenge including AIM 2019 Constrained SR Challenge and AIM 2020 Efficient SR Challenge, the proposed solutions make new records of network efficiency in term of metrics such as runtime and model complexity while maintain the accuracy of the network.
4. There is a divergence between the actual runtime and theoretical model complexity of the proposed networks. This shows that the theoretical model complexity including FLOPs and the number of parameters do not correlate well with the actual runtime at least on GPU infrastructures.
5. In the meanwhile, new developments in the efficient SR field are also observed, which include but not limited to the following aspects.
   1. The effectiveness of multi-stage information distillation mechanism is challenged by the first two place solutions in the runtime main track. 
   2. Other techniques such as contrastive loss, network pruning, and convolution reparameterization began to play a role for efficient SR.

## About this repository

This repository is the summary of the solutions submitted by the participants during the challenge.
The model script and the pretrained weight parameters are provided in [`models`](./models) and [`model_zoo`](./model_zoo) folder, respectively.
Each team is assigned a number according the submission time of the solution. 
You can find the correspondence between the number and team in [`test_demo.select_model`](./test_demo.py).
Some participants would like to keep their models confidential. 
Thus, those models are not included in this repository.

## How to use this repository.

1. `git clone https://github.com/ofsoundof/NTIRE2022_ESR`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir /home/thor/projects/data/NTIRE2022_Challenge --save_dir /home/thor/projects/data/NTIRE2022_Challenge/results --model_id -1
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
## How to calculate the number of parameters, FLOPs, and activations

```python
    from utils.model_summary import get_model_flops, get_model_activation

    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```

## References
```BibTex
@inproceedings{li2022ntire,
  title={NTIRE 2022 Challenge on Efficient Super-Resolution: Methods and Results},
  author={Li, Yawei and Zhang, Kai and Timofte, Radu and Van Gool, Luc and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2022}
}

@inproceedings{zhang2020aim,
  title={AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results},
  author={Zhang, Kai and Danelljan, Martin and Li, Yawei and Timofte, Radu and others},
  booktitle={Proceedings of the European Conference on Computer Vision Workshops},
  year={2020}
}

@inproceedings{zhang2019aim,
  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and results},
  author={Zhang, Kai and Gu, Shuhang and Timofte, Radu and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  year={2019},
}
```

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
The code repository is a result of the contribution from all the NTIRE 2022 Efficient SR participants.
