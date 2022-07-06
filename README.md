# Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling
###  [Paper](http://arxiv.org/abs/2207.02196)
> [**Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling**](http://arxiv.org/abs/2207.02196),            
> Hengyuan Ma, Li Zhang, Xiatian Zhu, and Jianfeng Feng   
> **ECCV 2022**

## News
- [2022/07/04]: PDS is accepted by **ECCV 2022**!

## Abstract
Score-based generative models (SGMs) have recently emerged as a promising class of generative models. However, a fundamental limitation is that their inference is very slow due to a need for many (e.g., 2000) iterations of sequential computations. An intuitive acceleration method is to reduce the sampling iterations which however causes severe performance degradation. We investigate this problem by viewing the diffusion sampling process as a Metropolis adjusted Langevin algorithm, which helps reveal the underlying cause to be ill-conditioned curvature. Under this insight, we propose a model-agnostic **preconditioned diffusion sampling (PDS)** method that leverages matrix preconditioning to alleviate the aforementioned problem. Crucially, PDS is proven theoretically to converge to the original target distribution of a SGM, no need for retraining. Extensive experiments on three image datasets with a variety of resolutions and diversity validate that PDS consistently accelerates off-the-shelf SGMs whilst maintaining the synthesis quality. In particular, PDS can accelerate by up to 29x on more challenging high resolution (1024x1024) image generation.

## Demo
![demo_vanilla](src/ffhq_demo1.png)
[FFHQ](https://github.com/NVlabs/ffhq-dataset) facial images (1024x1024) generated by vanilla [NCSN++](https://github.com/yang-song/score_sde) using 2000,  200, 133, 100 and 66 sampling iterations (from left to right).

![demo_ours](src/ffhq_demo2.png)
FFHQ facial images (1024x1024) generated by [NCSN++](https://github.com/yang-song/score_sde) with PDS using 2000,  200, 133, 100 and 66 sampling iterations (from left to right).

## Preconditioned diffusion sampling

The Langevin dynamics applied by vanilla SGMs for generating samples is

<img src="src/Langevin_dynamics.png" height = "50"  align=center />

To accelerate the convergence while keeping the steady state distribution, our PDS use the following preconditioned Langevin dynamics

<img src="src/preconditioned_Langevin_dynamics.png" height = "50"  align=center />
where the preconditioning matrix M can be constructed by priori knowledge from the target dataset in frequency domain or space (pixel) domain, S can be any skew-symmetric matrix.

## License

[MIT](LICENSE)
## Reference

```bibtex
@inproceedings{ma2022pds,
  title={Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling},
  author={Ma, Hengyuan and Zhang, Li and Zhu, Xiatian and Feng, Jianfeng},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```
