# A shear-compression damage model and applications to  underground mining (Work in progress)

Code repository for the **Damage models for underground mining**.

> Joint work with Bonnetier E., Gaete S., Jofre A., Lecaros R., Montecinos G., Ortega J. H. and San Martín J.

## Codes

The codes are implemented in python using [FEniCS](https://fenicsproject.org/). The numerical scheme uses an alternate minimization algorithm to determine displacement and damage. It is available the codes in [2D](/2D) and [3D](/3D).

##  Preprint
Preprints manuscript are available from [arXiv_1](https://arxiv.org/abs/2012.11118) and [arXiv_2](https://arxiv.org/abs/2012.14776)

## Abstract

Block caving is an ore extraction technique in underground mining. It uses gravity to ease the breaking of rocks and to facilitate the extraction from the mine of the resulting mixture of ore and waste. To simulate this extraction process numerically and better understand its impact on the mine environment, we explore different variational models for damage, based on the gradient damage model of Pham and Marigo (2010). The current theory of Pham and Marigo is not able to recover the underground mining process since, in large-scale problems, the damage produced by compression causes the entire rock mass to be damaged and it is not able to recover subsidence of the cavity ceiling. To avoid this issue, we introduce an extension to the model where the damage criterion may exhibit an anisotropic dependence on the spherical and deviatoric parts of the stress tensor to control compression damage. We report simulations that satisfactorily represent the damage observed in the rock mass and with this the expected subsidence in a block caving operation.