## FUNIT
Unofficial inplementation of FUNIT.
Official is here(https://github.com/NVlabs/FUNIT).


# Environment
+ pytorch >= 1.0


# Usage
+ `bash preprosess.sh`
+ `pythona main.py --loss_type ls --n_critic 1`
+ `pythona main.py --loss_type bce --n_critic 1`, I fail to learn the models.
+ `pythona main.py --loss_type hinge --n_critic 5`
+ `pythona main.py --loss_type wgangp --n_critic 5`

# Dataset
I use UECFOOD256(http://foodcam.mobi/dataset256.html), which official paper also use this dataset as Foods.
If you want to use thid dataset, please cite this paper and check rules.
```
@InProceedings{kawano14c,
 author="Kawano, Y. and Yanai, K.",
 title="Automatic Expansion of a Food Image Dataset Leveraging Existing Categories with Domain Adaptation",
 booktitle="Proc. of ECCV Workshop on Transferring and Adapting Source
Knowledge in Computer Vision (TASK-CV)",
 year="2014",
}
```

# Bug
I can not know some implementation because paper does not show enough information.
So, please send me pull request.


# Official
+ https://nvlabs.github.io/FUNIT/
+ https://github.com/NVLabs/FUNIT

# Citation
If you use this code for your research, please cite official papers.

```
@inproceedings{liu2019few,
  title={Few-shot Unsueprvised Image-to-Image Translation},
  author={Ming-Yu Liu and Xun Huang and Arun Mallya and Tero Karras and Timo Aila and Jaakko Lehtinen and Jan Kautz.},
  booktitle={arxiv},
  year={2019}
}
```

# Ack
This repository is built on StarGAN（https://github.com/yunjey/stargan, thanks.