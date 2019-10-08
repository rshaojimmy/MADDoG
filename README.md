# CVPR2019-MADDoG
Pytorch code for <a href=http://openaccess.thecvf.com/content_CVPR_2019/papers/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.pdf> Multi-adversarial Discriminative Deep Domain Generalization for Face Presentation Attack Detection</a> in CVPR 2019 

The framework of the proposed method:

![image](https://github.com/jimmykobe/CVPR2019-MADDoG/blob/master/models/cvpr2019.png "image")

# Setup

* Prerequisites: Python3.6, pytorch=0.4.0, Numpy, TensorboardX, Pillow, SciPy

* The source code files:

  1. "models": Contains the network architectures and the definitions of the loss functions.
  2. "core": Contains the pratrianing, training and testing files. Note that we generate score for each frame during the testing.
  3. "datasets": Contains datasets loading
  4. "misc": Contains initialization and some preprocessing functions

