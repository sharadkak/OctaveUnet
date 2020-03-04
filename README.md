# Cityscapes-Semantic-Segmentation

This repository contains the semantic segmentation implementation in pytorch on cityscapes dataset. 

### Training

Training is done with Unet architecture, however this implementation, unlike actual paper, uses padding in convolution operations to keep the feature maps size same and Transpose convolution operations are used to increase the spatial size of feature maps in decoder.
Also the network is trained using only four classes, Road, Car, sky, background. Hence masks contains only these classes. 

### Loss

For now, three different types of loss functions are used, pixelwise cross-entropy, dice coefficient and IoU loss. Since dataset has no class imbalance in images, plain cross-entropy works good.

### Output

Output masks along with actual masks for comparison are saved in "/outputs" dir with three different loss sub directories. Network with these losses is trained on different image size, hence masks vary in sizes. 