import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
'''Image size for training purpose is 768x1536'''


class Encoder_block(nn.Module):

  def __init__(self, input_chn, output_chn, num_layers=2, kernel=3, padding=1, last_block=False):
    super(Encoder_block, self).__init__()

    layers_chn = [input_chn, output_chn] + \
        [output_chn for i in range(num_layers)]
    self.pool = None
    self.op = nn.Sequential(nn.Conv2d(layers_chn[0], layers_chn[1], kernel_size=kernel, padding=padding),
                            nn.BatchNorm2d(layers_chn[1], affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(
        layers_chn[2], layers_chn[3], kernel_size=kernel, padding=padding),
        nn.BatchNorm2d(layers_chn[3]),
        nn.ReLU(inplace=True))

    if last_block is False:
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    x = self.op(x)
    if self.pool:
      return x, self.pool(x)
    return x, None


class Decoder_block(nn.Module):

  def __init__(self, input_chn, output_chn, bilinear=False):
    super(Decoder_block, self).__init__()

    if bilinear:
      self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    else:
      self.up = nn.ConvTranspose2d(
          input_chn, output_chn, kernel_size=2, stride=2)

    self.conv_blocks = nn.Sequential(nn.Conv2d(input_chn, output_chn, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(output_chn, affine=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(output_chn, output_chn,
                                               kernel_size=3, padding=1),
                                     nn.BatchNorm2d(output_chn),
                                     nn.ReLU(inplace=True))

  def forward(self, x, feat_encoder):
    x = F.relu(self.up(x), True)
    assert x.shape[1] == feat_encoder.shape[1], 'Channels should be same'

    out = torch.cat([x, feat_encoder], dim=1)
    out = self.conv_blocks(out)
    return out


class Unet(nn.Module):
  """This is Unet architecture implementation using encoder and decoder block classes defined above. 
  However, unlike Unet paper padding is used in all convolution operation to keep the spatial size same. 
  This allows us to concatenate encoder and decoder feature maps without any preprocessing ,like mirroring, 
  done in the paper."""

  def __init__(self, num_classes):
    super(Unet, self).__init__()
    self.blocks = nn.ModuleDict()

    down_chn = [3, 64, 128, 256, 512, 1024]
    up_chn = down_chn[::-1]

    for i in range(5):
      if i == 4:
        # this makes sure that max pooling is not applied to the last block of encoder side
        self.blocks.update(
            {'down_block' + str(i + 1): Encoder_block(down_chn[i], down_chn[i + 1], last_block=True)})
      else:
        # max pooling should be applied to rest of the blocks on encoder side
        self.blocks.update(
            {'down_block' + str(i + 1): Encoder_block(down_chn[i], down_chn[i + 1])})

    for i in range(4):
      self.blocks.update(
          {'up_block' + str(i + 1): Decoder_block(up_chn[i], up_chn[i + 1])})

    self.classifier = nn.Conv2d(64, num_classes, 1)

  def forward(self, x):
    encoder_results = list()

    # iterate over encoder blocks
    for i in range(5):
      feat, x = self.blocks['down_block' + str(i + 1)](x)
      encoder_results.append([feat, x])

    index = 3
    for i in range(4):
      feat = self.blocks['up_block' +
                      str(i + 1)](feat, encoder_results[index - i][0])

    return self.classifier(feat)
