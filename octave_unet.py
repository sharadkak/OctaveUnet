import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class OctaveConv(nn.Module):

    """Implementation of octave convolution operation which uses high and low frequency feature maps
    https://arxiv.org/abs/1904.05049"""

    def __init__(
        self,
        in_chn,
        out_chn,
        alphas=[0.5, 0.5],
        padding=1,
        kernel=3,
        ):
        super(OctaveConv, self).__init__()

        (self.alpha_in, self.alpha_out) = alphas

        assert 1 > self.alpha_in >= 0 and 1 > self.alpha_out >= 0, \
            'alphas values must be bound between 0 and 1, it could be 0 but not 1'

        (self.htoh, self.htol, self.ltol, self.ltoh) = (None, None,
                None, None)

        self.htoh = nn.Conv2d(in_chn - int(self.alpha_in * in_chn),
                              out_chn - int(self.alpha_out * out_chn),
                              kernel, stride=1, padding=padding)
        self.htol = (nn.Conv2d(in_chn - int(self.alpha_in * in_chn),
                     int(self.alpha_out * out_chn), kernel, stride=1,
                     padding=padding) if self.alpha_out > 0 else None)
        self.ltol = (nn.Conv2d(int(self.alpha_in * in_chn),
                     int(self.alpha_out * out_chn), kernel, stride=1,
                     padding=padding) if self.alpha_out > 0
                     and self.alpha_in > 0 else None)
        self.ltoh = (nn.Conv2d(int(self.alpha_in * in_chn), out_chn
                     - int(self.alpha_out * out_chn), kernel, stride=1,
                     padding=padding) if self.alpha_in > 0 else None)


    def forward(self, x):
        (high, low) = (x if isinstance(x, tuple) else (x, None))

        if self.htoh is not None:
            htoh = self.htoh(high)
        if self.htol is not None:
            htol = self.htol(F.avg_pool2d(high, 2, stride=2))
        if self.ltol and low is not None:
            ltol = self.ltol(low)
        if self.ltoh and low is not None:
            ltoh = F.interpolate(self.ltoh(low), scale_factor=2,
                                 mode='nearest')

        # if octave conv is being used as normal conv and both alpha are 0

        if self.alpha_in is 0 and self.alpha_out is 0:
            return (htoh, None)

        # for first layer when there is no low freq map was given and both maps were created from input image

        if low is None:
            return (htoh, htol)

        # this is for the last layer in the network when we dont want any low freq map output

        if self.alpha_out == 0:
            return (htoh.add_(ltoh), None)

        # otherwise add feature maps and return both high and low freq maps

        htoh.add_(ltoh)
        ltol.add_(htol)

        return (htoh, ltol)


class TransposeOctConv(nn.Module):

    """This is the implementation of Octave Transpose Conv from paper https://arxiv.org/abs/1906.12193"""

    def __init__(
        self,
        in_chn,
        out_chn,
        alphas=[0.5, 0.5],
        kernel=2,
        ):

        super(TransposeOctConv, self).__init__()

        (self.alpha_in, self.alpha_out) = alphas

        assert 1 > self.alpha_in >= 0 and 1 > self.alpha_out >= 0, \
            'alphas values must be bound between 0 and 1, it could be 0 but not 1'

        self.htoh = nn.ConvTranspose2d(in_chn - int(self.alpha_in
                * in_chn), out_chn - int(self.alpha_out * out_chn),
                kernel, 2)
        self.htol = (nn.ConvTranspose2d(in_chn - int(self.alpha_in
                     * in_chn), int(self.alpha_out * out_chn), kernel,
                     2) if self.alpha_out > 0 else None)
        self.ltol = (nn.ConvTranspose2d(int(self.alpha_in * in_chn),
                     int(self.alpha_out * out_chn), kernel,
                     2) if self.alpha_out > 0 and self.alpha_in
                     > 0 else None)
        self.ltoh = (nn.ConvTranspose2d(int(self.alpha_in * in_chn),
                     out_chn - int(self.alpha_out * out_chn), kernel,
                     2) if self.alpha_in > 0 else None)

    def forward(self, x):
        (high, low) = (x if isinstance(x, tuple) else (x, None))

        if self.htoh is not None:
            htoh = self.htoh(high)
        if self.htol is not None:
            htol = self.htol(F.avg_pool2d(high, 2, 2))
        if self.ltol is not None and low is not None:
            ltol = self.ltol(low)
        if self.ltoh is not None and low is not None:
            ltoh = F.interpolate(self.ltoh(low), scale_factor=2,
                                 mode='nearest')

        # it will behave as normal Transpose Conv operation

        if self.alpha_in is 0 and self.alpha_out is 0:
            return (htoh, None)

        # case where we don't want a low frequency map as output

        if self.alpha_out == 0:
            return (htoh.add_(ltoh), None)

        # otherwise add feature maps and return both high and low freq maps

        htoh.add_(ltoh)
        ltol.add_(htol)

        return (htoh, ltol)


class OctaveBnAct(nn.Module):

    '''This class applies OctaveConv-->BatchNorm-->ReLU operations on input'''

    def __init__(
        self,
        in_chn,
        out_chn,
        alphas=[0.5, 0.5],
        ):

        super(OctaveBnAct, self).__init__()
        self.oct = OctaveConv(in_chn, out_chn, alphas)
        self.bn1 = nn.BatchNorm2d(out_chn - int(alphas[1] * out_chn))
        self.bn2 = (None if alphas[1]
                    is 0 else nn.BatchNorm2d(int(alphas[1] * out_chn)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        (high, low) = self.oct(x)
        high = self.relu(self.bn1(high))
        if low is not None:
            low = self.relu(self.bn2(low))

        return (high, low)


class Encoder_block(nn.Module):

    def __init__(
        self,
        input_chn,
        output_chn,
        alphas=[0.5, 0.5],
        num_layers=2,
        kernel=3,
        padding=1,
        last_block=False,
        ):
        super(Encoder_block, self).__init__()

        layers_chn = [input_chn, output_chn] + [output_chn for i in
                range(num_layers)]

        if len(alphas) == 4:

        # means alphas for both blocks are given

            alphas1 = alphas[0:2]
            alphas2 = alphas[2:]
        else:

        # use same alphas for both blocks

            (alphas1, alphas2) = (alphas, alphas)

        self.pool = None
        self.block1 = OctaveBnAct(layers_chn[0], layers_chn[1], alphas1)
        self.block2 = OctaveBnAct(layers_chn[2], layers_chn[3], alphas2)

        if last_block is False:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        if self.pool:

            pool = (self.pool(x[0]), self.pool(x[1]))

        # here both x and pool are tuples consisting of high and low freq maps

            return (x, pool)
        return (x, None)


class Decoder_block(nn.Module):

    def __init__(
        self,
        input_chn,
        output_chn,
        alphas=[0.5, 0.5],
        bilinear=False,
        ):
        super(Decoder_block, self).__init__()

    # upsample feature maps size using either way

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = TransposeOctConv(input_chn, output_chn)

        if len(alphas) == 4:

        # means alphas for both blocks are given

            alphas1 = alphas[0:2]
            alphas2 = alphas[2:]
        else:

        # use same alphas for both blocks

            (alphas1, alphas2) = (alphas, alphas)

    # use two conv operation to process over upsample feature maps

        self.block1 = OctaveBnAct(input_chn, output_chn, alphas1)
        self.block2 = OctaveBnAct(output_chn, output_chn, alphas2)

    def forward(self, x, encoder_feat):
        x = self.up(x)
        x = (F.relu(x[0]), F.relu(x[1]))

    # concatenate high freq maps with corresponding encoder map

        assert x[0].shape[1] == encoder_feat[0].shape[1], \
            'High freq maps channels should be same'
        high = torch.cat([x[0], encoder_feat[0]], dim=1)

    # concatenate low freq maps with corresponding encoder map

        if x[1] is not None and encoder_feat[1] is not None:
            assert x[1].shape[1] == encoder_feat[1].shape[1], \
                'Low freq maps channels should be same'
            low = torch.cat([x[1], encoder_feat[1]], dim=1)

        x = (high, low)
        x = self.block1(x)
        x = self.block2(x)

        return x


class OctaveUnet(nn.Module):

    """ This is Unet architecture implementation using Oactave and Transpose Ovctave Convolution from the paper https://arxiv.org/abs/1906.12193. 
  However, in the paper feature maps retain same spatial size by using octave conv after max pool operation, here feature maps size is reduceed
  as per the Unet paper. """

    def __init__(self, num_classes):
        super(OctaveUnet, self).__init__()
        self.blocks = nn.ModuleDict()

        down_chn = [
            3,
            64,
            128,
            256,
            512,
            1024,
            ]
        up_chn = down_chn[::-1]

    # initial block which take in original image (only high freq map)

        self.blocks.update({'down_block1': Encoder_block(down_chn[0],
                           down_chn[1], [0, 0.5, 0.5, 0.5])})

    # rest of the encoder blocks

        for i in range(1, 5):
            if i == 4:

        # this makes sure that max pooling is not applied to the last block of encoder side

                self.blocks.update({'down_block' + str(i
                                   + 1): Encoder_block(down_chn[i],
                                   down_chn[i + 1], last_block=True)})
            else:

        # max pooling should be applied to rest of the blocks on encoder side

                self.blocks.update({'down_block' + str(i
                                   + 1): Encoder_block(down_chn[i],
                                   down_chn[i + 1])})

    # decoder blocks

        for i in range(3):
            self.blocks.update({'up_block' + str(i
                               + 1): Decoder_block(up_chn[i], up_chn[i
                               + 1])})

    # final block of decoder only outputs high freq map

        self.blocks.update({'up_block4': Decoder_block(up_chn[i + 1],
                           up_chn[i + 2], [0.5, 0.5, 0.5, 0])})

        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

    # save the encoder blocks output

        encoder_results = list()

    # iterate over encoder blocks

        for i in range(5):
            (feat, x) = self.blocks['down_block' + str(i + 1)](x)
            encoder_results.append([feat, x])

        index = 3
        for i in range(4):
            feat = self.blocks['up_block' + str(i + 1)](feat,
                    encoder_results[index - i][0])

    # classifier only gets the high freq map in the end

        return self.classifier(feat[0])