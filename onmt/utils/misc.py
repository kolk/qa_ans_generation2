# -*- coding: utf-8 -*-

import torch
from torch.nn._functions.packing import PackPadded
from torch.nn.utils.rnn import PackedSequence
import numpy as np
from onmt.utils.logging import logger

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)



def pack_padded_sequence_ans(input, lengths, batch_first=False):
    r"""Packs a Tensor containing padded sequences of variable length.

    Input can be of size ``T x B x *`` where `T` is the length of the longest sequence
    (equal to ``lengths[0]``), `B` is the batch size, and `*` is any number of
    dimensions (including 0). If ``batch_first`` is True ``B x T x *`` inputs are
    expected.

    The sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the
    shortest one.

    Note:
        This function accepts any input that has at least two dimensions. You
        can apply it to pack the labels, and use the output of the RNN with
        them to compute the loss directly. A Tensor can be retrieved from
        a :class:`PackedSequence` object by accessing its ``.data`` attribute.

    Arguments:
        input (Tensor): padded batch of variable length sequences.
        lengths (Tensor): list of sequences lengths of each batch element.
        batch_first (bool, optional): if ``True``, the input is expected in ``B x T x *``
            format.

    Returns:
        a :class:`PackedSequence` object
    """

    ## sort the answer vector ############
    sorted_lengths = sorted(lengths, reverse=True)
    indices = np.argsort(lengths)[::-1]

    if isinstance(sorted_lengths, list):
        sorted_lengths = torch.LongTensor(sorted_lengths)

    data_, batch_sizes_ = PackPadded.apply(input, sorted_lengths, batch_first)
    '''
    logger.info("data_ size")
    logger.info(data_.size())
    logger.info("batch_sizes")
    logger.info(batch_sizes_)
    logger.info("lengths size")
    logger.info(len(lengths))
    logger.info("len indices")
    logger.info(len(indices))
    '''
    data = data_.clone()#torch.from_numpy(np.zeros(data_.size()))
    batch_sizes = batch_sizes_.clone()#torch.from_numpy(np.zeros(batch_sizes_.size()))
    for i, index in enumerate(indices):
        data[index] = data_[i, :]
        #batch_sizes[index] = batch_sizes_[i]

    return PackedSequence(data, batch_sizes)

