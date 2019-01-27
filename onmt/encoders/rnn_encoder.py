"""Define RNN-based encoders."""
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory
from onmt.utils.logging import logger
from onmt.utils.misc import pack_padded_sequence_ans

class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, lengths=None, type="src"):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()
        packed_emb = emb

        '''
        ######## Modified #########################
        emb_ans = self.embeddings(src_ans)
        packed_emb_ans = emb_ans
        ###########################################
        '''

        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths = lengths.view(-1).tolist()
            '''
            logger.info("forward in encoder")
            logger.info(lengths)
            logger.info(emb.size())
            logger.info(len(lengths))
            logger.info("type " + type)
            '''
            if type ==  "src":
                packed_emb = pack(emb, lengths)
            elif type=="ans":
                packed_emb = pack_padded_sequence_ans(emb, lengths)

            '''
            ########### Modified #####################
            packed_emb_ans = pack(emb_ans, lengths)
            #########################################
            '''

        memory_bank, encoder_final = self.rnn(packed_emb)

        '''
        ############### Modified ################################
        memory_bank_ans, encoder_final_ans = self.rnn(packed_emb_ans)
        ########################################################
        '''

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

            '''
            ############### Modified ############################
            memory_bank_ans = unpack(memory_bank_ans)[0]
            ###################################################
            '''

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)

            '''
            ############### Modified ###############################
            encoder_final_ans = self._bridge(encoder_final_ans)
            #########################################################
            '''

        return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs
