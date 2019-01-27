""" Base Class and function for Decoders """

from __future__ import division
import torch
import torch.nn as nn

import onmt.models.stacked_rnn
from onmt.utils.misc import aeq
from onmt.utils.rnn_factory import rnn_factory
from onmt.modules.global_attention import GlobalAttention

from onmt.utils.logging import logger


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`models.NMTModel`.
    .. mermaid::
       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Memory_Bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I
    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )
        #logger.info("context_gate")
        #logger.info(context_gate)


        # Set up the standard attention.
        self._coverage = coverage_attn

        ########### Modified #####################
        self.attn = GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        ############################################

        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt, memory_bank, memory_bank_ans, state, memory_lengths=None,
                memory_lengths_ans=None, step=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Check
        assert isinstance(state, RNNDecoderState)
        # tgt.size() returns tgt length and batch
        #logger.info("tgt.size")
        #logger.info(tgt.size())
        #logger.info("mem size")
        #logger.info(memory_bank.size())
        _, tgt_batch, _ = tgt.size()
        _, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        # END

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_bank_ans, state, memory_lengths=memory_lengths, memory_lengths_ans=memory_lengths_ans)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: decoder_outputs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(decoder_outputs) == list:
            decoder_outputs = torch.stack(decoder_outputs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])

        #logger.info("decoder forward decoder_outputs")
        #logger.info(decoder_outputs.size())

        return decoder_outputs, state, attns

    def init_decoder_state(self, src, ans, memory_bank, memory_bank_ans, encoder_final, encoder_ans_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            ## TODO: ANS IMPLEMENTATION FOR BIDIRECTIONAL ENCODER
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        #logger.info("init decoder state")
        #logger.info(len(encoder_final))
        #logger.info(encoder_final[0].size())
        #logger.info("encoder final ans")
        #logger.info(len(encoder_ans_final))

        ######### Modified ################################
        #l = [_fix_enc_hidden(enc_hid) for enc_hid in encoder_final] # list
        #l_ans = [_fix_enc_hidden(enc_hid) for enc_hid in encoder_ans_final] # list
        '''
        for enc_hid in encoder_final:
            hid = _fix_enc_hidden(enc_hid)
            #logger.info("fix enc hidden")
            #logger.info(hid.size())
        '''
        l_final = [_fix_enc_hidden(torch.add(enc_q, enc_ans)) for enc_q, enc_ans in zip(encoder_final, encoder_ans_final)]
        ####################################

        #logger.info(len(l))
        #logger.info(type(l[0])) # <class 'torch.Tensor'>
        #logger.info("l[0] " + str(len(l[0])))
        #logger.info("l[1]" + str(len(l[1])))
        #logger.info("l_ans " + str(len(l_ans)))
        #logger.info("l_ans[0] " + str(len(l_ans[0])))
        #logger.info("l_ans[1] " + str(len(l_ans[1])))

        ########### Modified ######################
        rnn_decoderstate = RNNDecoderState(self.hidden_size,
                        tuple(l_final))
        #logger.info("rnn decoder state obj")
        #logger.info(rnn_decoderstate)
        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple(l_final))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   l_final)
        ##########################################


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.
    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`
    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """

    def _run_forward_pass(self, tgt, memory_bank, memory_bank_ans, state, memory_lengths=None, memory_lengths_ans=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Tensor): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        decoder_outputs, p_attn = self.attn(
            rnn_output.transpose(0, 1).contiguous(),
            memory_bank.transpose(0, 1),
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn


        # Calculate the context gate.
        if self.context_gate is not None:
            decoder_outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                decoder_outputs.view(-1, decoder_outputs.size(2))
            )
            decoder_outputs = \
                decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)

        decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, decoder_outputs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.
    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`
    .. mermaid::
       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Memory_Bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, memory_bank_ans,  state, memory_lengths=None, memory_lengths_ans=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        #logger.info("memory_bank")
        #logger.info(len(memory_bank))
        #logger.info(len(memory_bank[0]))
        #logger.info(len(memory_bank[0][0]))
        # Additional args check.
        input_feed = state.input_feed.squeeze(0) # list of tensors of len 64, len(input_feed[i]) = 500 [ size(): [batch x opts.rnn_size]
        input_feed_batch, _ = input_feed.size() # 64
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        if self._coverage:
            attns["coverage"] = []


        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim (51 x 64 x 500)

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)
            #logger.info("decoder input size")
            #logger.info(decoder_input.size())
            #logger.info("hidden size")
            #logger.info(hidden[0].size())
            rnn_output, hidden = self.rnn(decoder_input, hidden)
            #logger.info("rnn_output size")
            #logger.info(rnn_output.size())
            #logger.info("hidden size")
            #logger.info(hidden[0].size())
            ############## Modified ##########################
            #logger.info("rnn_output size")
            #logger.info(rnn_output.size())
            #logger.info("memory bank size")
            #logger.info(memory_bank.size())

            #memory_bank_final = torch.cat([memory_bank, memory_bank_ans], 1)
            #logger.info("memory_bank_final size")
            #logger.info(memory_bank_final.size())
            ## TODO: memory length for answer needs to change
            memory_bank_final = torch.cat([memory_bank, memory_bank_ans], 0)
            memory_lengths_final = torch.add(memory_lengths, memory_lengths_ans)

            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank_final.transpose(0, 1),
                memory_lengths=memory_lengths_final)

            #logger.info("decoder_output")
            #logger.info(decoder_output.size())
            #logger.info("p_attn")
            #logger.info(p_attn.size())
            #logger.info("p_attn_ans")
            #logger.info(p_attn_ans.size())
            ####################################################
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank_final.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        #logger.info("hidden size")
        #logger.info(hidden[0].size())
        #logger.info("decoder_outputs len")
        #logger.info(len(decoder_outputs))
        #logger.info("decoder outputs 0 index")
        #logger.info(decoder_outputs[0].size())
        #logger.info("attentions")
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.
    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()


class RNNDecoderState(DecoderState):
    """ Base class for RNN decoder state """

    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = self.hidden[0].data.new(*h_size).zero_() \
                              .unsqueeze(0)
        #logger.info("RNNDecoderState")
        #logger.info("hidden size")
        #logger.info(len(self.hidden))
        #logger.info(self.hidden[0].size())
        #logger.info("input_feed")
        #logger.info(self.input_feed.size())


    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        """ Update decoder state """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [e.data.repeat(1, beam_size, 1)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]

    def map_batch_fn(self, fn):
        self.hidden = tuple(map(lambda x: fn(x, 1), self.hidden))
        self.input_feed = fn(self.input_feed, 1)
