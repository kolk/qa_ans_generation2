from __future__ import division

import onmt.inputters as inputters
import onmt.utils
import onmt.opts as opts
from onmt.utils.logging import logger
import onmt.model_builder
import torch
import math
import os
import argparse

class Inference(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, valid_loss, fields, trunc_size=0,
                 shard_size=32, data_type='text', norm_method="sents", n_gpu=1,
                 gpu_rank=1, gpu_verbose_level=0, report_manager=None,
                 max_length=100,
                 copy_attn=False,
                 min_length=0,
                 stepwise_penalty=False,
                 sample_rate='16000',
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 use_filter_pred=False,
                 replace_unk=False,
                 ignore_when_blocking=[],
                 ):

        self.cuda = True
        # Basic attributes.
        self.model = model
        self.valid_loss = valid_loss
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.fields = fields
        self.sample_rate = sample_rate
        self.copy_attn = copy_attn
        self.min_length = min_length
        self.max_length = max_length
        self.stepwise_penalty = stepwise_penalty
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.ignore_when_blocking=  ignore_when_blocking
        # Set model in evaluation mode.
        self.model.eval()

    '''
    def make_valdiation(self, valid_iter_fct):
        valid_iter = valid_iter_fct()
        valid_stats = self.validate(valid_iter)
        return valid_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = onmt.utils.Statistics()

        # logger.info("inside validate")
        for batch in valid_iter:
            # logger.info("batch in valid iter")
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = inputters.make_features(batch, 'src', self.data_type)
            ans = inputters.make_features(batch, 'ans', self.data_type)
            # logger.info("self.data_type " + self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                #### Modified #######
                # logger.info(batch.src[0].size())
                # logger.info(batch.src[1].size())
                # logger.info("batch ans")
                # logger.info(batch.ans[0])
                # logger.info(batch.ans[1].size())
                _, ans_lengths = batch.ans
                #####################
            else:
                src_lengths = None
                ###### Modified #######
                ans_lengths = None
                #######################

            tgt = inputters.make_features(batch, 'tgt')
            # F-prop through the model.
            # src, ans, tgt, lengths, dec_state = None)
            outputs, attns, _ = self.model(src, ans, tgt, src_lengths, ans_lengths)

            # logger.info("outputs of model")
            # logger.info(outputs)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                batch, outputs, attns, train=False)

            # Update statistics.
            stats.update(batch_stats)

        return stats

    '''

    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  ans_path=None,
                  ans_data_iter=None,
                 ):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            src_data_iter (iterator): an interator generating source data
                e.g. it may be a list or an openned file
            tgt_path (str): filepath of target data
            tgt_data_iter (iterator): an interator generating target data
            src_dir (str): source directory path
                (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert src_data_iter is not None or src_path is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters.build_dataset(self.fields,
                                       self.data_type,
                                       src_path=src_path,
                                       src_data_iter=src_data_iter,
                                       tgt_path=tgt_path,
                                       tgt_data_iter=tgt_data_iter,
                                       src_dir=src_dir,
                                       sample_rate=self.sample_rate,
                                       window_size=self.window_size,
                                       window_stride=self.window_stride,
                                       window=self.window,
                                       use_filter_pred=self.use_filter_pred,
                                       ans_data_iter=ans_data_iter,
                                       ans_path=ans_path)



        print(data)
        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        for batch in data_iter:
            stats = self.translate_batch(batch, data)
            logger.info(stats)

    def translate_batch(self, batch, data):
        """
        Translate a batch of sentences.


        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
           return self._translate_batch(batch, data)


    def _translate_batch(self, batch, data):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        stats = onmt.utils.Statistics()
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])


        # Help functions for working with beams and batches
        def var(a): return torch.tensor(a, requires_grad=False)

        # (1) Run the encoder on the src.
        src = inputters.make_features(batch, 'src', data_type)
        ########## Modified #####################
        ans = inputters.make_features(batch, 'ans', data_type)
        tgt = inputters.make_features(batch, 'tgt', data_type)
        #####################################

        src_lengths = None
        ans_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src
            ############# Modified ###############
            _, ans_lengths = batch.ans
            ####################################

        outputs, attns, _ = self.model(src, ans, tgt, src_lengths, ans_lengths)

        # Compute loss.
        batch_stats = self.valid_loss.monolithic_compute_loss(
            batch, outputs, attns, train=False)

        # Update statistics.
        stats.update(batch_stats)
        ##### TODO: INCORPORATE BLUE AND OTHER STATS ###############
        return stats




    def _report_score(self, name, score_total, words_total):
        msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
            name, score_total / words_total,
            name, math.exp(-score_total / words_total)))
        return msg

    def _report_bleu(self, tgt_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")
        # Rollback pointer to the beginning.
        self.out_file.seek(0)
        print()

        res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s"
                                      % (base_dir, tgt_path),
                                      stdin=self.out_file,
                                      shell=True).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        res = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN"
            % (path, tgt_path),
            shell=True,
            stdin=self.out_file).decode("utf-8")
        msg = res.strip()
        return msg

def main():
    ## from translate.py
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    onmt.opts.model_opts(parser)
    logger.info("opts done")
    print("opts done")
    opt = parser.parse_args()

    logger.info("opt.gpu " + str(opt.gpu))
    print("opt.gpu " + str(opt.gpu))
    if opt.gpu > -1:
        torch.cuda.device(opt.gpu)
        n_gpu = 1
    else:
        n_gpu = 0

    # from train.py
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    logger.info("bulding loss")
    print("building loss")
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, fields["tgt"].vocab, opt, train=False)

    
    logger.info("Inference starts")
    print("Inference starts")
    infer = Inference(model, valid_loss, fields, trunc_size=0,
              shard_size=32, data_type='text', norm_method="sents",
              n_gpu=n_gpu, gpu_rank=1, gpu_verbose_level=0, report_manager=None)
    logger.info("translation")
    print("translation")

    infer.translate(src_path=opt.src,
                     tgt_path=opt.tgt,
                     src_dir=opt.src_dir,
                     batch_size=opt.batch_size,
                     ans_path=opt.ans)

if __name__ == "__main__":
    main()