# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)


"""
Predicting graphs from models
"""

from __future__ import print_function
from os import path as fp
import sys

from attelo.fold import (select_testing, select_training)
from attelo.io import (load_model)
from attelo.table import mpack_pairing_distances
from attelo.decoding.intra import (IntraInterPair)
from attelo.harness.util import (makedirs)
from attelo.util import (Team)
import attelo.harness.decode as ath_decode

from .path import (attelo_doc_model_paths,
                   attelo_sent_model_paths,
                   decode_output_path)
from .util import (test_evaluation)


def _eval_banner(econf, lconf, fold):
    """
    Which combo of eval parameters are we running now?
    """
    msg = ("Reassembling "
           "fold {fnum} [{dset}]\t"
           "learner(s): {learner}\t"
           "decoder: {decoder}")
    return msg.format(fnum=fold,
                      dset=lconf.dataset,
                      learner=econf.learner.key,
                      decoder=econf.decoder.key)


def _say_if_decoded(lconf, econf, fold, stage='decoding'):
    """
    If we have already done the decoding for a given config
    and fold, say so and return True
    """
    if fp.exists(decode_output_path(lconf, econf, fold)):
        print(("skipping {stage} {learner} {decoder} "
               "(already done)").format(stage=stage,
                                        learner=econf.learner.key,
                                        decoder=econf.decoder.key),
              file=sys.stderr)
        return True
    else:
        return False


def delayed_decode(lconf, dconf, econf, fold):
    """
    Return possible futures for decoding groups within
    this model/decoder combo for the given fold
    """
    if fold is None and test_evaluation() is None:
        return []
    if _say_if_decoded(lconf, econf, fold, stage='decoding'):
        return []

    output_path = decode_output_path(lconf, econf, fold)
    makedirs(fp.dirname(output_path))

    if fold is None:
        # TODO read max_dist_by_lbl computed on train, stored in datapack maybe
        subpack = dconf.pack
    else:
        # get maximal length of relations for each label, from train
        # this will be passed to the decoders and used for pruning
        train_mpack = select_training(dconf.pack, dconf.folds, fold)
        max_dist_by_lbl = mpack_pairing_distances(train_mpack)
        subpack = select_testing(dconf.pack, dconf.folds, fold)

    doc_model_paths = attelo_doc_model_paths(lconf, econf.learner, fold)
    intra_flag = econf.settings.intra
    if intra_flag is not None:
        sent_model_paths =\
            attelo_sent_model_paths(lconf, econf.learner, fold)

        intra_model = Team('oracle', 'oracle')\
            if intra_flag.intra_oracle\
            else sent_model_paths.fmap(load_model)
        inter_model = Team('oracle', 'oracle')\
            if intra_flag.inter_oracle\
            else doc_model_paths.fmap(load_model)

        models = IntraInterPair(intra=intra_model,
                                inter=inter_model)
    else:
        models = doc_model_paths.fmap(load_model)

    return ath_decode.jobs(subpack, models,
                           econf.decoder.payload,
                           econf.settings.mode,
                           max_dist_by_lbl,
                           output_path)


def post_decode(lconf, dconf, econf, fold):
    """
    Join together output files from this model/decoder combo
    """
    if _say_if_decoded(lconf, econf, fold, stage='reassembly'):
        return

    print(_eval_banner(econf, lconf, fold), file=sys.stderr)
    if fold is None:
        subpack = dconf.pack
    else:
        subpack = select_testing(dconf.pack, dconf.folds, fold)
    ath_decode.concatenate_outputs(subpack,
                                   decode_output_path(lconf, econf, fold))
