"""This module evaluates the output of discourse parsers.

Included are dependency and constituency tree metrics.
"""

from __future__ import print_function

import os

# from educe.rst_dt.annotation import RSTTree, SimpleRSTTree, _binarize
from educe.rst_dt.corpus import RstRelationConverter # , Reader as RstReader

# from educe.rst_dt.dep2con import (deptree_to_simple_rst_tree)
# from educe.rst_dt.deptree import (RstDepTree, RstDtException)
#
# from attelo.metrics.constituency import (LBL_FNS, parseval_detailed_report,
#                                          parseval_report)
# local to this package
from evals.codra import eval_codra_output
from evals.ours import load_deptrees_from_attelo_output


# RST corpus
CORPUS_DIR = os.path.join('corpus', 'RSTtrees-WSJ-main-1.0/')
CD_TRAIN = os.path.join(CORPUS_DIR, 'TRAINING')
CD_TEST = os.path.join(CORPUS_DIR, 'TEST')
# relation converter (fine- to coarse-grained labels)
RELMAP_FILE = os.path.join('/home/mmorey/melodi/educe',
                           'educe', 'rst_dt',
                           'rst_112to18.txt')
REL_CONV = RstRelationConverter(RELMAP_FILE).convert_tree


#
# EVALUATIONS
#

# * syntax: pred vs gold
EDUS_FILE = os.path.join('/home/mmorey/melodi',
                         'irit-rst-dt/TMP/syn_gold_coarse',
                         'TEST.relations.sparse.edu_input')
# outputs of parsers
EISNER_OUT_SYN_PRED = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/syn_pred_coarse',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt-eisner')

EISNER_OUT_SYN_GOLD = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/syn_gold_coarse',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt-eisner')

CODRA_OUT_DIR = '/home/mmorey/melodi/joty/Doc-level'



# FIXME load gold trees here once and for all, pass them to each
# evaluation

print('CODRA (Joty)')
eval_codra_output(CODRA_OUT_DIR)
print('=======================')

print('Eisner, predicted syntax')
load_deptrees_from_attelo_output(EISNER_OUT_SYN_PRED, EDUS_FILE,
                                 nuc_strategy="unamb_else_most_frequent",
                                 # nuc_strategy="most_frequent_by_rel",
                                 rank_strategy='closest-intra-rl-inter-rl',
                                 prioritize_same_unit=True)
print('======================')

print('Eisner, gold syntax')
load_deptrees_from_attelo_output(EISNER_OUT_SYN_GOLD, EDUS_FILE,
                                 nuc_strategy="unamb_else_most_frequent",
                                 # nuc_strategy="most_frequent_by_rel",
                                 rank_strategy='closest-intra-rl-inter-rl',
                                 prioritize_same_unit=True)
print('======================')


# TODO use nuclearity classifier
# starting with baseline: DummyNuclearityClassifier, that assigns to each
# EDU the most frequent nuclearity of its (incoming) relation in the
# training corpus, i.e. 'S' for 'NS', 'N' for 'NN'
