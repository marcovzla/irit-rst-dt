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

# 2016-09-14 "tree" transform, predicted syntax
EISNER_OUT_TREE_SYN_PRED_SU = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/latest',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt_su-eisner')
# end 2016-09-14

EISNER_OUT_SYN_PRED_SU = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/latest',  # lbl
    'scratch-current/combined',
    'output.maxent-AD.L-jnt_su-eisner')

EISNER_OUT_SYN_GOLD = os.path.join(
    '/home/mmorey/melodi',
    'irit-rst-dt/TMP/syn_gold_coarse',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt-eisner')

CODRA_OUT_DIR = '/home/mmorey/melodi/rst/joty/Doc-level'



# FIXME:
# * [ ] load gold trees here once and for all, pass them to each evaluation
# * [ ] create summary table with one system per row, one metric per column,
#   keep only the f-score (because for binary trees with manual segmentation
#   precision = recall = f-score).

print('CODRA (Joty)')
eval_codra_output(CODRA_OUT_DIR, EDUS_FILE,
                  'chain',
                  nuc_strategy="unamb_else_most_frequent",
                  rank_strategy='sdist-edist-rl',
                  prioritize_same_unit=True,
                  binarize_ref=False,
                  detailed=False)
print('=======================')

print('[chain] Eisner, predicted syntax')
load_deptrees_from_attelo_output(EISNER_OUT_SYN_PRED, EDUS_FILE,
                                 'chain',
                                 nuc_strategy="unamb_else_most_frequent",
                                 # nuc_strategy="most_frequent_by_rel",
                                 rank_strategy='sdist-edist-rl',
                                 prioritize_same_unit=True,
                                 order='weak',
                                 binarize_ref=False,
                                 detailed=False)
print('======================')

print('[tree] Eisner, predicted syntax + same-unit')
load_deptrees_from_attelo_output(EISNER_OUT_TREE_SYN_PRED_SU, EDUS_FILE,
                                 'tree',
                                 nuc_strategy="unamb_else_most_frequent",
                                 # nuc_strategy="most_frequent_by_rel",
                                 rank_strategy='sdist-edist-rl',
                                 prioritize_same_unit=True,
                                 order='weak',
                                 binarize_ref=False,
                                 detailed=False)
print('======================')

print('Eisner, predicted syntax + same-unit')
load_deptrees_from_attelo_output(EISNER_OUT_SYN_PRED_SU, EDUS_FILE,
                                 'chain',
                                 nuc_strategy="unamb_else_most_frequent",
                                 # nuc_strategy="most_frequent_by_rel",
                                 rank_strategy='sdist-edist-rl',
                                 prioritize_same_unit=True,
                                 detailed=False)
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
