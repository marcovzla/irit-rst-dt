"""This module evaluates the output of discourse parsers.

Included are dependency and constituency tree metrics.
"""

from __future__ import print_function

import os

from educe.rst_dt.annotation import _binarize
from educe.rst_dt.corpus import (RstRelationConverter,
                                 Reader as RstReader)
from educe.rst_dt.dep2con import (DummyNuclearityClassifier,
                                  InsideOutAttachmentRanker)
from educe.rst_dt.deptree import RstDepTree
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


# hyperparams
NUC_STRATEGY = 'unamb_else_most_frequent'
RNK_STRATEGY = 'sdist-edist-rl'
RNK_PRIORITY_SU = True
RNK_ORDER = 'weak'


# FIXME:
# * [ ] create summary table with one system per row, one metric per column,
#   keep only the f-score (because for binary trees with manual segmentation
#   precision = recall = f-score).

# 1. load train section of the RST corpus, fit (currently dummy) classifiers
# for nuclearity and rank
reader_train = RstReader(CD_TRAIN)
corpus_train = reader_train.slurp()
# gold RST trees
ctree_true = dict()  # ctrees
ctree_bin_true = dict()  # ctrees, binarized
dtree_true = dict()  # dtrees from the original ctrees ('tree' transform)
dtree_bin_true = dict()  # dtrees from the binarized ctrees ('chain' transform)
for doc_id, ct_true in sorted(corpus_train.items()):
    doc_name = doc_id.doc
    # flavours of ctree
    ct_true = REL_CONV(ct_true)  # map fine to coarse relations
    ctree_true[doc_name] = ct_true
    ct_bin_true = _binarize(ct_true)
    ctree_bin_true[doc_name] = ct_bin_true
    # flavours of dtree
    dt_true = RstDepTree.from_rst_tree(ct_true, nary_enc='tree')
    dt_bin_true = RstDepTree.from_rst_tree(ct_true, nary_enc='chain')
    # alt:
    # dt_bin_true = RstDepTree.from_rst_tree(ct_bin_true, nary_enc='chain')
    dtree_true[doc_name] = dt_true
    dtree_bin_true[doc_name] = dt_bin_true
# fit classifiers for nuclearity and rank (DIRTY)
# NB: both are (dummily) fit on weakly ordered dtrees
X_train = []
y_nuc_train = []
y_rnk_train = []
for doc_name, dt in sorted(dtree_true.items()):
    X_train.append(dt)
    y_nuc_train.append(dt.nucs)
    y_rnk_train.append(dt.ranks)
# nuclearity clf
nuc_clf = DummyNuclearityClassifier(strategy=NUC_STRATEGY)
nuc_clf.fit(X_train, y_nuc_train)
# rank clf
rnk_clf = InsideOutAttachmentRanker(strategy=RNK_STRATEGY,
                                    prioritize_same_unit=RNK_PRIORITY_SU,
                                    order=RNK_ORDER)
rnk_clf.fit(X_train, y_rnk_train)

# load test section of the RST corpus
reader_test = RstReader(CD_TEST)
corpus_test = reader_test.slurp()
# gold RST trees
ctree_true = dict()  # ctrees
ctree_bin_true = dict()  # ctrees, binarized
dtree_true = dict()  # dtrees from the original ctrees ('tree' transform)
dtree_bin_true = dict()  # dtrees from the binarized ctrees ('chain' transform)
for doc_id, ct_true in sorted(corpus_test.items()):
    doc_name = doc_id.doc
    # flavours of ctree
    ct_true = REL_CONV(ct_true)  # map fine to coarse relations
    ctree_true[doc_name] = ct_true
    ct_bin_true = _binarize(ct_true)
    ctree_bin_true[doc_name] = ct_bin_true
    # flavours of dtree
    dt_true = RstDepTree.from_rst_tree(ct_true, nary_enc='tree')
    dt_bin_true = RstDepTree.from_rst_tree(ct_true, nary_enc='chain')
    # alt:
    # dt_bin_true = RstDepTree.from_rst_tree(ct_bin_true, nary_enc='chain')
    dtree_true[doc_name] = dt_true
    dtree_bin_true[doc_name] = dt_bin_true


if True:
    print('CODRA (Joty)')
    eval_codra_output(ctree_true, dtree_true,
                      CODRA_OUT_DIR, EDUS_FILE,
                      nuc_clf, rnk_clf,
                      detailed=False)
    print('=======================')

if True:
    print('[chain] Eisner, predicted syntax')
    load_deptrees_from_attelo_output(ctree_true, dtree_true,
                                     EISNER_OUT_SYN_PRED, EDUS_FILE,
                                     nuc_clf, rnk_clf,
                                     detailed=False)
    print('======================')

if True:
    print('[tree] Eisner, predicted syntax + same-unit')
    load_deptrees_from_attelo_output(ctree_true, dtree_true,
                                     EISNER_OUT_TREE_SYN_PRED_SU, EDUS_FILE,
                                     nuc_clf, rnk_clf,
                                     detailed=False)
    print('======================')

print('Eisner, predicted syntax + same-unit')
load_deptrees_from_attelo_output(ctree_true, dtree_true,
                                 EISNER_OUT_SYN_PRED_SU, EDUS_FILE,
                                 nuc_clf, rnk_clf,
                                 detailed=False)
print('======================')

print('Eisner, gold syntax')
load_deptrees_from_attelo_output(ctree_true, dtree_true,
                                 EISNER_OUT_SYN_GOLD, EDUS_FILE,
                                 nuc_clf, rnk_clf,
                                 detailed=False)
print('======================')
