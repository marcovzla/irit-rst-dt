"""Use the same evaluation procedure Evaluate the output of CODRA

"""

from __future__ import print_function

import itertools
import os

from educe.rst_dt.annotation import SimpleRSTTree, _binarize
from educe.rst_dt.codra import load_codra_output_files
from educe.rst_dt.corpus import (Reader as RstReader,
                                 RstRelationConverter as RstRelationConverter)
from educe.rst_dt.deptree import RstDepTree

from attelo.metrics.constituency import (parseval_detailed_report,
                                         parseval_report)
from attelo.metrics.deptree import compute_uas_las


# RST corpus
CORPUS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', 'corpus',
    'RSTtrees-WSJ-main-1.0/'))
CD_TRAIN = os.path.join(CORPUS_DIR, 'TRAINING')
CD_TEST = os.path.join(CORPUS_DIR, 'TEST')
# relation converter (fine- to coarse-grained labels)
RELMAP_FILE = os.path.join('/home/mmorey/melodi/educe',
                           'educe', 'rst_dt',
                           'rst_112to18.txt')
REL_CONV = RstRelationConverter(RELMAP_FILE).convert_tree


def eval_codra_output(codra_out_dir):
    """Load and evaluate the .dis files output by CODRA.

    This currently runs on the document-level files (.doc_dis).
    """
    # load reference trees
    dtree_true = dict()  # dependency trees
    ctree_true = dict()  # constituency trees
    # FIXME: find ways to read the right (not necessarily TEST) section
    # and only the required documents
    rst_reader = RstReader(CD_TEST)
    rst_corpus = rst_reader.slurp()

    for doc_id, rtree_true in sorted(rst_corpus.items()):
        doc_name = doc_id.doc

        # transform into binary tree with coarse-grained labels
        coarse_rtree_true = REL_CONV(rtree_true)
        bin_rtree_true = _binarize(coarse_rtree_true)
        ctree_true[doc_name] = bin_rtree_true

        # transform into dependency tree via SimpleRSTTree
        bin_srtree_true = SimpleRSTTree.from_rst_tree(coarse_rtree_true)
        dt_true = RstDepTree.from_simple_rst_tree(bin_srtree_true)
        dtree_true[doc_name] = dt_true

    # load predicted trees
    data_pred = load_codra_output_files(codra_out_dir)
    # filenames = data_pred['filenames']
    doc_names_pred = data_pred['doc_names']
    rst_ctrees_pred = data_pred['rst_ctrees']

    # gather predictions
    dtree_pred = dict()  # dependency trees
    ctree_pred = dict()  # constituency trees

    for doc_name, rst_ctree in itertools.izip(doc_names_pred, rst_ctrees_pred):
        # constituency tree
        # replace fine-grained labels with coarse-grained labels
        # 2016-06-27 useless, the files we have already contain the coarse
        # labels
        coarse_rtree_pred = REL_CONV(rst_ctree)
        ctree_pred[doc_name] = coarse_rtree_pred

        # dependency tree
        # conversion via SimpleRSTTree to RstDepTree
        bin_srtree_pred = SimpleRSTTree.from_rst_tree(coarse_rtree_pred)
        dt_pred = RstDepTree.from_simple_rst_tree(bin_srtree_pred)
        dtree_pred[doc_name] = dt_pred

    # compare pred and true
    common_doc_names = set(dtree_true.keys()) & set(dtree_pred.keys())

    # dep scores
    dtree_true_list = [dt for doc_name, dt in sorted(dtree_true.items())
                       if doc_name in common_doc_names]
    dtree_pred_list = [dt for doc_name, dt in sorted(dtree_pred.items())
                       if doc_name in common_doc_names]

    score_uas, score_las, score_ls = compute_uas_las(dtree_true_list,
                                                     dtree_pred_list)
    print('UAS / LAS / LS : {:.4f} / {:.4f} / {:.4f}'.format(
        score_uas, score_las, score_ls))

    skipped_docs = set()
    # convert dicts to aligned lists of SimpleRSTTrees, skipping docs where
    # needed
    ctree_true = [ct for doc_name, ct in sorted(ctree_true.items())
                  if doc_name not in skipped_docs]
    ctree_pred = [ct for doc_name, ct in sorted(ctree_pred.items())
                  if doc_name not in skipped_docs]
    # compute and print PARSEVAL scores
    print(parseval_report(ctree_true, ctree_pred, digits=4))
    # detailed report on S+N+R
    print(parseval_detailed_report(ctree_true, ctree_pred,
                                   metric_type='S+R'))
