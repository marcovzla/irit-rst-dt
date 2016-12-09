"""Load dependencies output by Hayashi et al.'s parsers.

This module enables to process files in auto_parse/{dep/li,cons/trans_li}.
"""

from __future__ import absolute_import, print_function

import os
from glob import glob

from educe.learning.edu_input_format import load_edu_input_file
from educe.rst_dt.corpus import Reader
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.dep2con import deptree_to_rst_tree


# load true ctrees, from the TEST section of the RST-DT, to get gold EDUs
RST_DT_DIR = '/home/mmorey/corpora/rst-dt/rst_discourse_treebank/data'
RST_TEST_DIR = os.path.join(RST_DT_DIR, 'RSTtrees-WSJ-main-1.0/TEST')
if not os.path.exists(RST_TEST_DIR):
    raise ValueError('Unable to find RST test files at ', RST_TEST_DIR)
RST_TEST_READER = Reader(RST_TEST_DIR)
RST_TEST_CTREES_TRUE = {k.doc: v for k, v in RST_TEST_READER.slurp().items()}


def _load_hayashi_dep_file(f, edus):
    """Do load.

    Parameters
    ----------
    f: File
        dep file, open
    edus: list of EDU
        True EDUs in this document.

    Returns
    -------
    dt: RstDepTree
        Predicted dtree
    """
    dt = RstDepTree(edus=edus, origin=None, nary_enc='tree')  # FIXME origin
    for line in f:
        line = line.strip()
        if not line:
            continue
        dep_idx, gov_idx, lbl = line.split()
        dep_idx = int(dep_idx)
        gov_idx = int(gov_idx)
        dt.add_dependency(gov_idx, dep_idx, label=lbl)
    return dt


def load_hayashi_dep_file(fname, edus):
    """Load a file.

    Parameters
    ----------
    fname: str
        Path to the file

    Returns
    -------
    dt: RstDepTree
        Dependency tree corresponding to the content of this file.
    """
    with open(fname) as f:
        return _load_hayashi_dep_file(f, edus)


def load_hayashi_dep_files(out_dir):
    """Load dep files output by one of Hayashi et al.'s parser.

    Parameters
    ----------
    out_dir: str
        Path to the folder containing the .dis files.
    """
    dtrees = dict()
    for fname in glob(os.path.join(out_dir, '*.dis')):
        doc_name = os.path.splitext(os.path.basename(fname))[0]
        edus = RST_TEST_CTREES_TRUE[doc_name].leaves()
        dtrees[doc_name] = load_hayashi_dep_file(fname, edus)
    return dtrees


def load_hayashi_dep_dtrees(out_dir, rel_conv, edus_file_pat, nuc_clf,
                            rnk_clf):
    """Load the dtrees output by one of Hayashi et al.'s dep parsers.

    Parameters
    ----------
    out_dir : str
        Path to the folder containing .dis files.

    rel_conv : RstRelationConverter
        Converter for relation labels (fine- to coarse-grained, plus
        normalization).

    edus_file_pat : str
        Pattern for the .edu_input files.

    nuc_clf : NuclearityClassifier
        Nuclearity classifier

    rnk_clf : RankClassifier
        Rank classifier

    Returns
    -------
    dtree_pred: dict(str, RstDepTree)
        RST dtree for each document.
    """
    dtree_pred = dict()

    dtrees = load_hayashi_dep_files(out_dir)
    for doc_name, dt_pred in dtrees.items():
        if rel_conv is not None:
            dt_pred = rel_conv(dt_pred)
        # WIP add nuclearity and rank
        edus_data = load_edu_input_file(edus_file_pat.format(doc_name),
                                        edu_type='rst-dt')
        edu2sent = edus_data['edu2sent']
        dt_pred.sent_idx = [0] + edu2sent  # 0 for fake root ; DIRTY
        dt_pred.nucs = nuc_clf.predict([dt_pred])[0]
        dt_pred.ranks = rnk_clf.predict([dt_pred])[0]
        # end WIP
        dtree_pred[doc_name] = dt_pred
        
    return dtree_pred


def load_hayashi_dep_ctrees(out_dir, rel_conv, edus_file_pat, nuc_clf,
                            rnk_clf):
    """Load the dtrees output by one of Hayashi et al.'s dep parsers.

    Parameters
    ----------
    out_dir : str
        Path to the folder containing .dis files.

    rel_conv : RstRelationConverter
        Converter for relation labels (fine- to coarse-grained, plus
        normalization).

    edus_file_pat : str
        Pattern for the .edu_input files.

    nuc_clf : NuclearityClassifier
        Nuclearity classifier

    rnk_clf : RankClassifier
        Rank classifier

    Returns
    -------
    ctree_pred: dict(str, RSTTree)
        RST ctree for each document.
    """
    ctree_pred = dict()

    dtree_pred = load_hayashi_dep_dtrees(out_dir, rel_conv, edus_file_pat,
                                         nuc_clf, rnk_clf)
    for doc_name, dt_pred in dtree_pred.items():
        try:
            ct_pred = deptree_to_rst_tree(dt_pred)
        except RstDtException:
            print(doc_name)
            raise
        else:
            ctree_pred[doc_name] = ct_pred

    return ctree_pred
