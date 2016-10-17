"""Load dependencies output by Hayashi et al.'s parsers.

This module enables to process files in auto_parse/{dep/li,cons/trans_li}.
"""

from __future__ import absolute_import, print_function

import os
from glob import glob

from educe.rst_dt.corpus import Reader
from educe.rst_dt.deptree import RstDepTree


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


def load_hayashi_dtrees(out_dir, rel_conv):
    """Load the dtrees output by one of Hayashi et al.'s parser.

    Parameters
    ----------
    out_dir: str
        Path to the folder containing .dis files.
    rel_conv: RstRelationConverter
        Converter for relation labels (fine- to coarse-grained, plus
        normalization).

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
        dtree_pred[doc_name] = dt_pred
    return dtree_pred
