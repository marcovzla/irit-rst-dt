"""Load the output of the parser from (Feng and Hirst, 2014).

This is 99% a copy/paste from evals/joty.py .
I need to come up with a better API and refactor accordingly.
"""

from __future__ import absolute_import, print_function

import itertools

from nltk import Tree

from educe.rst_dt.feng import load_feng_output_files
from educe.rst_dt.deptree import RstDepTree


def load_feng_ctrees(out_dir, rel_conv):
    """Load the ctrees output by Feng's parser as .dis files.

    This currently runs on the document-level files (.doc_dis).

    Parameters
    ----------
    out_dir: str
        Path to the base directory containing the output files.

    Returns
    -------
    ctree_pred: dict(str, RSTTree)
        RST ctree for each document.
    """
    # load predicted trees
    data_pred = load_feng_output_files(out_dir)
    # filenames = data_pred['filenames']
    doc_names_pred = data_pred['doc_names']
    rst_ctrees_pred = data_pred['rst_ctrees']

    # build a dict from doc_name to ctree (RSTTree)
    ctree_pred = dict()  # constituency trees
    for doc_name, ct_pred in itertools.izip(doc_names_pred, rst_ctrees_pred):
        # constituency tree
        # replace fine-grained labels with coarse-grained labels ;
        # the files we have already contain the coarse labels, except their
        # initial letter is capitalized whereas ours are not
        if rel_conv is not None:
            ct_pred = rel_conv(ct_pred)
        # "normalize" names of classes of RST relations:
        # "textual-organization" => "textual"
        for pos in ct_pred.treepositions():
            t = ct_pred[pos]
            if isinstance(t, Tree):
                node = t.label()
                if node.rel == 'textual-organization':
                    node.rel = 'textual'
        # end normalize
        ctree_pred[doc_name] = ct_pred

    return ctree_pred


def load_feng_dtrees(out_dir, rel_conv, nary_enc='chain'):
    """Get the dtrees that correspond to the ctrees output by Feng's parser.

    Parameters
    ----------
    out_dir: str
        Path to the base directory containing the output files.
    nary_enc: one of {'chain', 'tree'}
        Encoding for n-ary nodes.

    Returns
    -------
    dtree_pred: dict(str, RstDepTree)
        RST dtree for each document.
    """
    # load predicted c-trees
    ctree_pred = load_feng_ctrees(out_dir, rel_conv)

    # build a dict from doc_name to ordered dtree (RstDepTree)
    dtree_pred = dict()
    for doc_name, ct_pred in ctree_pred.items():
        # convert to an ordered dependency tree ;
        # * 'tree' produces a weakly-ordered dtree strictly equivalent
        # to the original ctree,
        # * 'chain' produces a strictly-ordered dtree for which strict
        # equivalence is not preserved
        dt_pred = RstDepTree.from_rst_tree(ct_pred, nary_enc=nary_enc)
        dtree_pred[doc_name] = dt_pred

    return dtree_pred
