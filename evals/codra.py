"""Use the same evaluation procedure Evaluate the output of CODRA

"""

from __future__ import absolute_import, print_function

from collections import defaultdict
import itertools

import numpy as np

from educe.rst_dt.codra import load_codra_output_files
from educe.rst_dt.dep2con import deptree_to_rst_tree
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.document_plus import align_edus_with_paragraphs
#
from attelo.io import load_edus
from attelo.metrics.constituency import (parseval_detailed_report,
                                         parseval_report)
from attelo.metrics.deptree import compute_uas_las


def load_codra_ctrees(codra_out_dir, rel_conv):
    """Load the ctrees output by CODRA as .dis files.

    This currently runs on the document-level files (.doc_dis).

    Parameters
    ----------
    codra_out_dir: str
        Path to the base directory containing the output files.

    Returns
    -------
    ctree_pred: dict(str, RSTTree)
        RST ctree for each document.
    """
    # load predicted trees
    data_pred = load_codra_output_files(codra_out_dir)
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
        ctree_pred[doc_name] = ct_pred

    return ctree_pred


def load_codra_dtrees(codra_out_dir, rel_conv, nary_enc='chain'):
    """Get the dtrees that correspond to the ctrees output by CODRA.

    Parameters
    ----------
    codra_out_dir: str
        Path to the base directory containing the output files.
    nary_enc: one of {'chain', 'tree'}
        Encoding for n-ary nodes.

    Returns
    -------
    dtree_pred: dict(str, RstDepTree)
        RST dtree for each document.
    """
    # load predicted trees
    data_pred = load_codra_output_files(codra_out_dir)
    # filenames = data_pred['filenames']
    doc_names_pred = data_pred['doc_names']
    rst_ctrees_pred = data_pred['rst_ctrees']

    # build a dict from doc_name to ordered dtree (RstDepTree)
    dtree_pred = dict()
    for doc_name, ct_pred in itertools.izip(doc_names_pred, rst_ctrees_pred):
        # constituency tree
        # replace fine-grained labels with coarse-grained labels ;
        # the files we have already contain the coarse labels, except their
        # initial letter is capitalized whereas ours are not
        if rel_conv is not None:
            ct_pred = rel_conv(ct_pred)
        # convert to an ordered dependency tree ;
        # * 'tree' produces a weakly-ordered dtree strictly equivalent
        # to the original ctree,
        # * 'chain' produces a strictly-ordered dtree for which strict
        # equivalence is not preserved
        dt_pred = RstDepTree.from_rst_tree(ct_pred, nary_enc=nary_enc)
        dtree_pred[doc_name] = dt_pred

    return dtree_pred


# TODO move this generic util to a more appropriate place.
# This implementation is quite ad-hoc, tailored for RST e.g. to retrieve
# the edu_num, so I would need to generalize this code first.
def get_edu2sent(att_edus):
    """Get edu2sent mapping, from a list of attelo EDUs.

    Parameters
    ----------
    att_edus: list of attelo EDUs
        List of attelo EDUs, as produced by `load_edus`.

    Returns
    -------
    doc_name2edu2sent: dict(str, [int])
        For each document, get the sentence index for every EDU.

    Example:
    ```
    att_edus = load_edus(edus_file)
    doc_name2edu2sent = get_edu2sent(att_edus)
    for doc_name, edu2sent in doc_name2edu2sent.items():
        dtree[doc_name].edu2sent = edu2sent
    ```

    """
    edu2sent_idx = defaultdict(dict)
    for att_edu in att_edus:
        doc_name = att_edu.grouping
        edu_num = int(att_edu.id.rsplit('_', 1)[1])
        sent_idx = int(att_edu.subgrouping.split('_sent')[1])
        edu2sent_idx[doc_name][edu_num] = sent_idx
    # sort EDUs by num
    # rebuild educe-style edu2sent ; prepend 0 for the fake root
    doc_name2edu2sent = {doc_name: ([0]
                                    + [s_idx for e_num, s_idx
                                       in sorted(edu2sent.items())])
                         for doc_name, edu2sent in edu2sent_idx.items()}
    return doc_name2edu2sent
