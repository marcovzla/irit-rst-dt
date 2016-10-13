## buildedu.py
## Author: Yangfeng Ji
## Date: 05-03-2015
## Time-stamp: <yangfeng 09/25/2015 15:35:08>

from os import listdir
from os.path import join, basename
from model.classifier import Classifier
from model.docreader import DocReader
from model.sample import SampleGenerator
from cPickle import load
import gzip


# MM
from glob import glob
import os

DOC_EDUS = {os.path.splitext(os.path.basename(f))[0]: f
            for f in glob(os.path.join(
                    '/home/mmorey/melodi/rst/ji_eisenstein',
                    'DPLP/data/edus/*/*.edus'))}


def load_gold_edus(conll_file):
    """Load gold EDUs for injection into a conll file.

    Parameters
    ----------
    conll_file: str
        Path to the conll file.

    Returns
    -------
    edu_idc: list? of int
        Index of the EDU for each token.
    """
    result = []  # 1 if token is the last of its EDU, 0 otherwise

    doc_name = os.path.splitext(os.path.basename(conll_file))[0]
    # find corresponding file with gold EDUs
    fname_edus = DOC_EDUS[doc_name]
    edus = []
    with open(fname_edus) as f_edus:
        for line in f_edus:
            line = line.strip()
            if not line:
                continue
            # non-empty line
            edus.append(line)
    # open conll file and align tokens
    edu_idx = 0
    edu_txt = edus[edu_idx]  # remaining text of current EDU
    with open(conll_file) as f_conll:
        for line in f_conll:
            line = line.strip()
            if not line:
                continue
            fields = line.split('\t')
            wform_conll = fields[2]  # word form
            # try to read the same amount of characters off the current EDU
            wform_edus = edu_txt[:len(wform_conll)]
            try:
                assert wform_edus == wform_conll
            except AssertionError:
                if len(wform_edus) < len(wform_conll):
                    # EDU boundary happens in the middle of a token:
                    # possible causes: error in the text of the original doc
                    # (missing whitespace, wrong version of quotes...), or
                    # a plain error of the segmenter
                    assert wform_conll.startswith(wform_edus)
                    # set the EDU boundary at the current token
                    result.append(1)
                    # remaining text
                    rem_txt = wform_conll[len(wform_edus):].strip()
                    # read the first characters off the next EDU
                    edu_idx += 1
                    if edu_idx == len(edus):
                        edu_txt = ''
                    else:
                        edu_txt = edus[edu_idx]
                        # read the first characters off the beginning of the
                        # next EDU, assert that they match
                        assert edu_txt[:len(rem_txt)] == rem_txt
                        edu_txt = edu_txt[len(rem_txt):].lstrip()
                else:
                    # we don't know how to handle this (yet)
                    print(wform_conll, wform_edus)
                    raise
            else:
                # print(fields + [edu_idx + 1])
                # update the state of edu_txt for the next iteration
                edu_txt = edu_txt[len(wform_conll):].lstrip()
                if not edu_txt:
                    # when the current EDU is exhausted, pass to the next
                    result.append(1)
                    edu_idx += 1
                    if edu_idx == len(edus):
                        # normally, the text should be exhausted on both sides
                        # (.conll and .edus) at the same time ;
                        # if the .conll has extra text, the following should
                        # make the assertion above break at the next iteration
                        # of the loop
                        edu_txt = ''
                    else:
                        edu_txt = edus[edu_idx]
                else:
                    result.append(0)
    return result
# end MM

def main(fmodel, fvocab, rpath, wpath):
    clf = Classifier()
    dr = DocReader()
    clf.loadmodel(fmodel)
    flist = [join(rpath,fname) for fname in listdir(rpath) if fname.endswith('conll')]
    vocab = load(gzip.open(fvocab))
    for (fidx, fname) in enumerate(flist):
        print "Processing file: {}".format(fname)
        doc = dr.read(fname, withboundary=False)
        # predict segmentation
        if False:
            sg = SampleGenerator(vocab)
            sg.build(doc)
            M, _ = sg.getmat()
            predlabels = clf.predict(M)
        else:
            predlabels = load_gold_edus(fname)  # RESUME HERE
        doc = postprocess(doc, predlabels)
        writedoc(doc, fname, wpath)


def postprocess(doc, predlabels):
    """ Assign predlabels into doc
    """
    tokendict = doc.tokendict
    for gidx in tokendict.iterkeys():
        if predlabels[gidx] == 1:
            tokendict[gidx].boundary = True
        else:
            tokendict[gidx].boundary = False
        if tokendict[gidx].send:
            tokendict[gidx].boundary = True
    return doc


# def writedoc(doc, fname, wpath):
#     """ Write doc into a file with the CoNLL-like format
#     """
#     tokendict = doc.tokendict
#     N = len(tokendict)
#     fname = basename(fname) + '.edu'
#     fname = join(wpath, fname)
#     eduidx = 0
#     with open(fname, 'w') as fout:
#         for gidx in range(N):
#             fout.write(str(eduidx) + '\n')
#             if tokendict[gidx].boundary:
#                 eduidx += 1
#             if tokendict[gidx].send:
#                 fout.write('\n')
#     print 'Write segmentation: {}'.format(fname)


def writedoc(doc, fname, wpath):
    """ Write file
    """
    tokendict = doc.tokendict
    N = len(tokendict)
    fname = basename(fname).replace(".conll", ".merge")
    fname = join(wpath, fname)
    eduidx = 1
    with open(fname, 'w') as fout:
        for gidx in range(N):
            tok = tokendict[gidx]
            line = str(tok.sidx) + "\t" + str(tok.tidx) + "\t"
            line += tok.word + "\t" + tok.lemma + "\t" 
            line += tok.pos + "\t" + tok.deplabel + "\t" 
            line += str(tok.hidx) + "\t" + tok.ner + "\t"
            line += tok.partialparse + "\t" + str(eduidx) + "\n"
            fout.write(line)
            # Boundary
            if tok.boundary:
                eduidx += 1
            if tok.send:
                fout.write("\n")
