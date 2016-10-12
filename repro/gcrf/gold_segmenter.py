"""Pseudo-segmenter for manual (gold) EDU segmentation.

"""

from __future__ import print_function
import os

import utils.utils


class GoldSegmenter(object):
    """Gold segmenter"""

    def __init__(self, root, _name='gold_segmenter', verbose=False):
        self.root = root  # root dir for gold .edu files
        self.name = _name
        self.verbose = verbose

    def segment(self, doc, filename):
        """Segment a document.

        Parameters
        ----------
        doc: Document
            Internal representation of a document
        filename: str
            Name of the document
        """
        # load true segmentation
        doc_predictions = []
        fname_doc = os.path.basename(filename)
        fname_edus = os.path.join(self.root, fname_doc + '.edus')
        with open(fname_edus) as f_edus:
            fedus_sentences = f_edus.readlines()
        doc_predictions = []
        for sent in fedus_sentences:
            toks = sent.strip().split(' ')
            predictions = []
            for tok in toks[:-1]:
                if tok == 'EDU_BREAK':
                    if predictions:
                        # "not predictions" should not happen, but
                        # apparently it does, e.g. wsj_1376:
                        # "EDU_BREAK It provides..."
                        predictions[-1] = 1
                else:
                    predictions.append(0)
            # set a marginal proba of 1.0 for each prediction
            doc_predictions.append([(x, 1.0) for x in predictions])

        # c/c
        doc.edu_word_segmentation = []
        doc.cuts = []
        doc.edus = []
        # end c/c

        for sentence, predictions in zip(doc.sentences, doc_predictions):
            self.segment_sentence(sentence, predictions)

        # c/c
        doc.start_edu = 0
        doc.end_edu = len(doc.edus)
        # end c/c

    def segment_sentence(self, sentence, predictions):
        """Segment a sentence.
        """
        # c/c from crf_segmenter
        if len(sentence.tokens) == 1:
            edus = [[sentence.tokens[0].word, sentence.raw_text[-3 : ]]]
            
            sentence.doc.cuts.append((len(sentence.doc.edus), len(sentence.doc.edus) + len(edus)))
            sentence.start_edu = len(sentence.doc.edus)
            sentence.end_edu = len(sentence.doc.edus) + len(edus)
            sentence.doc.edu_word_segmentation.append([(0, 1)])
            sentence.doc.edus.extend(edus)
            return
        # end c/c

        # another c/c
        edus = []
        edu_word_segmentations = []
        start = 0
        for i in range(len(predictions)):
            pred = int(predictions[i][0])
            if pred == 1:
#                print i, pred
                edu_word_segmentations.append((start, i + 1))
                start = i + 1
        
        edu_word_segmentations.append((start, len(sentence.tokens)))
        
        for (start_word, end_word) in edu_word_segmentations:
            edu = []
            for j in range(start_word, end_word):
                edu.extend(utils.utils.unescape_penn_special_word(sentence.tokens[j].word).split(' '))
            
            if end_word == len(sentence.tokens):
#                print sentence.raw_text
                edu.append(sentence.raw_text[-3 : ])
            edus.append(edu)
        
        sentence.doc.cuts.append((len(sentence.doc.edus), len(sentence.doc.edus) + len(edus)))
        sentence.start_edu = len(sentence.doc.edus)
        sentence.end_edu = len(sentence.doc.edus) + len(edus)
        sentence.doc.edu_word_segmentation.append(edu_word_segmentations)
        sentence.doc.edus.extend(edus)
        # end another c/c

    def unload(self):
        """Unload ; a no-op here"""
        pass
