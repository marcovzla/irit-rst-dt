import os.path
import subprocess

import paths


class CRFClassifier:
    def __init__(self, name, model_type, model_path, model_file, verbose):
        self.verbose = verbose
        self.name = name
        self.type = model_type
        self.model_fname = model_file
        self.model_path = model_path

        model_fpath = os.path.join(self.model_path, self.model_fname)
        if not os.path.exists(model_fpath):
            print ('The model path %s for CRF classifier %s does not exist.'
                   % model_fpath)
            raise OSError('Could not create classifier subprocess')
        
        self.classifier_cmd = [
            '%s/crfsuite-stdin' % paths.CRFSUITE_PATH,
            'tag', '-pi',
            '-m', '%s' % model_fpath
        ]
#        print self.classifier_cmd
        self.classifier = subprocess.Popen(self.classifier_cmd,
                                           stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
        
        if self.classifier.poll():
            raise OSError('Could not create classifier subprocess, with error info:\n%s' % self.classifier.stderr.readline())
        #self.cnt = 0

    def classify(self, vectors):
#        print '\n'.join(vectors) + "\n\n"        
        vectors_str = '\n'.join(vectors) + "\n\n"

        lines_out, lines_err = self.classifier.communicate(vectors_str)

        lines = []
        for line in lines_out.split('\n'):
            if not line.strip():
                break
            lines.append(line)

        # HACKY replace the subprocess closed by communicate()
        self.classifier = subprocess.Popen(self.classifier_cmd,
                                           stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
        
        if self.classifier.poll():
            raise OSError('Could not create classifier subprocess, with error info:\n%s' % self.classifier.stderr.readline())
        # end HACKY

        if self.classifier.poll():
            raise OSError('crf_classifier subprocess died')
        
        predictions = []
        for line in lines[1:]:
            line = line.strip()
#            print line
            if line != '':
                fields = line.split(':')
#                print fields
                label = fields[0]
                prob = float(fields[1])
                predictions.append((label, prob))
        
        seq_prob = float(lines[0].split('\t')[1])
        
        return seq_prob, predictions

    def poll(self):
        """
        Checks that the classifier processes are still alive
        """
        if self.classifier is None:
            return True
        return self.classifier.poll() is not None
    
    def unload(self):
        if self.classifier is not None and not self.poll():
            self.classifier.stdin.write('\n')
            print 'Successfully unloaded %s' % self.name
