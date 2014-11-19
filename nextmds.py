import csv
import os
import struct
import unittest
from itertools import count
from math import floor
from numpy import array,dot,mat,sqrt,trace,zeros,sum,ndarray,square,insert
from numpy.linalg import norm
from numpy.random import randn,shuffle

def updateModel(model,query,STEP_COUNTER):
    def StochasticGradient(Xq):
        dims = model.shape[1]
        H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])

        g = zeros( (3,dims) )
        emploss = 0
        hingeloss = 0

        loss_ijk = trace(dot(H,dot(Xq,Xq.T)))
        if loss_ijk + 1. >= 0:
            hingeloss = loss_ijk + 1.
            g = g + H * Xq

            if loss_ijk >= 0:
                emploss = 1.

        return g, emploss, hingeloss, loss_ijk

    def regularize(Xq, regularization=10):
        for i in range(3):
            norm_i = norm(Xq[i, :])
            if norm_i > regularization:
                Xq[i, :] *= (regularization / norm_i)

        return Xq

    # Select the model elements relevant to the current query.
    Xq = model[query, :]
    stepSize  = sqrt(100.)/sqrt(next(STEP_COUNTER) + 100.)
    g,emploss,hingeloss,loss = StochasticGradient(Xq)
    Xq -= (stepSize * g)  # max-norm
    #Xq = regularize(Xq)
    model[query, :] = Xq

    return {'gradient':g,'emploss':emploss,'hingeloss':hingeloss,'loss':loss}

def evaluateModel(model,query):
    def computeLoss(Xq):
        H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
        loss_ijk = trace(dot(H,dot(Xq,Xq.T)))
        emploss = 0
        hingeloss = 0
        if loss_ijk + 1. >= 0:
            hingeloss = loss_ijk + 1.
            if loss_ijk >= 0:
                emploss = 1.
        return emploss, hingeloss, loss_ijk

    # Select the model elements relevant to the current query.
    Xq = model[query, :]
    emploss,hingeloss,loss = computeLoss(Xq)

    return {'emploss':emploss,'hingeloss':hingeloss,'loss':loss}

def read_triplets(data_path):
    with open(data_path, 'rb') as ifile:
        reader = csv.reader(ifile,escapechar='\\')

        print "starting read..."
        RANDOM_l = []
        ADAPTIVE_l = []
        CV_l = []

        header = reader.next() # reads first row
        primary = header.index('primary')
        alternate = header.index('alternate')
        target = header.index('target')
        qt = header.index('queryType')
        labels = []
        query_type_count = {}
        query_names = ('random','adaptive','cv')
        for row in reader: # reads rest of rows
            query = [ row[i].strip() for i in (primary,alternate,target) ]
            query_type = int(row[qt].strip())
            [ labels.append(x) for x in query if not x in labels ]

            if query_type == 0:
                RANDOM_l.append(query)
            elif query_type == 1:
                ADAPTIVE_l.append(query)
            elif query_type == 2:
                CV_l.append(query)

            try:
                query_type_count[query_names[query_type]] += 1
            except KeyError:
                query_type_count[query_names[query_type]] = 1

    item_count = len(labels)
    intconv = False
    try:
        # If all labels are actually integers, convert
        # so they end up in numeric order after sort.
        labels = [int(x) for x in labels]
        intconv = True
    except ValueError:
        pass

    labels.sort()

    if intconv:
        # Put them back as strings, so they can match
        # against queries.
        labels = [str(x) for x in labels]

    RANDOM = []
    ADAPTIVE = []
    CV = []
    n = 0
    for queryLabels in RANDOM_l:
        q = [ labels.index(x) for x in queryLabels ]
        RANDOM.append(q)
        n += 1

    for queryLabels in ADAPTIVE_l:
        q = [ labels.index(x) for x in queryLabels ]
        ADAPTIVE.append(q)
        n += 1

    for queryLabels in CV_l:
        q = [ labels.index(x) for x in queryLabels ]
        CV.append(q)
        n += 1

    print "done reading! " + "n="+str(item_count) + "  |S|="+str(n)
    print "queryType0 (random)   = " + str(query_type_count['random'])
    print "queryType1 (adaptive) = " + str(query_type_count['adaptive'])
    print "queryType2 (cv)       = " + str(query_type_count['cv'])

    return {'RANDOM': RANDOM,
            'ADAPTIVE': ADAPTIVE,
            'CV': CV,
            'nitems': item_count,
            'nqueries': query_type_count,
            'labels': labels
            }

def writeModel(model, outdir, writemode):
    def writeModelBinary():
        nitem = model.shape[0]
        try:
            ndim = model.shape[1]
        except IndexError:
            ndim = 1

        BINOUT = struct.pack('2i', nitem, ndim)
        BINOUT += struct.pack('{n}d'.format(n=nitem*ndim), *model.flatten())
        with open(os.path.join(outdir,'model.bin'), 'wb') as f:
            f.write(BINOUT)

    def writeModelText():
        with open(os.path.join(outdir,'model.csv'), 'wb') as f:
            writer = csv.writer(f,escapechar='\\')
            writer.writerows(model)

    try:
        if writemode == 'binary':
            writeModelBinary()
        elif writemode == 'text':
            writeModelText()
        else:
            writeModelBinary()
            writeModelText()

    except KeyError:
        writeModelBinary()
        writeModelText()

def writeLoss(lossLog, outdir, writemode):
    lossLog = array(lossLog)
    def writeLossBinary():
        nepoch = lossLog.shape[0]
        ntype = lossLog.shape[1]
        BINOUT = struct.pack('2i', nepoch, ntype)
        BINOUT += struct.pack('{n}d'.format(n=nepoch*ntype), *lossLog.flatten())
        with open(os.path.join(outdir,'loss.bin'), 'wb') as f:
            f.write(BINOUT)

    def writeLossText():
        with open(os.path.join(outdir,'loss.csv'), 'wb') as f:
            writer = csv.writer(f,escapechar='\\')
            writer.writerow(['eloss','hloss','eloss_t','hloss_t'])
            writer.writerows(lossLog)

    try:
        if writemode == 'binary':
            writeLossBinary()
        elif writemode == 'text':
            writeLossText()
        else:
            writeLossBinary()
            writeLossText()

    except KeyError:
        writeLossBinary()
        writeLossText()

def initializeEmbedding(nitems, dimensions):
    model = randn(nitems,dimensions)
    model = model/norm(model)*sqrt(nitems)
    return model

def fitModel(model, responses, opts=False):
    STEP_COUNTER = count(1)
    type_names = ('random','adaptive','cv')
    def printLoss(MDATA):
        epoch_str = "epoch = {:2d}".format(MDATA['epoch'])
        emp_str = "emploss = {:.3f}".format(MDATA['emploss']/MDATA['ntrain'])
        hinge_str = "hingeloss = {:.3f}".format(MDATA['hingeloss'])
        norm_str = "norm(X)/sqrt(n) = {:.3f}".format(MDATA['norm'])
        print '  '.join([epoch_str,emp_str,hinge_str,norm_str])

    if not opts:
        opts = {
            'proportion': 1.0,
            'nepochs': 10,
            'traincode': 1,
            'testcode': 2,
            'verbose': True,
            'log': True,
            'debug': False
        }

    train_name = type_names[opts['traincode']]
    test_name = type_names[opts['testcode']]
    lastQuery = int(floor(responses['nqueries'][train_name] * opts['proportion']))
    ntest = 0
    ntrain = 0
    for k,v in responses['nqueries'].items():
        if k == train_name:
            ntrain += v
        elif k == test_name:
            ntest += v

    TRAIN = []
    if opts['traincode']==0:
        TRAIN.extend(responses['RANDOM'])
    elif opts['traincode']==1:
        TRAIN.extend(responses['ADAPTIVE'])
    elif opts['traincode']==2:
        TRAIN.extend(responses['CV'])

    TEST= []
    if opts['testcode']==0:
        TEST.extend(responses['RANDOM'])
    elif opts['testcode']==1:
        TEST.extend(responses['ADAPTIVE'])
    elif opts['testcode']==2:
        TEST.extend(responses['CV'])

    TRAIN = TRAIN[:lastQuery]
    lossLog = []
    for epoch in range(opts['nepochs']):
        MDATA = {'emploss': 0, 'hingeloss': 0, 'epoch': epoch, 'ntrain': float(ntrain)}
        MDATA_test = {'emploss': 0, 'hingeloss': 0, 'epoch': epoch}
        shuffle(TRAIN)
        for i,query in enumerate(TRAIN):
            QDATA = updateModel(model, query, STEP_COUNTER)
            MDATA['emploss'] += QDATA['emploss']
            MDATA['hingeloss'] += QDATA['hingeloss']

        for i,query in enumerate(TEST):
            QDATA = evaluateModel(model, query)
            MDATA_test['emploss'] += QDATA['emploss']
            MDATA_test['hingeloss'] += QDATA['hingeloss']

        if opts['verbose'] == True:
            MDATA['emploss']
            MDATA['norm'] = norm(model) / sqrt(responses['nitems'])
            printLoss(MDATA)

        if opts['log'] == True:
            lossLog.append(
                    [
                        MDATA['emploss']/float(ntrain),
                        MDATA['hingeloss'],
                        MDATA_test['emploss']/float(ntest),
                        MDATA_test['hingeloss']
                    ]
                )

    if opts['debug']:
        return len(TRAIN)
    else:
        return lossLog

class ModelTests(unittest.TestCase):
    def setUp(self):
        self.primary = 0
        self.alternate = 1
        self.target = 2

    def testModelIsCorrect(self):
        def modelCheck(model, query):
            d1 = sqrt(sum(square(model[query[0],] - model[query[2],])))
            d2 = sqrt(sum(square(model[query[1],] - model[query[2],])))
            return(d1<d2)

        MODEL = mat([[1.0,1.0,1.0],[2.0,2.0,2.0],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        self.failUnless(modelCheck(MODEL,QUERY))

    def testModelImproves(self):
        MODEL = mat([[1.5,1.5,1.5],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        MODEL_orig = MODEL.copy()
        QUERY = [self.primary,self.alternate,self.target]
        # Remember that lists/numpy arrays are mutable, so they are updated in place.
        # updateModel() will update MODEL without needing to return a new model.
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(sum(MODEL[0,:]) < sum(MODEL_orig[0,:]))

    def testRight_eloss(self):
        # Right
        MODEL = mat([[1.0,1.0,1.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['emploss'] == 0)

    def testWrong_eloss(self):
        # Wrong
        MODEL = mat([[2.0,2.0,2.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['emploss'] == 1)

    def testEqualIsWrong_eloss(self):
        # Wrong
        MODEL = mat([[1.5,1.5,1.5],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['emploss'] == 1)

    def testRight_hloss(self):
        # Right
        MODEL = mat([[1.0,1.0,1.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['hingeloss'] == 0)

    def testWrong_hloss(self):
        # Wrong
        MODEL = mat([[2.0,2.0,2.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['hingeloss'] > 1)

    def testEqualIsWrong_hloss(self):
        # Wrong
        MODEL = mat([[1.5,1.5,1.5],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = count(1)
        QDATA = updateModel(MODEL, QUERY, STEP_COUNTER)
        self.failUnless(QDATA['hingeloss'] == 1)

    def testStepCounter(self):
        def a(c):
            for i in range(10):
                n = next(c)
            return n
        STEP_COUNTER = count(1)
        for i in range(10):
            n = a(STEP_COUNTER)
        print n
        self.failUnless(n==100)

class IOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Remember that lists are mutable, and so updates to model that take
        # place within fit model will have affects beyond the scope of the
        # function.
        data_path = 'Test/NEXT/test.csv'
        cls.responses = read_triplets(data_path)
        cls.model = initializeEmbedding(cls.responses['nitems'],3)
        cls.lossLog = fitModel(cls.model, cls.responses)
        with open(data_path, 'rb') as ifile:
            reader = csv.reader(ifile,escapechar='\\')
            header = reader.next() # reads first row
            primary = header.index('primary')
            alternate = header.index('alternate')
            target = header.index('target')
            cls.check = []
            for row in reader: # reads rest of rows
                cls.check.append([ row[i].strip() for i in (primary,alternate,target) ])

    def testRead_nitems(self):
        self.failUnless(self.responses['nitems']==7)

    def testRead_nqueries(self):
        check = {'random':2, 'adaptive':2, 'cv':2}
        match = [check[key] == val for key,val in self.responses['nqueries'].items()]
        self.failUnless(all(match))

    def testRead_ListSizes(self):
        check = (2,2,2)
        keys = ('random','adaptive','cv')
        sz = [len(self.responses[k.upper()]) for k in keys]
        print sz
        match = [sz[i] == check[i] for i in range(3)]
        self.failUnless(all(match))

#    def testRead_queryConstruction(self):
#        q = [[self.responses['labels'][i] for i in indexes] for indexes in self.responses['queries']]
#        self.failUnless(q == self.check)
#
    def testWrite_model(self):
        writeModel(self.model,'/tmp','text')
        self.failUnless(True)

    def testWrite_loss(self):
        writeLoss(self.lossLog,'/tmp','text')
        self.failUnless(True)

    def test_proportion(self):
        opts = {
            'proportion': 0.5,
            'nepochs': 10,
            'traincode': 1,
            'testcode': 2,
            'verbose': True,
            'log': True,
            'debug':True
        }
        n = fitModel(self.model, self.responses, opts)
        print '\n\n\n{n}\n\n\n'.format(n=n)
        self.failUnless(n==1)

# class ImportantTests(unittest.TestCase):
#     def test_queryCheck(self):
#         """This confirms that the internal coding scheme for items exactly
#         maps to the original data."""
#         with open('/Users/Chris/activeMDS/example/test.csv', 'rb') as ifile:
#             reader = csv.reader(ifile,escapechar='\\')
#             header = reader.next() # reads first row
#             primary = header.index('primary')
#             alternate = header.index('alternate')
#             target = header.index('target')
#             queries = []
#             for row in reader: # reads rest of rows
#                 queries.append([ row[i].strip() for i in (primary,alternate,target) ])
#
#         q = [[self.responses['labels'][i] for i in indexes] for indexes in self.responses['queries']]
#         self.failUnless(q == queries)

def main():
    unittest.main()

if __name__ == "__main__":
    main()
