import csv
import os
import struct
import unittest
from itertools import count
from math import floor
from numpy import array,dot,mat,sqrt,trace,zeros,sum,ndarray,square,insert,diff
from numpy.linalg import norm
from numpy.random import randn,shuffle
import shutil
import json
from datetime import datetime

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
        query_type_count = {'random':0,'adaptive':0,'cv':0}
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

            query_type_count[query_names[query_type]] += 1

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
        emp_str = "emploss = {:.3f}".format(sum(MDATA['emploss'])/float(len(MDATA['emploss'])))
        hinge_str = "hingeloss = {:.3f}".format(sum(MDATA['hingeloss']))
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

    ntrain_full = len(TRAIN)
    TRAIN = TRAIN[:lastQuery]
    ntrain = len(TRAIN)
    ntest = len(TEST)
    print '\n\n{ntrain_full} {ntrain} {ntest}\n\n'.format(ntrain_full=ntrain_full, ntrain=ntrain, ntest=ntest)
    lossLog = []
    funcVal = []
    epoch = 0
    while epoch < opts['maxepochs']:
        MDATA = {'emploss': [], 'hingeloss': [], 'epoch': epoch}
        MDATA_test = {'emploss': [], 'hingeloss': [], 'epoch': epoch}
        shuffle(TRAIN)
        for query in TRAIN:
            QDATA = updateModel(model, query, STEP_COUNTER)

        for query in TRAIN:
            QDATA = evaluateModel(model, query)
            MDATA['emploss'].append(QDATA['emploss'])
            MDATA['hingeloss'].append(QDATA['hingeloss'])

        for query in TEST:
            QDATA = evaluateModel(model, query)
            MDATA_test['emploss'].append(QDATA['emploss'])
            MDATA_test['hingeloss'].append(QDATA['hingeloss'])

        if opts['verbose'] == True:
            MDATA['emploss']
            MDATA['norm'] = norm(model) / sqrt(responses['nitems'])
            printLoss(MDATA)

        if opts['log'] == True:
            lossLog.append(
                    [
                        sum(MDATA['emploss'])/float(len(MDATA['emploss'])),
                        sum(MDATA['hingeloss']),
                        sum(MDATA_test['emploss'])/float(len(MDATA_test['emploss'])),
                        sum(MDATA_test['hingeloss'])
                    ]
                )

        funcVal.append(sum(MDATA[opts['stopFunc']])/float(len(MDATA[opts['stopFunc']])))
        tol = opts['tolerance']
        if epoch > 20: # Min epochs
            diffs = diff(funcVal[-10::1][::-1]) # sample most recent 10 values, and then reverse their order
            smooth_funcVal = sum(diffs)/float(len(diffs))
            if smooth_funcVal < tol:
                break
        epoch += 1

    if opts['debug']:
        return len(TRAIN)
    else:
        return lossLog

def runJob(jobdir):
    cfgfile = os.path.join(jobdir, 'config.json')
    lossfile = os.path.join(jobdir, 'loss.json')
    modelfile = os.path.join(jobdir, 'model.json')
    sharedir = 'shared'
    archivedir = 'archive'
    querycountfile = os.path.join(sharedir,'querydata.json')

    # Check if the job is setup properly.
    if not os.path.isdir(jobdir):
        print "\nERROR: {d} is not a directory.".format(d=job)
    elif not os.path.isfile(cfgfile):
        print "\nERROR: Directory {d} does not contain config.json.".format(d=job)

    with open(cfgfile,'rb') as f:
        config = json.load(f)

    # Check that the data exists
    queryfile = config['responses']
    if not os.path.isfile(queryfile):
        print "\nERROR: {f} does not exist.".format(f=queryfile)

    responses = load_response_data(sharedir)
    model = initializeEmbedding(responses['nitems'],config['ndim'])
    lossLog = fitModel(model, responses, config)

    with open(lossfile,'wb') as f:
        writer = csv.writer(f)
        writer.writerows(lossLog)

    with open(modelfile,'wb') as f:
        writer = csv.writer(f)
        writer.writerows(model)

def load_response_data(sharedir):
    randomfile = os.path.join(sharedir, 'queries_random.csv')
    adaptivefile = os.path.join(sharedir, 'queries_adaptive.csv')
    cvfile = os.path.join(sharedir, 'queries_cv.csv')
    labelfile = os.path.join(sharedir, 'labels.txt')
    qcountfile = os.path.join(sharedir, 'querydata.json')

    Q = {'RANDOM':[],'ADAPTIVE':[],'CV':[],'labels':[]}

    for xfile in [randomfile,adaptivefile,cvfile,labelfile,qcountfile]:
        if not os.path.isfile(randomfile):
            print "ERROR: {f} does not exist.".format(f=xfile)
            raise IOError

    with open(randomfile,'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            Q['RANDOM'].append(line)

    with open(adaptivefile,'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            Q['ADAPTIVE'].append(line)

    with open(cvfile,'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            Q['CV'].append(line)

    with open(labelfile,'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            Q['labels'].append(line)

    with open(qcountfile,'rb') as f:
        qdata = json.load(f)

    Q.update(qdata)

    return(Q)

def archive(archivedir):
    rootdir = os.getcwd()
    with open('master.json','rb') as f:
        jdat = json.load(f)

    allConfigs = jdat['config']

    t = datetime.now()
    t = t.replace(microsecond=0)
    ISO_time_str = datetime.isoformat(t).replace(':','')
    adir = os.path.join(rootdir,ISO_time_str)
    os.makedirs(adir)
    tarball = '{t}.tar.gz'.format(t=ISO_time_str)

    for i in range(len(allConfigs)):
        cfgdir = '{cfgnum:03d}'.format(cfgnum=i)
        targdir = os.path.join(adir,cfgdir)
        shutil.copytree(cfgdir,targdir)

    shutil.copytree('shared',os.path.join(adir,'shared'))
    shutil.copy('master.json',adir)

    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(adir)

    shutil.move(tarball,archivedir)
    shutil.rmtree(adir)

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
