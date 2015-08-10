import csv
import json
import os
import struct
import sys
import itertools
import numpy
import numpy.linalg
import numpy.random
import unittest
from datetime import datetime
import utilsMDS

class QueryCodeStruct:
    def __init__(self,keys):
        self.keys = keys
        self.n = len(self.keys)

    def translate(self,qcode):
        return self.keys[qcode]

QueryCodes = QueryCodeStruct(keys=('RANDOM','ADAPTIVE','CV'))

def updateModel(model,query,reg,STEP_COUNTER):
    def StochasticGradient(Xq):
        dims = model.shape[1]
        H = numpy.mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])

        g = numpy.zeros( (3,dims) )
        emploss = 0
        hingeloss = 0

        loss_ijk = numpy.trace(numpy.dot(H,numpy.dot(Xq,Xq.T)))
        if loss_ijk + 1. >= 0:
            hingeloss = loss_ijk + 1.
            g = g + H * Xq

            if loss_ijk >= 0:
                emploss = 1.

        return g, emploss, hingeloss, loss_ijk

    def regularize(Xq, regularization=10):
        for i in range(3):
            norm_i = numpy.linalg.norm(Xq[i, :])
            if norm_i > regularization:
                Xq[i, :] *= (regularization / norm_i)

        return Xq

    # Select the model elements relevant to the current query.
    Xq = model[query, :]
    stepSize  = numpy.sqrt(100.)/numpy.sqrt(next(STEP_COUNTER) + 100.)
    g,emploss,hingeloss,loss = StochasticGradient(Xq)
    Xq -= (stepSize * g)  # max-norm
    Xq = regularize(Xq,reg)
    model[query, :] = Xq

    return {'gradient':g,'emploss':emploss,'hingeloss':hingeloss,'loss':loss}

def evaluateModel(model,query):
    def computeLoss(Xq):
        H = numpy.mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
        loss_ijk = numpy.trace(numpy.dot(H,numpy.dot(Xq,Xq.T)))
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
        QUERIES = {k: [] for k in QueryCodes.keys}

        header = reader.next() # reads first row
        header = [h for h in header if h not in ['targetIdent','primaryIdent','alternateIdent']]
        primary = header.index('primary')
        alternate = header.index('alternate')
        target = header.index('target')
        qt = header.index('queryType')
        labels = []
        query_type_count = {k: 0 for k in QueryCodes.keys}
        for row in reader: # reads rest of rows
            if not len(row) == len(header):
                raise IndexError
            query = [ row[i].strip() for i in (primary,alternate,target) ]
            query_type = int(row[qt].strip())
            [ labels.append(x) for x in query if not x in labels ]

            QUERIES[QueryCodes.translate(query_type)].append(query)
            query_type_count[QueryCodes.translate(query_type)] += 1

    item_count = len(labels)
    intconv = False
    try:
        # If all labels are actually integers, convert
        # so they end up in numeric order after sort.
        labels = [int(x) for x in labels]
        intconv = True
        print 'Labels appear to be numeric. Sorting into ascending order.'
    except ValueError:
        print 'Labels appear to be strings. Sorting alphabetically.'
        pass

    labels.sort()

    if intconv:
        # Put them back as strings, so they can match
        # against queries.
        labels = [str(x) for x in labels]

    OUT = {k:[] for k in QueryCodes.keys}
    OUT['nitems'] = item_count
    OUT['nqueries'] = query_type_count
    OUT['labels'] = labels

    n = 0
    for k, qlist in QUERIES.items():
        for queryLabels in qlist:
            queryIndices = [ labels.index(x) for x in queryLabels ]
            OUT[k].append(queryIndices)
            n += 1

    print "done reading! n={items:d} |S|={queries:d}".format(items=item_count, queries=n)
    for i,k in enumerate(QueryCodes.keys):
        print '{i}: {k:>8s} = {n: 4d}'.format(i=i,k=k,n=query_type_count[k])

    return OUT

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
    lossLog = numpy.array(lossLog)
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
    model = numpy.random.randn(nitems,dimensions)
    model = model/numpy.linalg.norm(model)*numpy.sqrt(nitems)
    return model

def fitModel(model, responses, opts=False):
    #### NOTE! This function is deprecated in favor of code in utilsMDS, which is part of the NEXT codebase.
    STEP_COUNTER = itertools.count(1)
    type_names = ('random','adaptive','cv')
    def printLoss(MDATA):
        epoch_str = "epoch = {:2d}".format(MDATA['epoch'])
        emp_str = "emploss = {:.3f}".format(numpy.sum(MDATA['emploss'])/float(len(MDATA['emploss'])))
        hinge_str = "hingeloss = {:.3f}".format(numpy.sum(MDATA['hingeloss']))
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
    lastQuery = int(numpy.floor(responses['nqueries'][train_name] * opts['proportion']))

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

    MDATA = {'emploss': [], 'hingeloss': [], 'epoch': epoch}
    MDATA_test = {'emploss': [], 'hingeloss': [], 'epoch': epoch}
    for query in TRAIN:
        QDATA = evaluateModel(model, query)
        MDATA['emploss'].append(QDATA['emploss'])
        MDATA['hingeloss'].append(QDATA['hingeloss'])

    for query in TEST:
        QDATA = evaluateModel(model, query)
        MDATA_test['emploss'].append(QDATA['emploss'])
        MDATA_test['hingeloss'].append(QDATA['hingeloss'])

    lossLog.append(
            [
                numpy.sum(MDATA['emploss'])/float(len(MDATA['emploss'])),
                numpy.sum(MDATA['hingeloss']),
                numpy.sum(MDATA_test['emploss'])/float(len(MDATA_test['emploss'])),
                numpy.sum(MDATA_test['hingeloss'])
            ]
        )

    while epoch < opts['maxepochs']:
        MDATA = {'emploss': [], 'hingeloss': [], 'epoch': epoch}
        MDATA_test = {'emploss': [], 'hingeloss': [], 'epoch': epoch}
        numpy.random.shuffle(TRAIN)
        for query in TRAIN:
            QDATA = updateModel(model, query, opts['regularize'], STEP_COUNTER)

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
            MDATA['norm'] = numpy.linalg.norm(model) / numpy.sqrt(responses['nitems'])
            printLoss(MDATA)

        if opts['log'] == True:
            lossLog.append(
                    [
                        numpy.sum(MDATA['emploss'])/float(len(MDATA['emploss'])),
                        numpy.sum(MDATA['hingeloss']),
                        numpy.sum(MDATA_test['emploss'])/float(len(MDATA_test['emploss'])),
                        numpy.sum(MDATA_test['hingeloss'])
                    ]
                )

        funcVal.append(numpy.sum(MDATA[opts['stopFunc']])/float(len(MDATA[opts['stopFunc']])))
        tol = opts['tolerance']
        if epoch > 20: # Min epochs
            diffs = numpy.diff(funcVal[-10::1][::-1]) # sample most recent 10 values, and then reverse their order
            smooth_funcVal = numpy.sum(diffs)/float(len(diffs))
            if smooth_funcVal < tol:
                break
        epoch += 1

    if opts['debug']:
        return len(TRAIN)
    else:
        return lossLog

def runJob(jobdir):
    cfgfile = os.path.join(jobdir, 'params.json')
    lossfile = os.path.join(jobdir, 'loss.csv')
    modelfile = os.path.join(jobdir, 'model.csv')
    sharedir = 'shared'
    archivedir = 'archive'
    querycountfile = os.path.join(sharedir,'querydata.json')

    # Check if the job is setup properly.
    if not os.path.isdir(jobdir):
        print "\nERROR: {d} is not a directory.".format(d=job)
    elif not os.path.isfile(cfgfile):
        print "\nERROR: Directory {d} does not contain params.json.".format(d=job)

    with open(cfgfile,'rb') as f:
        config = json.load(f)

    # Check that the data exists
    queryfile = config['responses']
    if not os.path.isfile(queryfile):
        print "\nERROR: {f} does not exist.".format(f=queryfile)

    # If the data has not already been parsed, parse it into shared.
    OUT=read_triplets(config['responses'])
    randomfile = os.path.join(sharedir, 'queries_random.csv')
    adaptivefile = os.path.join(sharedir, 'queries_adaptive.csv')
    cvfile = os.path.join(sharedir, 'queries_cv.csv')
    labelfile = os.path.join(sharedir, 'labels.txt')
    qcountfile = os.path.join(sharedir, 'querydata.json')
    with open(randomfile,'wb') as f:
        writer = csv.writer(f)
        writer.writerows(OUT['RANDOM'])
    with open(adaptivefile,'wb') as f:
        writer = csv.writer(f)
        writer.writerows(OUT['ADAPTIVE'])
    with open(cvfile,'wb') as f:
        writer = csv.writer(f)
        writer.writerows(OUT['CV'])
    with open(labelfile,'wb') as f:
        for x in OUT['labels']:
            f.write(x+'\n')
    with open(qcountfile,'wb') as f:
        json.dump(OUT['nqueries'],f)

    #responses = load_response_data(sharedir)
    responses = OUT
    #CRC code
    #model = initializeEmbedding(responses['nitems'],config['ndim'])
    #lossLog = fitModel(model, responses, config)
    # Kevin code
    print QueryCodes.translate(config['traincode'])
    print QueryCodes.translate(config['testcode'])
    training = responses[QueryCodes.translate(config['traincode'])]
    testing = responses[QueryCodes.translate(config['testcode'])]
    n = len(training)
    ix = max(int(numpy.floor(n*config['proportion'])),1)
    training = training[0:ix]
    model, trainloss = utilsMDS.computeEmbedding(responses['nitems'],config['ndim'],
            S=training,
            max_num_passes_SGD=config['max_num_passes_SGD'],
            max_iter_GD=config['max_iter_GD'],
            num_random_restarts=config['randomRestarts'],
            verbose=config['verbose'],
            epsilon=config['epsilon'])
    trainloss, hinge_loss = utilsMDS.getLoss(model,training)
    testloss, hinge_loss = utilsMDS.getLoss(model,testing)

    with open(lossfile,'wb') as f:
        writer = csv.writer(f)
        #writer.writerows(lossLog)
        writer.writerow([trainloss,testloss])

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
            Q['RANDOM'].append([int(x) for x in line])

    with open(adaptivefile,'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            Q['ADAPTIVE'].append([int(x) for x in line])

    with open(cvfile,'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            Q['CV'].append([int(x) for x in line])

    with open(labelfile,'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            Q['labels'].append(line)

    with open(qcountfile,'rb') as f:
        qdata = json.load(f)

    Q.update(qdata)

    return(Q)

def archive(archivedir):
    import shutil

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
        self.reg=10

    def testModelIsCorrect(self):
        def modelCheck(model, query):
            d1 = numpy.sqrt(numpy.sum(numpy.square(model[query[0],] - model[query[2],])))
            d2 = numpy.sqrt(numpy.sum(numpy.square(model[query[1],] - model[query[2],])))
            return(d1<d2)

        MODEL = numpy.mat([[1.0,1.0,1.0],[2.0,2.0,2.0],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        self.failUnless(modelCheck(MODEL,QUERY))

    def testModelImproves(self):
        MODEL = numpy.mat([[1.5,1.5,1.5],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        MODEL_orig = MODEL.copy()
        QUERY = [self.primary,self.alternate,self.target]
        # Remember that lists/numpy arrays are mutable, so they are updated in place.
        # updateModel() will update MODEL without needing to return a new model.
        STEP_COUNTER = itertools.count(1)
        QDATA = updateModel(MODEL, QUERY, 10, STEP_COUNTER)
        self.failUnless(numpy.sum(MODEL[0,:]) < numpy.sum(MODEL_orig[0,:]))

    def testRight_eloss(self):
        # Right
        MODEL = numpy.mat([[1.0,1.0,1.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = itertools.count(1)
        QDATA = updateModel(MODEL, QUERY, 10, STEP_COUNTER)
        self.failUnless(QDATA['emploss'] == 0)

    def testWrong_eloss(self):
        # Wrong
        MODEL = numpy.mat([[2.0,2.0,2.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = itertools.count(1)
        QDATA = updateModel(MODEL, QUERY, 10, STEP_COUNTER)
        self.failUnless(QDATA['emploss'] == 1)

    def testEqualIsWrong_eloss(self):
        # Wrong
        MODEL = numpy.mat([[1.5,1.5,1.5],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = itertools.count(1)
        QDATA = updateModel(MODEL, QUERY, 10, STEP_COUNTER)
        self.failUnless(QDATA['emploss'] == 1)

    def testRight_hloss(self):
        # Right
        MODEL = numpy.mat([[1.0,1.0,1.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = itertools.count(1)
        QDATA = updateModel(MODEL, QUERY, 10, STEP_COUNTER)
        self.failUnless(QDATA['hingeloss'] == 0)

    def testWrong_hloss(self):
        # Wrong
        MODEL = numpy.mat([[2.0,2.0,2.0],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = itertools.count(1)
        QDATA = updateModel(MODEL, QUERY, 10, STEP_COUNTER)
        self.failUnless(QDATA['hingeloss'] > 1)

    def testEqualIsWrong_hloss(self):
        # Wrong
        MODEL = numpy.mat([[1.5,1.5,1.5],[1.5,1.5,1.5],[0.0,0.0,0.0]])
        QUERY = [self.primary,self.alternate,self.target]
        STEP_COUNTER = itertools.count(1)
        QDATA = updateModel(MODEL, QUERY, 10, STEP_COUNTER)
        self.failUnless(QDATA['hingeloss'] == 1)

    def testStepCounter(self):
        def a(c):
            for i in range(10):
                n = next(c)
            return n
        STEP_COUNTER = itertools.count(1)
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
