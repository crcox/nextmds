import csv
import json
import os
import sys
import numpy
import unittest
import utilsMDS

class QueryCodeStruct:
    def __init__(self,keys):
        self.keys = keys
        self.n = len(self.keys)

    def translate(self,qcode):
        return self.keys[qcode]

QueryCodes = QueryCodeStruct(keys=('RANDOM','ADAPTIVE','CV'))

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

def runJob(jobdir):
    cfgfile        = os.path.join(jobdir, 'params.json')
    lossfile       = os.path.join(jobdir, 'loss.csv')
    modelfile      = os.path.join(jobdir, 'model.csv')
    sharedir       = 'shared'
    archivedir     = 'archive'
    querycountfile = os.path.join(sharedir,'querydata.json')

    WriteFileErrorMessage = """
{path:s} could not be written. Check permissions and that folders along the
intended path exist.
"""
    ReadFileErrorMessage = """
{path:s} could not be read. Check permissions and that the file exists in the
expected location.
"""

    try:
        with open(cfgfile,'rb') as f:
            config = json.load(f)
    except IOError:
        print ReadFileErrorMessage.format(path=cfgfile)
        raise

    queryfile = config['responses']
    try:
        with open(queryfile, 'r') as f:
            responses = read.triplets(f)
    except:
        print ReadFileErrorMessage.format(path=queryfile)
        raise

    referencedata = {
            "RANDOM"   : os.path.join(sharedir, 'queries_random.csv'),
            "ADAPTIVE" : os.path.join(sharedir, 'queries_adaptive.csv'),
            "CV"       : os.path.join(sharedir, 'queries_cv.csv'),
            "labels"   : os.path.join(sharedir, 'labels.txt'),
            "nqueries" : os.path.join(sharedir, 'querydata.json')
        }

    if not os.path.isdir(sharedir):
        os.makedir(sharedir)

    for key, path in referencedata.items:
        try:
            with open(path, 'wb') as f:
                if key is 'nqueries':
                    json.dump(responses[key],f)
                else:
                    writer = csv.writer(f)
                    writer.writerows(responses[key])
        except IOError:
            print WriteFileErrorMessage.format(path=path)
            raise

    print QueryCodes.translate(config['traincode'])
    print QueryCodes.translate(config['testcode'])

    training = responses[QueryCodes.translate(config['traincode'])]
    testing = responses[QueryCodes.translate(config['testcode'])]

    n = len(training)
    ix = max(int(numpy.floor(n*config['proportion'])),1)
    training = training[0:ix]

    model, trainloss = utilsMDS.computeEmbedding(
            n = responses['nitems'],
            d = config['ndim'],
            S = training,
            max_num_passes_SGD = config['max_num_passes_SGD'],
            max_iter_GD = config['max_iter_GD'],
            num_random_restarts = config['randomRestarts'],
            verbose = config['verbose'],
            epsilon = config['epsilon'])

    trainloss, hinge_loss = utilsMDS.getLoss(model,training)
    testloss, hinge_loss = utilsMDS.getLoss(model,testing)

    try:
        with open(lossfile,'wb') as f:
            writer = csv.writer(f)
            writer.writerow([trainloss,testloss])
    except IOError:
        print WriteFileErrorMessage.format(path=lossfile)
        raise

    try:
        with open(modelfile,'wb') as f:
            writer = csv.writer(f)
            writer.writerows(model)
    except IOError:
        print WriteFileErrorMessage.format(path=lossfile)
        raise

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
