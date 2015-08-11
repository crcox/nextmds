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

def read_triplets(ifile):
    reader = csv.reader(ifile,escapechar='\\')

    print "starting read..."
    QUERIES = {k: [] for k in QueryCodes.keys}

    header    = reader.next() # reads first row
    header    = [h for h in header if h not in ['targetIdent','primaryIdent','alternateIdent']]
    primary   = header.index('primary')
    alternate = header.index('alternate')
    target    = header.index('target')
    qt        = header.index('queryType')
    labels    = []
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

def runJob(jobdir, cfgfile, sharedir):
    cfgfile        = os.path.join(jobdir, cfgfile)
    lossfile       = os.path.join(jobdir, 'loss.csv')
    modelfile      = os.path.join(jobdir, 'model.csv')
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
            responses = read_triplets(f)
    except IOError:
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

    for key, path in referencedata.items():
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
