import csv
import json
import os
import sys

def read_triplets(ifile):
    reader = csv.reader(ifile,escapechar='\\')

    print "starting read..."
    header   = reader.next() # reads first row
    left     = header.index('Left')
    right    = header.index('Right')
    target   = header.index('Center')
    answer   = header.index('Answer')
    alglabel = header.index('Alg Label')
    labels   = []
    QUERIES  = {}
    query_type_count = {}
    for row in reader: # reads rest of rows
        if not len(row) == len(header):
            raise IndexError

        if row[left] == row[answer]:
            primary = left
            alternate = right
        else:
            primary = right
            alternate = left

        query = [ row[i].strip() for i in (primary,alternate,target) ]
        query = [ row[i].strip() for i in (left,right,target) ]
        query_answer = row[answer].strip()
        query_type = row[alglabel].strip()

        [ labels.append(x) for x in query if not x in labels ]

        if not query_type in QUERIES.keys():
            query_type_count[query_type] = 0
            QUERIES[query_type] = []

        QUERIES[query_type].append(query)
        query_type_count[query_type] += 1

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

    OUT = {k:[] for k in QUERIES.keys()}
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
    for i,k in enumerate(QUERIES.keys()):
        print '{i}: {k:>8s} = {n: 4d}'.format(i=i,k=k,n=query_type_count[k])

    print ''
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

    useCrowdKernel = config['ActiveLearningMethod'].lower() in ['crowdkernel','ck']
    if useCrowdKernel:
        import utilsCrowdKernel as utilsMDS
    else:
        import utilsMDS as utilsMDS

    queryfile = config['responses']
    try:
        with open(queryfile, 'r') as f:
            responses = read_triplets(f)
    except IOError:
        print ReadFileErrorMessage.format(path=queryfile)
        raise

    referencedata = {
            "labels"   : os.path.join(sharedir, 'labels.txt'),
            "nqueries" : os.path.join(sharedir, 'querydata.json')
        }
    for AlgLab in responses.keys():
        referencedata[AlgLab] = os.path.join(sharedir, 'responses_{a:s}.csv'.format(a=AlgLab))

    if not os.path.isdir(sharedir):
        os.makedir(sharedir)

    for key, path in referencedata.items():
        try:
            with open(path, 'wb') as f:
                if key is 'nqueries':
                    json.dump(responses[key],f)
                elif key is 'labels':
                    for x in responses['labels']:
                        f.write(x+'\n')
                elif key is 'nitems':
                    pass
                else:
                    writer = csv.writer(f)
                    writer.writerows(responses[key])
        except IOError:
            print WriteFileErrorMessage.format(path=path)
            raise

    training = responses[config['traincode']]
    testing = responses[config['testcode']]

    n = len(training)
    ix = max(int(n*config['proportion']),1)
    training = training[0:ix]
    print "Training set size: {n:d} ({p:.1f}%)".format(n=len(training),p=config['proportion']*100)

    if useCrowdKernel:
        if 'max_iter_GD' in config.keys() and config['max_iter_GD']:
            print "Value for `max_iter_GD` is being ignored. Not exposed as argument to CrowdKernel."
        model, trainloss = utilsMDS.computeEmbedding(
                n = responses['nitems'],
                d = config['ndim'],
                S = training,
                mu = config['mu'],
                num_random_restarts = config['randomRestarts'],
                max_num_passes = config['max_num_passes_SGD'],
                max_norm = config['max_norm'],
                epsilon = config['epsilon'],
                verbose = config['verbose'])

        trainloss, hinge_loss, trainlogloss = utilsMDS.getLoss(model,training)
        testloss, hinge_loss, testlogloss = utilsMDS.getLoss(model,testing)

    else:
        if 'mu' in config.keys() and config['mu']:
            print "Value for `mu` is being ignored. Only relevant to CrowdKernel."
        if 'max_norm' in config.keys() and config['max_norm']:
            print "Value for `max_norm` is being ignored. Only relevant to CrowdKernel."

        model, trainloss = utilsMDS.computeEmbedding(
                n = responses['nitems'],
                d = config['ndim'],
                S = training,
                max_num_passes = config['max_num_passes_SGD'],
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
