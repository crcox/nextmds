import csv
import json
import os
import sys
import operator
import datetime

def read_triplets(ifile):
    reader = csv.reader(ifile,escapechar='\\')

    print "starting read..."
    header   = reader.next() # reads first row
    left     = header.index('Left')
    right    = header.index('Right')
    target   = header.index('Center')
    answer   = header.index('Answer')
    alglabel = header.index('Alg Label')
    timestamp= header.index('Timestamp')
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

        query = [ row[i].strip() for i in (primary,alternate,target,timestamp) ]
        query_type = row[alglabel].strip()

        [ labels.append(x) for x in query[:3] if not x in labels ]

        if not query_type in QUERIES.keys():
            query_type_count[query_type] = 0
            QUERIES[query_type] = []

        QUERIES[query_type].append(query)
        query_type_count[query_type] += 1

    labels.sort()
    labels = list(set(labels))
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

    #OUT = {k:[] for k in QUERIES.keys()}
    OUT = dict((k,[]) for k in QUERIES.keys())

    OUT['nitems'] = item_count
    OUT['nqueries'] = query_type_count
    OUT['labels'] = labels

    n = 0
    for k, qlist in QUERIES.items():
        for queryLabels in qlist:
            queryIndices = [ labels.index(x) for x in queryLabels[0:3] ] + [queryLabels[3]]
            OUT[k].append(queryIndices)
            n += 1

    print "done reading! n={items:d} |S|={queries:d}".format(items=item_count, queries=n)
    for i,k in enumerate(QUERIES.keys()):
        print '{i}: {k:>8s} = {n: 4d}'.format(i=i,k=k,n=query_type_count[k])

    print ''
    return OUT

def runJob(jobdir, cfgfile):
    cfgfile        = os.path.join(jobdir, cfgfile)
    modelfile      = os.path.join(jobdir, 'model.csv')
    lossfile       = os.path.join(jobdir, 'loss.csv')
    lossjson       = os.path.join(jobdir, 'loss.json')
    labelfile      = os.path.join(jobdir, 'labels.txt')
    querycountfile = os.path.join(jobdir, 'querydata.json')

    lossd = {'train':{'empirical':[], 'hinge':[]},'test':{'empirical':[], 'hinge':[]}}
    qc = {'all':{},'used':{},'nlabels':[]}

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
        #import nextmds.utilsCrowdKernel as utilsMDS
        utilsMDS = nextmds.utilsCrowdKernel
    else:
        #import nextmds.utilsMDS as utilsMDS
        utilsMDS = nextmds.utilsMDS

    queryfile = config['responses']
    try:
        with open(queryfile, 'r') as f:
            responses = read_triplets(f)
    except IOError:
        print ReadFileErrorMessage.format(path=queryfile)
        raise

    # Divide responses into test and training sets
    # and prepare for analysis
    # ============================================
    training = responses[config['traincode']]
    testing = responses[config['testcode']]

    # sort training set by datetime
    for row in training:
        row[3] = datetime.datetime.strptime(row[3], '%Y-%m-%d %H:%M:%S.%f')

    training = sorted(training, key=operator.itemgetter(3))

    # Strip Timestamps
    training = [row[0:3] for row in training]
    testing = [row[0:3] for row in testing]

    n = len(training)
    ix = max(int(n*config['proportion']),1)
    training = training[0:ix]
    print "Training set size: {n:d} ({p:.1f}%)".format(n=len(training),p=config['proportion']*100)

    # Write contents of the response structure to files
    # =================================================
    # nqueries  : counts by algorithm.
    # labels    : stimulus identifiers corrsponding to rows of the model matrix.
    # nitems    : the number of labels.
    # responses : The actual response data, as numeric indexes. Indexes are
    #            zero-based and map to labels in the order that they occur in
    #            responses['labels'].
    qc['all'] = responses['nqueries']
    qc['used'][config['traincode']] = len(training)
    qc['used'][config['testcode']] = len(testing)
    qc['nlabels'] = responses['nitems']
    with open(querycountfile, 'wb') as f:
        json.dump(qc, f)

    with open(labelfile, 'wb') as f:
        for x in responses['labels']:
            f.write(x+'\n')

    rfile = os.path.join(jobdir, 'responses_{a:s}_TRAIN.csv'.format(a=config['traincode']))
    with open(rfile, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(responses[config['traincode']])

    rfile = os.path.join(jobdir, 'responses_{a:s}_TEST.csv'.format(a=config['testcode']))
    with open(rfile, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(responses[config['testcode']])

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

        empirical, hinge, logloss = utilsMDS.getLoss(model,training)
        lossd['train']['empirical'] = empirical
        lossd['train']['hinge'] = hinge
        lossd['train']['log'] = logloss
        empirical, hinge, logloss = utilsMDS.getLoss(model, testing)
        lossd['test']['empirical'] = empirical
        lossd['test']['hinge'] = hinge
        lossd['test']['log'] = logloss

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

        empirical, hinge = utilsMDS.getLoss(model,training)
        lossd['train']['empirical'] = empirical
        lossd['train']['hinge'] = hinge
        empirical, hinge = utilsMDS.getLoss(model, testing)
        lossd['test']['empirical'] = empirical
        lossd['test']['hinge'] = hinge

    # Write detailed loss info to json structured text
    # ================================================
    try:
        with open(lossjson, 'wb') as f:
            json.dump(lossd, f)
    except IOError:
        print WriteFileErrorMessage.format(path=lossjson)
        raise

    # Write empirical loss to a simple 1-row csv
    # ==========================================
    try:
        with open(lossfile,'wb') as f:
            writer = csv.writer(f)
            writer.writerow([lossd['train']['empirical'],lossd['test']['empirical']])
    except IOError:
        print WriteFileErrorMessage.format(path=lossfile)
        raise


    # Write the model itself to a csv
    # ===============================
    # NB. Should have one row per label
    try:
        with open(modelfile,'wb') as f:
            writer = csv.writer(f)
            writer.writerows(model)
    except IOError:
        print WriteFileErrorMessage.format(path=lossfile)
        raise
