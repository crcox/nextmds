#!/usr/bin/env python

import nextmds as mds
import csv
import json
import os
import sys

#############################################################
#   Load data and parameters from the "master" json file    #
#############################################################
jsonfile = sys.argv[1]
with open(jsonfile,'rb') as f:
    jdat = json.load(f)

#############################################################
#     Set a current config that inherits defaults to be     #
#        potentially overwritten by individual configs      #
#############################################################
try:
    allConfigs = jdat['config']
except KeyError:
    # This just ensures there is a list to loop over.
    # The idea is to initialize currentConfig with defaults and then update
    # those with the paramaters from each config dict.
    allConfigs = [jdat['config']]

#############################################################
#  Define a root folder (current directory if not a condor  #
#                           run)                            #
#############################################################
rootdir = ''
try:
    if jdat['condor']:
        rootdir = jdat['version']
except KeyError:
    pass

#############################################################
#   Parse NEXT responses and write data to shared folder    #
#############################################################
responsepath = jdat['responses']
responses = mds.read_triplets(responsepath)

sharedir = os.path.join(rootdir,'shared')
if not os.path.isdir(sharedir):
    os.makedirs(sharedir)

#with open(os.path.join(sharedir,'querytype.txt'), 'w') as f:
#    for code in responses['querytype']:
#        f.write(str(code)+'\n')

with open(os.path.join(sharedir,'queries_random.csv'), 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(responses['RANDOM'])

with open(os.path.join(sharedir,'queries_adaptive.csv'), 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(responses['ADAPTIVE'])

with open(os.path.join(sharedir,'queries_cv.csv'), 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(responses['CV'])

with open(os.path.join(sharedir,'labels.txt'), 'w') as f:
    for label in responses['labels']:
        f.write(label+'\n')

querydata = {k: responses[k] for k in ['nqueries','nitems']}
with open(os.path.join(sharedir,'querydata.json'), 'wb') as f:
    json.dump(querydata,f)

#############################################################
#        Loop over configs (if condor, do setup only)       #
#############################################################
for i, cfg in enumerate(allConfigs):
    # Reset to defaults on each loop
    currentConfig = jdat.copy()
    try:
        # Strip out the config list if it exists
        del currentConfig['config']
    except KeyError:
        pass

    # Update defaults with the new config data
    currentConfig.update(cfg)

    outdir = os.path.join(rootdir,'{cfgnum:03d}'.format(cfgnum=i))
    if os.path.isdir(outdir):
        continue
    else:
        os.makedirs(outdir)

    with open(os.path.join(outdir,'config.json'),'wb') as f:
        json.dump(currentConfig, f, sort_keys=True, indent=2, separators=(',', ': '))

    if currentConfig['condor']:
        continue
    else: # if not a condor build, go ahead and fit the models as you go.
        nitem = querydata['nitems']
        ndim = currentConfig['ndim']

        try:
            writemode = currentConfig['writemode']
        except KeyError:
            writemode = "BinaryAndText"

        model = mds.initializeEmbedding(nitem, ndim)
        lossLog  = mds.fitModel(model, responses, currentConfig)

        mds.writeLoss(lossLog,outdir,writemode)
        mds.writeModel(model,outdir,writemode)
