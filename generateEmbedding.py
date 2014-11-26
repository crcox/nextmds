#!/usr/bin/env python

import nextmds as mds
import csv
import json
import sys
import os

# This script takes either no arguments or a list of job directories as
# arguments. Passing no arguments is the same as passing './', indicating that
# the current directory contains the instructions for the job.

if len(sys.argv) > 1:
    joblist = sys.argv[1:]
else:
    joblist = ['./']

cfgfile = 'config.json'
for job in joblist:
    if not os.path.isdir(job):
        print "\nERROR: {d} is not a directory.".format(d=job)
        raise IOError
    elif not os.path.isfile(os.path.join(job,cfgfile)):
        print "\nERROR: Directory {d} does not contain config.json.".format(d=job)
        raise IOError

    mds.runJob(job)
