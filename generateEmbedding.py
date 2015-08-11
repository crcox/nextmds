#!/usr/bin/env python

import nextmds as mds
import csv
import json
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Generate embeddings from responses.')
parser.add_argument('jobdirs', metavar='JobDir', type=str, nargs='*', default='.',
        help='One or more paths to job directories that contain a parameter file to be executed. If no paths are provided, the current directory is treated as the one and only job directory.')
parser.add_argument('--cfg', metavar="CFGFILE", type=str, nargs=1, default='params.json',
        help='The name of the file config file in each job directories. All config files must have the same name. The default is "params.json". N.B. Program only understands JSON.')
parser.add_argument('--shared', metavar="SHAREDDIR", type=str, nargs=1, default='shared',
        help='The name of the directory where shared data will be read and written. The default is "./shared"')
args = parser.parse_args()

joblist  = args.jobdirs
cfgfile  = args.cfg
sharedir = args.shared

for job in joblist:
    if not os.path.isdir(job):
        print "\nERROR: {d} is not a directory.".format(d=job)
        raise IOError

    mds.runJob(job, cfgfile, sharedir)
