#!/usr/bin/env python
import os
import shutil
from distutils.core import setup

# In the condor environment, the first line has to be:
#!./python277/bin/python
# So, if the main source file is going to be executed as a remote job, then the
# first line needs to be updated (if there is already a #! line) or added (if
# the source file did not specify an interpretter at all).
# In either case, a new file <srcfile>_condor.py will be written, and this is
# the file that should be used with Condor jobs.
with open('generateEmbedding.py','r') as orig:
    with open('generateEmbedding_condor.py','w') as new:
        firstline = orig.readline()
        if firstline[0:2] != '#!':
            new.write(firstline)

        new.write('#!./python277/bin/python\n')
        shutil.copyfileobj(orig, new)

os.chmod('generateEmbedding_condor.py',0755)

# Now we need to 
setup(name='nextmds',
      version='1.0',
      description='A collection of functions for generating embeddings from NEXT.discovery response data.',
      author='Kevin Jamieson and Chris Cox',
      author_email='crcox@wisc.edu',
      py_modules=['nextmds']
     )

os.remove('MANIFEST')
shutil.move('dist/nextmds-1.0.tar.gz','nextmds-1.0.tar.gz')
shutil.rmtree('dist')

print "NEW FILE: generateEmbedding_condor.py"
print "NEW FILE: nextmds-1.0.tar.gz"
