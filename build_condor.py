#!/usr/bin/env python
import os
import shutil
from distutils.core import setup
import urllib2
import hashlib
import subprocess
import pycon

def download_file(url):
    import urllib2
    import hashlib
    u = urllib2.urlopen(url)
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    files_size_dl = 0
    block_sz = 8192
    m = hashlib.md5()

    with open(file_name,'wb') as f:
        print "Downloading: {n} Bytes: {b}".format(n=file_name,b=file_size)
        while True:
            buffer_ = u.read(block_sz)
            if not buffer_:
                break
            else:
                m.update(buffer_)

            file_size_dl += len(buffer_)
            f.write(buffer_)
            status = r"{prog:10d} [{pct:3.2f}%]".format(prog=file_size_dl,file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print status,

            return m

ChtcRun_URL = "http://chtc.cs.wisc.edu/downloads/ChtcRun.tar.gz"

# Setup the build environment.
if not os.path.isdir('Build'):
    os.makedirs('Build')

os.chdir('Build')

new_build_env = False
new_chtc_md5 = download_file(ChtcRun_URL)

if os.path.isdir('.buildmd5'):
    with open('.buildmd5/ChtcRun.md5','rb') as f:
        old_chtc_md5 = f.read()

    if not new_chtc_md5.digest() == old_chtc_md5:
        new_build_env = True
        with open('.buildmd5/ChtcRun.md5','wb') as f:
            f.write(new_chtc_md5)

        with tarfile.open('ChtcRun.tar.gz','r:gz') as tf:
            tf.extractall()

else:
    new_build_env = True
    os.makedirs('.buildmd5')
    with open('.buildmd5/ChtcRun.md5','wb') as f:
        f.write(new_chtc_md5)

if new_build_env:
    print "ChtcRun has changed since last build."

# In the condor environment, the first line has to be:
#!./python277/bin/python
# So, if the main source file is going to be executed as a remote job, then the
# first line needs to be updated (if there is already a #! line) or added (if
# the source file did not specify an interpretter at all).  In either case, a
# new file <srcfile>_condor.py will be written, and this is the file that
# should be used with Condor jobs.
pycon.fixshebang('../generateEmbedding.py', 'generateEmbedding_condor.py')

# THE BELOW IS NOW ENCAPSULATED IN pycon.fixshebang
#with open('../generateEmbedding.py','r') as orig:
#    with open('generateEmbedding_condor.py','w') as new:
#        firstline = orig.readline()
#        if firstline[0:2] == '#!':
#            new.write('#!./python277/bin/python\n')
#        else:
#            new.write(firstline)
#
#        shutil.copyfileobj(orig, new)
#
#os.chmod('generateEmbedding_condor.py',0755)

new_script_md5 = hashlib.md5()
with open('generateEmbedding_condor.py', 'r') as f:
    new_script_md5.update(f.read())

    with open('.buildmd5/ChtcRun.md5','rb') as f:
        old_script_md5 = f.read()

    if not new_script_md5 == old_script_md5:
        new_build_env = True
        print "generateEmbedding_condor.py has changed since last build."
        print "Creating a new source distribution for module..."

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
        print "removing: dist/**"
        print "NEW FILE: generateEmbedding_condor.py"
        print "NEW FILE: nextmds-1.0.tar.gz"

if new_build_env:
    print "Re-compiling python module nextmds-1.0..."
    pycon.build('nextmds-1.0.tar.gz','./ChtcRun')

    # The following is now encapsulated in pycon.build
    #shutil.copy('ChtcRun/Pythonin/shared/SLIBS.tar.gz','SLIBS_base.tar.gz')
    #shutil.copy('ChtcRun/Pythonin/shared/ENV','ENV')
    #subprocess.call(['chtc_buildPythonmodules','--pversion=sl6-Python-2.7.7','--pmodules=nextmds-1.0.tar.gz'])
    #with tarfile.open('SLIBS_base.tar.gz','r:gz') as tf:
    #    tf.extractall()
    #with tarfile.open('SLIBS.tar.gz','r:gz') as tf:
    #    tf.extractall()
    #with tarfile.open('SLIBS.tar.gz','w:gz') as tf:
    #    tf.add('SS')
    #with tarfile.open('nextmds_condor-1.0.tar.gz','w:gz') as tf:
    #    tf.add('SLIBS.tar.gz')
    #    tf.add('sl6-SITEPACKS.tar.gz')
    #    tf.add('ENV')
