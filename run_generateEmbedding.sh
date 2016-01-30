#!/bin/bash
# script for execution of deployed applications
#
cleanup() {
  # Remove the Matlab runtime distribution
  if [ -f "python.tar.gz" ]; then
    rm -v "python.tar.gz"
  fi
  # Check the home directory for any transfered files.
  if [ -f ALLURLS ]; then
    while read url; do
      fname=$(basename "$url")
      if [ -f "$fname" ]; then
        rm -v "$fname"
      fi
    done < ALLURLS
  fi
  echo "all clean"
}

abort() {
  echo >&2 '
*************
** ABORTED **
*************
'
  echo >&2 "Files at time of error/interrupt"
  echo >&2 "--------------------------------"
  ls >&2 -l

  cleanup

  echo "An error occured. Exiting ..." >&2
  echo 1 > EXIT_STATUS
  exit 1
}

success() {
  echo '
*************
** SUCCESS **
*************
'
  cleanup
  echo 0 > EXIT_STATUS

  exit 0
}

# If an exit or interrupt occurs while the script is executing, run the abort
# function.
trap abort EXIT SIGTERM SIGKILL

set -e


## Download all large data files listed in URLS from SQUID
if [ ! -f URLS ]; then
  touch URLS
fi
if [ ! -f URLS_SHARED ]; then
  touch URLS_SHARED
fi
cat URLS URLS_SHARED > ALLURLS
cat ALLURLS
while read url; do
  wget -q "http://proxy.chtc.wisc.edu/SQUID/${url}"
done < ALLURLS

# Run the Python application
exe_name=$0
exe_dir=`dirname "$0"`
echo "Unpack Python distribution"
tar xzf "python.tar.gz"
echo "Unpack and install ATLAS libraries"
tar xzf "atlas.tar.gz"
export LD_LIBRARY_PATH="${HOME}/usr/lib64/atlas":${LD_LIBRARY_PATH}
echo "Unpack Data"
tar xzf "data.tar.gz"
export PATH=$(pwd)/python/bin:$PATH
eval "${exe_dir}/generateEmbedding.py --shared ./"

# Exit successfully. Hooray!
trap success EXIT SIGTERM SIGKILL
