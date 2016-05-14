# RESPONSE DATA
# =============
# (str) responses: Path to csv returned from NEXT containing participant
# response data.
# (str) traincode: Every response is tagged with an "Alg Label". Here you may
# specify the Alg Label associated with the responses that you want to compose
# the training set. What you specify has to match exactly (case sensitive) the
# value in the Alg Label column.
# (str) testcode: Traincode, but for the holdout cross-validation set.
responses:
  - data/queries_men.csv
  - data/queries_women.csv
traincode: Uncertainty
testcode: Test

# EMBEDDING OPTIONS
# =================
# (int) ndim: Number of dimensions for the generated embedding.
# (float) proportion: Proportion of available training responses to use when
# fitting the model. Useful for visualizing the model's performance with
# increasing number of responses.
ndim: [1,2,3,4,5]
proportion: [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# OPTIMIZATION OPTIONS
# ====================
# N.B. Optimization involves both standard and stochasitc gradient descent.

# AFFECT BOTH GRADIENT DECENT AND STOCHASTIC GRADIENT DESCENT
# -----------------------------------------------------------
# (float) max_norm : The maximum allowed norm of any one object (default equals
# 10*d)
# (float) epsilon : Parameter that controls stopping condition, exits if
# gamma<epsilon (default = 0.01)
# (int) randomRestarts: Each random restart will compute a new embedding from
# scratch with a different random seed. The embedding that fits the training
# set best is retained.
max_norm: 0
epsilon: 0.000001
randomRestarts: 10

# GRADIENT DESCENT
# ----------------
# (int) max_iters: Maximum number of iterations of SGD (default equals
# 40*len(S))
max_iter_GD: 50

# STOCHASTIC GRADIENT DESCENT
# ---------------------------
# (int) max_num_passes: Maximum number of passes over data during Stochastic
# Gradient Descent (default equals 16)
max_num_passes_SGD: 32

# ACTIVE LEANRING
# ===============
# (str) ActiveLearningMethod: specify the active learning method
# ("UncertaintySampling" or "CrowdKernel")
ActiveLearningMethod: CrowdKernel

# UNCERTAINTY SAMPLING PARAMETERS
# -------------------------------
# No parameters.

# CROWDKERNEL PARAMETERS
# ----------------------
# (float) mu: scales the regularization severity.
mu: 0.01

# MISC
# ====
# (bool) verbose: controls the amount of output while fitting embeddings.
verbose: true

# CONDOR OPTIONS
# ==============
# N.B. If running locally these can all be omitted or left blank.
# (str) executable: Full path the generateEmbedding.py (or something like it).
# (str) wrapper: Full path to run_generateEmbedding.sh (or something like it).
# (str) libfiles: A list of files that executable depends on.
# (str) PythonDist: Path to a location on /squid that references a full python
# distribution to send along with jobs.
# (str) ATLASDist: Path to a location on /squid that references ATLAS.
executable: "/home/crcox/src/nextmds/generateEmbedding.py"
wrapper: "/home/crcox/src/nextmds/run_generateEmbedding.sh"
libfiles:
  - "/home/crcox/src/nextmds/nextmds.py"
  - "/home/crcox/src/nextmds/utilsMDS.py"
  - "/home/crcox/src/nextmds/utilsCrowdKernel.py"
PythonDist: "/squid/crcox/python/python.tar.gz"
ATLASDist: "/squid/crcox/python/atlas.tar.gz"

# CONDORTOOLS INSTRUCTIONS
# ========================
# N.B. If running locally and you do not want to copy or define URLS files, set
# COPY: [] and URLS: [].
# ExpandFields : If you provided a list of value in a field above, and you list
# that field under the ExpandFields option, setupJobs.py will generate a unique
# job for each value in the field. If you specify multiple field names under
# ExpandFields, then those fields will be crossed. For more detail and advanced
# usage, see github.com/crcox/condortools/README.md.
# COPY : Fields listed under COPY should contain paths to files. These files
# will either be copied to the project's shared folder or to job-specific
# folders (if used in conjunction with ExpandFields).
# URLS : This is a CONDOR specific option. Fields listed under URLS should
# contain paths to files. These file names will be written to special files
# named URLS, that can either be in the shared folder or job specific folders
# (if used in conjunction with ExpandFields). These files need to be hosted on
# SQUID, a CHTC proxy server for delivering large files to jobs, and the file
# path should be defined wrt. /squid. 
ExpandFields:
  - responses
  - ndim
  - proportion

COPY:
  - executable
  - wrapper
  - libfiles

URLS:
  - PythonDist
  - ATLASDist
```
