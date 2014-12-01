# This should be the only thing you need to change. This should match the
# "version" field of your master.json file.
IN="NEXT_food_words"

# Nothing below this line should need to be altered.
CMD="generateEmbedding.py"
TOP=$(shell pwd)
OUT=$(IN)"_out"
PAT="model.csv"
SRC="$(HOME)/src/nextmds"
DIST="$(SRC)/dist"
MOD="nextmds-1.0.tar.gz"
setup:
	cp $(SRC)/master.json .
	$(SRC)/setupJobs.py master.json
	$(SRC)/CondorStuff/setProcessTemplate.py \
		+WantFlocking=true \
		+WantGlidein=true \
		request_cpus=1 \
		request_disk=100000 \
		request_memory=1000 \
		when_to_transfer_output=ON_EXIT
	rsync -avz $(SRC) $(IN)/shared

build: Build build-env SLIBS.tar.gz

Build:
	mkdir -p Build/$(IN)/shared

build-env: Build
	cd Build && \
		wget http://chtc.cs.wisc.edu/downloads/ChtcRun.tar.gz && \
		tar xzf ChtcRun.tar.gz

.PHONEY: build-env

SLIBS.tar.gz: Build generateEmbedding.py
	cd Build
	$(SRC)/setup_condor.py sdist
	chtc_buildPythonmodules --pversion=sl6-Python-2.7.7 --pmodules=$(MOD) && \
		cp ChtcRun/Pythonin/shared/SLIBS.tar.gz $(IN)/shared/ && \
		cd $(IN)/shared/ && tar xzvf SLIBS.tar.gz && \
		cp $(TOP)/SLIBS.tar.gz $(TOP)/sl6-SITEPACKS.tar.gz . && tar xzvf SLIBS.tar.gz && \
		tar czvf SLIBS.tar.gz SS
	cd $(TOP)

dag:
	./mkdag --cmdtorun=$(CMD) --data=$(IN) --outputdir=$(OUT) --pattern=$(OUT) --type=Other

run:
	cd EXP01_out && condor_submit_dag mydag.dag

clean:
	rm -rf $(IN) process.json process.template
