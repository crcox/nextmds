CMD="generateOneEmbedding.py"
IN="EXP01"
OUT=$(IN)"_out"
PAT="embedding.csv"

setup:
	python embedding_generator.py master.json
	python setProcessTemplate.py \
		+WantFlocking=true \
		+WantGlidein=true \
		request_cpus=1 \
		request_disk=1000000 \
		request_memory=1000 \
		when_to_transfer_output=ON_EXIT

run:
	./mkdag --cmdtorun=$(CMD) --data=$(IN) --outputdir=$(OUT) --pattern=$(OUT)

clean:
	rm -rf $(OUT)
