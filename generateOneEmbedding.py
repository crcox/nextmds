import activeMDS as mds
import csv
import json

jsonfile = 'config.json'

with open(jsonfile,'rb') as f:
    config = json.load(f)

responses = mds.read_triplets(config['data'])
model = mds.initializeEmbedding(responses['nitems'],config['ndim'])
lossLog = mds.fitModel(model, responses, config)
with open('loss.csv','wb') as f:
    writer = csv.writer(f)
    writer.writerows(lossLog)

with open('embedding.csv','wb') as f:
    writer = csv.writer(f)
    writer.writerows(model)
