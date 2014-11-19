import csv
import json
import sys

with open('process.default.json','rb') as f:
    process = json.load(f)

with open('process.valid.json','rb') as f:
    validate = json.load(f)

for arg in sys.argv[1:]:
    k, v = arg.split('=')
    try:
        process[k] = v
    except KeyError:
        raise

with open('process.json','wb') as f:
    json.dump(process,f,sort_keys=True, indent=2, separators=(',', ': '))

with open('process.template','wb') as f:
    for kvp in sorted(process.items()):
        f.write('='.join([str(x) for x in kvp])+'\n')

