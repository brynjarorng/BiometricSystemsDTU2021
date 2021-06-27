import os

ref = os.listdir("reference")
probe = os.listdir("probe")

ref_set = set()
probe_set = set()

for i in ref:
    ref_set.add(i[:5])
for i in probe:
    probe_set.add(i[:5])

print("reference: {}, unique: {}".format(len(ref), len(ref_set)))
print("probe: {}, unique: {}".format(len(probe), len(probe_set)))
print("unique: {}".forat(len(probe_set.intersection(ref_set))))