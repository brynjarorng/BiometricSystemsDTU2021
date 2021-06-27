import json
import numpy as np


probe_file = "representations/probe_vectors.txt"
reference_files = [
    "representations/output_pixelation_high_vectors.txt",
    "representations/output_pixelation_med_vectors.txt",
    "representations/output_pixelation_low_vectors.txt",
    "representations/output_bars_high_vectors.txt",
    "representations/output_bars_low_vectors.txt",
    "representations/reference_vectors.txt"]


with open(probe_file, 'r') as infile:
    probe_data = np.array(json.load(infile))


for f in reference_files:
    with open(f, 'r') as infile:
        reference_data = json.load(infile)

    dists_mated = []
    dists_nonmated = []


    for i in probe_data:
        probe_name = i["name"][:5]
        for j in reference_data:
            reference_name = j["name"][:5]

            dist = np.linalg.norm(np.array(i["vector"]) - np.array(j["vector"]))

            if probe_name == reference_name:
                dists_mated.append(dist)
            else:
                dists_nonmated.append(dist)



    with open("{}_{}".format(f.split(".")[0], "mated.txt"), 'a') as outfile:
        for i in dists_mated:
            outfile.write(str(i)+"\n")
    with open("{}_{}".format(f.split(".")[0], "nonmated.txt"), 'w') as outfile:
        for i in dists_nonmated:
            outfile.write(str(i)+"\n")

