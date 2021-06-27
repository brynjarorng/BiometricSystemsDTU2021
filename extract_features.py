from deepface import DeepFace
import os
import json

det = ["retinaface", "ssd"]

dirs = ["reference", "probe", "output/pixelation_low", "output/pixelation_med", "output/pixelation_high", "output/bars_low", "output/bars_high"]

total_missed = {}

for d in dirs:
    print(d)
    print("--------------")

    dir = d

    files = os.listdir(dir)

    files.sort()

    vectors = []

    progress_counter = 0
    total_loops = len(files)

    for i in files:
        try:
            vec = DeepFace.represent(dir + "/" + i, model_name="ArcFace", detector_backend=det[0])
            vectors.append({"vector": vec, "name": i})
        except:
            total_missed.setdefault(d, 0)
            total_missed[d] += 1
            print(total_missed)
        
        
        # Print progress
        progress_counter += 1
        if progress_counter % 10 == 0:
            print("Progress: {}/{} - {:.2f}%".format(progress_counter, total_loops, (progress_counter/total_loops*100)))



    with open("representations/{}_vectors.txt".format(dir.replace("/", "_")), 'w') as outfile:
        json.dump(vectors, outfile)

print(total_missed)