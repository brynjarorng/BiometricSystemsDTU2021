from deepface import DeepFace
import os
import json
import numpy as np

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

# TODO: remember to try catch and report what failed
probe = "probe"
reference = "reference"

probes = os.listdir(probe)
references = os.listdir(reference)

probes.sort()
references.sort()
# probes = probes[:2]
# references = references[:2]

vectors = []

for i in probes:
    a = DeepFace.represent(probe + "/" + "00002_940422_fa.png", model_name="ArcFace", detector_backend="ssd")
    b = DeepFace.represent(reference + "/" + "00002_940928_fa.png", model_name="ArcFace", detector_backend="ssd")
    ver = DeepFace.verify(probe + "/" + "00002_940422_fa.png", reference + "/" + "00002_940928_fa.png", distance_metric="euclidean", model_name="ArcFace", detector_backend="ssd")

    print(np.linalg.norm(np.array(b) - np.array(a)))
    print(ver)
    exit()


print(vectors)
print(len(vectors))




exit()

with open('reference.txt', 'w') as outfile:
    formatted = []
    for probe_name in probes:
        for reference_name in references:
            formatted.append(["{}/{}".format(probe, probe_name), "{}/{}".format(reference, reference_name)])

    try:
        result  = DeepFace.verify(formatted, detector_backend = backends[1], distance_metric="euclidean", model_name="Facenet")
    except KeyboardInterrupt:
        # quit
        exit()
    except Exception as e:
        print("---------------------------")
        print(e)
        print(probe_name)
        print(reference_name)


    counter = 0
    result_list = []
    for key in result.keys():
        probe_name = formatted[counter][0].split("/")
        probe_name = probe_name[len(probe_name) - 1]
        reference_name = formatted[counter][1].split("/")
        reference_name = reference_name[len(reference_name) - 1]

        is_mated = reference_name[:5] == probe_name[:5]
        result_list.append({"probe": probe_name, "reference": reference_name, "mated": is_mated, "dissimilarity": result[key]["distance"]})
        

    # print(obj)
    json.dump(result_list, outfile)
    # exit()
            
