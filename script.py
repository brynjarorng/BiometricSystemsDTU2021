from deepface import DeepFace
result  = DeepFace.verify("probe/00002_940422_fa.png", "reference/00002_940928_fa.png", distance_metric="euclidean")
print("Is verified: ", result)