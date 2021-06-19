from deepface import DeepFace

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

result  = DeepFace.verify("reference/00424_940422_fa.png", "output/00424_940422_fa.png", detector_backend = backends[1])
print("Is verified: ", result)