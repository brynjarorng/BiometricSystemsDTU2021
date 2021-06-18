import cv2
import os
import numpy as np
import random

import face_recognition

files_to_alter = os.listdir("reference")
use_radius_filter = True
print_roi_rect = True

"""
	0: Mean filter
	1: Median filter
"""
filter_type = 1
filter_radius = 10

for i in files_to_alter:
	image = face_recognition.load_image_file("reference/{}".format(i))
	face_landmarks = face_recognition.face_landmarks(image)
	face_locations = face_recognition.face_locations(image)
	# print(face_landmarks[0])

	for (top, right, bottom, left) in face_locations:
		if print_roi_rect:
			cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 5)
	
	rows = bottom - top
	columns = right - left
	pixels_in_image = rows * columns
	# print(pixels_in_image)

	"""filters with radius"""
	if use_radius_filter:
		# TODO: find ceiling function
		for i in range(int(columns / (filter_radius * 2)) + 1):
			for j in range(int(rows / (filter_radius * 2)) + 1):
				# Find center points to calc on
				x = left + filter_radius * (j * 2)
				y = top + filter_radius * (i * 2)

				# Do filtering based on n nearest neighbours, don't go outside border
				min_x = x-filter_radius
				if min_x <= left:
					min_x = left+1
				max_x = x+filter_radius+1
				if max_x >= right:
					max_x = right-1
				min_y = y-filter_radius
				if min_y <= top:
					min_y = top+1
				max_y = y+filter_radius+1
				if max_y >= bottom:
					max_y = bottom-1

				img_slice = image[min_y:max_y, min_x:max_x]

				num_pixels = len(img_slice) * len(img_slice[0])

				# Mean filter
				if filter_type == 0:
					r = 0
					g = 0
					b = 0
					for p in img_slice:
						for q in p:
							r += q[0]
							g += q[1]
							b += q[2]
					r = int(r / num_pixels)
					g = int(g / num_pixels)
					b = int(b / num_pixels)
					image[min_y:max_y, min_x:max_x] = [r, g, b]

				# Median filter
				elif filter_type == 1:
					r = []
					g = []
					b = []
					for p in img_slice:
						for q in p:
							r.append(q[0])
							g.append(q[1])
							b.append(q[2])
					r = np.median(r)
					g = np.median(g)
					b = np.median(b)
					image[min_y:max_y, min_x:max_x] = [r, g, b]
	else:
		"""Filters without radius"""
		for i in range(columns):
			for j in range(rows):
				x = left + j
				y = top + i

				# Salt and pepper noise
				nois_probability = 0.05
				if random.random() < nois_probability:
					# salt or pepper
					if random.random() > 0.5:
						image[y,x] = [255, 255, 255]
					else:
						image[y,x] = [0, 0, 0]



	cv2.imshow('frame', image)
	cv2.waitKey(0)
	exit()





