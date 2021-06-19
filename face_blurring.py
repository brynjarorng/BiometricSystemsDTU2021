import cv2
import os
import numpy as np
import random

import face_recognition

files_to_alter = os.listdir("reference")

"""
	0: Radius based filters
	1: Salt and pepper filter
	2: Black eye bars filter
"""
filter_selector = 2

print_roi_rect = False
save_image = False
show_image = True

"""
	0: Mean filter
	1: Median filter <- Use this one
"""
filter_type = 1

"""
	high value: 20
	med value: 10
	low value: 5
"""
filter_radius = 20	# Calculate number of boxes hor and ver on the face

for image_name in files_to_alter:
	image = face_recognition.load_image_file("reference/{}".format(image_name))
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
	if filter_selector == 0:
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
	elif filter_selector == 1:
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

	elif filter_selector == 2:
		pass



	# Image saving and displaying
	if show_image:
		cv2.imshow('frame', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		cv2.waitKey(0)
	
	if save_image:
		cv2.imwrite("output/SP_low/{}".format(image_name), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	exit()





