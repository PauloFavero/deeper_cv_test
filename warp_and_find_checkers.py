import os
import glob
import cv2
import json
import argparse
import numpy as np

def projective_trans(image, coordinates, board_width_to_board_height):
	rect = np.zeros((4, 2), dtype = "float32")
	rect[0] = coordinates[0]
	rect[1] = coordinates[1]
	rect[2] = coordinates[2]
	rect[3] = coordinates[3]

	(tl, tr, br, bl) = rect

	width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	max_width = max(int(width_A), int(width_B))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	max_height = max(int(heightA), int(heightB))

	max_height = int(max_width/board_width_to_board_height)


	dst = np.array([
		[0, 0],
		[max_width, 0],
		[max_width, max_height],
		[0, max_height]], dtype = "float32")

	# compute the perspective transform matrix
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (int(max_width*1.05), int(max_height*1.03)))
	
	return warped, M

parser = argparse.ArgumentParser('Warp Image script')
parser.add_argument('indir', type=str)
parser.add_argument('outdir', type=str)

args = parser.parse_args()

input_dir = args.indir
output_dir = args.outdir

input_imgs = list(glob.glob(os.path.join(input_dir, "*.jpg")))

for img_path in input_imgs:
	#read image
	img_original = cv2.imread(img_path)
	# metadata path
	data_path = os.path.join(img_path + ".info.json")
	# read metadata file
	with open(data_path) as json_file:
		json_data = json.load(json_file)
	# extract metadata
	board_coord = json_data["canonical_board"]["tl_tr_br_bl"]
	bar_width_to_checker_width = json_data["canonical_board"]["bar_width_to_checker_width"]
	board_width_to_board_height = json_data["canonical_board"]["board_width_to_board_height"]
	pip_length_to_board_height = json_data["canonical_board"]["pip_length_to_board_height"]
	
	(tl, tr, br, bl) = board_coord
	width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	max_width = max(int(width_A), int(width_B))

	max_height = int(max_width/board_width_to_board_height)

	# objects dimensions
	checker_diameter = int(max_width/(12 + bar_width_to_checker_width))
	bar_width = int((bar_width_to_checker_width*checker_diameter)/2)
	pip_length = int(5*checker_diameter)

	warped, transform = projective_trans(img_original, board_coord, board_width_to_board_height)
	# cv2.imshow("img_original", img_original)
	# cv2.imshow("warped", warped)
	# cv2.waitKey(0)

	#transform warped image to grayscale
	gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	#compute the circles
	circles = cv2.HoughCircles(gray_warped,cv2.HOUGH_GRADIENT,1,int(checker_diameter/2),
                            param1=50,param2=30,minRadius=int((checker_diameter/2)*0.9),maxRadius=int((checker_diameter/2)*1.1))
	circles = np.uint16(np.around(circles))

	for i in circles[0,:]:
		# draw the outer circle
		cv2.circle(warped,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
		cv2.circle(warped,(i[0],i[1]),2,(0,0,255),3)

	#image size 
	height, width = warped.shape[:2]
	#half board width
	half_width = int( width/2 - bar_width)
	#half board height
	half_height = int( height/2)

	top_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	bottom_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	#for each pip on the bottom left
	for n_pip in range(0,6):
		tl_pip = (n_pip*checker_diameter, height - pip_length)
		br_pip = (tl_pip[0] + checker_diameter, height)
		# cv2.rectangle(warped, tl_pip, br_pip, color=(255,0,0), thickness=5)
		for i in circles[0,:]:
			x_center = i[0]
			y_center = i[1]
			radius = i[2]
			# checker is inside pip
			if x_center > tl_pip[0] and x_center < br_pip[0]:
				if y_center > tl_pip[1]:
					bottom_list[n_pip] = bottom_list[n_pip] + 1

	#for each pip on the bottom right
	for n_pip in range(6,12):
		tl_pip = (n_pip*checker_diameter + 2*bar_width, height -  pip_length)
		br_pip = (tl_pip[0] + checker_diameter, height)
		# cv2.rectangle(warped, tl_pip, br_pip, color=(0,255,0), thickness=5)
		for i in circles[0,:]:
			x_center = i[0]
			y_center = i[1]
			radius = i[2]

			# # checker is over the bar
			# if x_center > half_width and x_center < half_width + 2*bar_width:
			# 	continue
			# checker is inside pip
			if x_center > tl_pip[0] and x_center < br_pip[0]:
				if y_center > tl_pip[1]:
					bottom_list[n_pip] = bottom_list[n_pip] + 1


	#for each pip on the top left
	for n_pip in range(0,6):
		tl_pip = (n_pip*checker_diameter, 0)
		br_pip = (tl_pip[0] + checker_diameter, pip_length)
		# cv2.rectangle(warped, tl_pip, br_pip, color=(0,0,255), thickness=5)
		for i in circles[0,:]:
			x_center = i[0]
			y_center = i[1]
			radius = i[2]
			# checker is inside pip
			if x_center > tl_pip[0] and x_center < br_pip[0]:
				if y_center > tl_pip[1] and y_center < br_pip[1]:
					top_list[n_pip] = top_list[n_pip] + 1

	#for each pip on the top right
	for n_pip in range(6,12):
		tl_pip = (n_pip*checker_diameter + 2*bar_width, 0)
		br_pip = (tl_pip[0] + checker_diameter, pip_length)
		# cv2.rectangle(warped, tl_pip, br_pip, color=(255,255,255), thickness=5)
		for i in circles[0,:]:
			x_center = i[0]
			y_center = i[1]
			radius = i[2]

			# # checker is over the bar
			# if x_center > half_width and x_center < half_width + 2*bar_width:
			# 	continue
			# checker is inside pip
			if x_center > tl_pip[0] and x_center < br_pip[0]:
				if y_center > tl_pip[1] and y_center < br_pip[1]:
					top_list[n_pip] = top_list[n_pip] + 1

	# cv2.rectangle(warped, tl, br, color=(0,255,0), thickness=5)
	# cv2.imshow('detected circles', warped)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# save image results
	out_img_name = os.path.join(output_dir, os.path.basename(img_path).split(".jpg")[0] + ".visual_feedback.jpg")
	cv2.imwrite(out_img_name, warped)
	out_json_name = os.path.join(output_dir, os.path.basename(img_path).split(".jpg")[0] + ".checkers.json")
	answer = {"top": top_list, "bottom": bottom_list}
	with open(out_json_name, mode='w') as json_file:
		json.dump(answer, json_file)
		




