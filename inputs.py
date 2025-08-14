import numpy as np

# Constants
WEDGENUM = 3
TIMELIM = 10
INC = 100

# Hardcoded inputs					# Description										# Range			# Units

N = [1, 1, 1]
STARTTHETAX = 10             		# Initial laser angle with respect to x-axis		# [0,90) 		# degrees
STARTTHETAY = 5              		# Initial laser angle with respect to y-axis		# [0,90) 		# degrees
STARTPHIX = [15, 20, 15]
STARTPHIY = [0, 0, 0]
RX = 0                   			# Initial laser height with respect on x-axis		# (-inf,+inf)	# unit length
RY = 0    							# Initial laser height with respect on y-axis		# (-inf,+inf)	# unit length

DX = 10                       		# Wedge diameter in x direction						# (0,+inf)		# unit length
DY = 10								# Wedge diameter in y direction						# (0,+inf)		# unit length				
int_dist = [6, 6, 6, 6]
int_dist.append(6)        			# Distance from last wedge to workpiece 			# (0*,+inf)		# unit length
ref_ind = [1.1, 1.2, 1.3, 1.4]

printit = 'off'						# Prints optional data in console
plotit = 'off'

# * Because in a realstic scenario you dont want the wedge intersecting with the work piece, 
# this value should technically be greater than (D/2)*cos(phi_last). Therefore any succesive 
# value in the list that's greater than D/2 will be definitively be fine. Otherwise, check.

# Note, just because something is possible doesn't mean you should do it™