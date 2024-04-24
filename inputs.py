import numpy as np

# Constants
WEDGENUM = 3  # Number of interfaces
TIMELIM = 10  # Amount of time in seconds
INC = 100     # Number of time-steps

# Hardcoded inputs					# Description										# Possible Range		# Units

N = [1, 1, 1]                  		# Rotations per second for each wedges				# range: (0,+inf) 		# rev/sec
STARTTHETAX = 30             		# Initial laser angle with respect to x-axis		# range: [0,90) 		# degrees
STARTTHETAY = 45              		# Initial laser angle with respect to y-axis		# range: [0,90) 		# degrees
STARTPHIX = [15, 20, 25]			# Angles from wedges' x-axis projection & z-axis 	# range: (-90,90) 		# degrees
STARTPHIY = [0, 0, 0]           	# Angles from wedges' y-axis projection & z-axis 	# range: [-90,90] 		# degrees
RX = 0                   			# Initial laser height with respect on x-axis		# range: (-inf,+inf)	# unit length
RY = 0    							# Initial laser height with respect on y-axis		# range: (-inf,+inf)	# unit length

D = 5                       		# Wedge diameter									# range: (0,+inf)		# unit length
int_dist = [2, 3, 4]				# Optical axis distance between interfaces 			# range: (0*,+inf)		# unit length
int_dist.append(5)        			# Distance from last wedge to workpiece 			# range: (0*,+inf)		# unit length
ref_ind = [1.1, 1.2, 1.3, 1.4] 		# Refractive index before each interface 			# range: [1,+inf)		# unitless


# * Because in a realstic scenario you dont want the wedge intersecting with the work piece, 
# this value should technically be greater than (D/2)*cos(phi_last). Therefore any succesive 
# value in the list that's greater than D/2 will be definitively be fine. Otherwise, check.

# Note, just because something is possible doesn't mean you should do it™