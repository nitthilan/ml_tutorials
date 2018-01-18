X_WTH = 4
Y_HGT = 4
Z_LYR = 4
NUM_CORES = X_WTH*Y_HGT*Z_LYR
SW_MAT_DIM = NUM_CORES*2

K_AVG = 4.5 # Average number of links per router. Each router-router connection has two links
K_MAX = 7 # Max number of links per router

ALPHA = 2.4 # Alpha parameter for powerlaw contraint

# Per core distribution
DISTRIBUTION = [ 16,  5,   2,   1]
NUM_CORE_IN_LAYER = X_WTH*Y_HGT

if __name__=="__main__":
	print GET_MAX_DISTANCE()