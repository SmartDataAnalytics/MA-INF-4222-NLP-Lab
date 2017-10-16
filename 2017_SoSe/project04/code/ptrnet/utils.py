import math
import random
import itertools

def generate_data(input_length):
	radius = 1
	rangeX = (0, 10)
	rangeY = (0, 10)
	qty = input_length  # how many points you want

	# Generate a set of all points within 200 of the origin, to be used as offsets later
	deltas = set()
	for x in range(-radius, radius+1):
		for y in range(-radius, radius+1):
			if x*x + y*y <= radius*radius:
				deltas.add((x,y))

	randPoints = []
	excluded = set()
	i = 0
	while i<qty:
		x = random.randrange(*rangeX)
		y = random.randrange(*rangeY)
		if (x,y) in excluded: continue
		randPoints.append((x,y))
		i += 1
		excluded.update((x+dx, y+dy) for (dx,dy) in deltas)
	randPoints
	return randPoints



class Tsp:

    def __init__(self, points):
        self.points = points
        self.all_distances = []
        self.answer = []
    
    def length(self,x,y):
        return (math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))

    def solve_tsp_dynamic(self):
        #calc all lengths
        self.all_distances = [[self.length(x,y) for y in self.points] for x in self.points]
        #initial value - just distance from 0 to every other point + keep the track of edges
        A = {(frozenset([0, idx+1]), idx+1): (dist, [0,idx+1]) for idx,dist in enumerate(self.all_distances[0][1:])}
        cnt = len(self.points)
        for m in range(2, cnt):
            B = {}
            for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
                for j in S - {0}:
                    B[(S, j)] = min( [(A[(S-{j},k)][0] + self.all_distances[k][j], A[(S-{j},k)][1] + [j]) for k in S if k != 0 and k!=j])  #this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
            A = B
        res = min([(A[d][0] + self.all_distances[0][d[1]], A[d][1]) for d in iter(A)])
        self.answer = res[1] 
        return res[1]