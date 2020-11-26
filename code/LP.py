from scipy.optimize import linprog

if __name__ == "__main__":
	c = [-2, -3]
	A_ub = [[1, 2], [4, 0], [0, 4]]
	b_ub = [[8], [16], [12]]
	bounds = [(0, None), (0, None)]
	print(linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds))