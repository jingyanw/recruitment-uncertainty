# Run simulation
import numpy as np
import itertools
import time
import scipy
import scipy.stats
from  numpy.random import randint
from pandas import read_pickle, to_pickle
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg')


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern'

# dictionaries for managing plotting style
MARKERS_MAP = {
	'ours': 'o', #  'd', '*'
	'greedy_x': 's',
	'greedy_xp': '<',
	'opt': '*' #  'd', '*'
}
LINESTYLES_MAP = { # 'dashed', 'dashdot'
	'ours': 'solid',
	'greedy_x': (0, (1, 1)),
	'greedy_xp': 'dotted',
	'opt': 'dashed',
}
LABELS = {
	'ours': 'LowValueL1+',
	'greedy_x': 'xGreedy',
	'greedy_xp': 'xpGreedy',
	'opt': 'OPT'
}

COLORS = {
	'ours': 'C0',
	'greedy_x': 'C1',
	'greedy_xp': 'C2',
	'opt': 'C3'
}

# data and figure file destinations
PLOT_DIR = 'figures_simulation/'
DATA_DIR = 'figures_data/'

if not os.path.exists(PLOT_DIR):
	print('mkdir %s...' % PLOT_DIR)
	os.makedirs(PLOT_DIR)
if not os.path.exists(DATA_DIR):
	print('mkdir %s...' % DATA_DIR)
	os.makedirs(DATA_DIR)

# dictionaries for accelerating the calculation of the objective function
PROBS = {}
MEMOIZE = False


# function for generating data to evaluate performance of various algos for different correlations for a range of targets
def run_beta_vary_target(n=50, include_opt=False, data_name=None):
	global MEMOIZE
	MEMOIZE = True

	lda = 3
	thresh = 0 # no brute-force calls for small instances (up to thresh)
	thresh_opt = None
	targets = [1, 2, 3, 4, 5]
	T = len(targets)
	repeat = 3

	modes = ['pos', 'no', 'neg']
	M = len(modes)

	objs_l1 = np.zeros((T, M, repeat))
	objs_greedy_x = np.zeros((T, M, repeat))
	objs_greedy_xp = np.zeros((T, M, repeat))
	objs_opt = np.zeros((T, M, repeat))

	tic = time.time()
	# loop for each target threshold M
	for it in range(T):
		print('%d/%d' % (it+1, T))
		target = targets[it]

		# loop over repeats of the data
		for r in range(repeat):
			if r % 1 == 0:
				print('[%d/%d] %.1f sec' % (r+1, repeat, time.time()-tic))

			xs = np.random.uniform(size=n)

			# loop over types of distributions
			for m in range(M):
				mode = modes[m]
				if mode == 'neg':
					ps = np.random.beta(10 * (1-xs), 10 * xs)
				elif mode == 'pos':
					ps = np.random.beta(10 * xs, 10 * (1-xs))
				elif mode == 'no':
					ps = np.random.uniform(size=n)
				else:
					raise Exception('Unknown mode')

				pmin = 0.01
				ps = pmin + (1 - pmin) * ps

				(ps_solve, xs_solve, objs_l1[it, m, r]) = solve_small(ps, xs, lda, target, thresh=thresh)
				(ps_greedy_x, xs_greedy_x, objs_greedy_x[it, m, r]) = heuristic_greedy(ps, xs, lda, target, mode='x')
				(ps_greedy_xp, xs_greedy_xp, objs_greedy_xp[it, m, r]) = heuristic_greedy(ps, xs, lda, target, mode='xp')
				(ps_opt, xs_opt, objs_opt[it, m, r]) = solve_opt_simple(ps, xs, lda, target, max_size=None)

	# save data
	if data_name is None:
		data_name = f'n={n}_lda={lda}_targets={targets}_ID={randint(10 ** 3)}'
	save_dict = {'solve': objs_l1, 'greedy_x': objs_greedy_x, 'greedy_xp': objs_greedy_xp, 'opt': objs_opt,
		'M': M, 'targets': targets, 'repeat': repeat, 'modes': modes, 'lda': lda, 'n': n, 'include_opt': include_opt
	}
	save_data(data_name, save_dict)


# function for generating data to evaluate performance of various algos for negative correlation and various levels of regularization for a range of targets
def run_neg_corr_vary_lda(n=50, include_opt=False, data_name=None):
	global MEMOIZE
	MEMOIZE = True

	lambdas = [1.5, 5, 30]
	L = len(lambdas)
	thresh = 0 # do not brute-force over small instances
	thresh_opt = None
	targets = [1, 2, 3, 4, 5]
	T = len(targets)
	repeat = 3

	mode = 'neg'

	objs_l1 = np.zeros((T, L, repeat))
	objs_greedy_x = np.zeros((T, L, repeat))
	objs_greedy_xp = np.zeros((T, L, repeat))
	objs_opt = np.zeros((T, L, repeat))

	tic = time.time()
	# loop for each target threshold M
	for it in range(T):
		print('%d/%d' % (it+1, T))
		target = targets[it]

		# loop for collecting multiple data points
		for r in range(repeat):
			if r % 1 == 0:
				print('[%d/%d] %.1f sec' % (r+1, repeat, time.time()-tic))

			xs = np.random.uniform(size=n)
			ps = np.random.beta(10 * (1-xs), 10 * xs)
			pmin = 0.01
			ps = pmin + (1 - pmin) * ps

			# loop over values of the penalty regularizer lambda (lda)
			for l in range(L):
				lda = lambdas[l]
				(ps_solve, xs_solve, objs_l1[it, l, r]) = solve_small(ps, xs, lda, target, thresh=thresh)
				(ps_greedy_x, xs_greedy_x, objs_greedy_x[it, l, r]) = heuristic_greedy(ps, xs, lda, target, mode='x')
				(ps_greedy_xp, xs_greedy_xp, objs_greedy_xp[it, l, r]) = heuristic_greedy(ps, xs, lda, target, mode='xp')
				(ps_opt, xs_opt, objs_opt[it, l, r]) = solve_opt_simple(ps, xs, lda, target, max_size=None)

	# save code
	if data_name is None:
		data_name = f'lambda_n={n}_lda={lambdas}_targets={targets}_ID={randint(10 ** 3)}'
	save_dict = {'solve': objs_l1, 'greedy_x': objs_greedy_x, 'greedy_xp': objs_greedy_xp, 'opt': objs_opt,
		'L': L, 'targets': targets, 'repeat': repeat, 'mode': mode, 'lambdas': lambdas, 'n': n, 'include_opt': include_opt
	}
	save_data(data_name, save_dict)


def save_data(data_name, data_dict):
    # takes:
    #   save_file_name: of the form 'save_file_name'
	#	data_dict: dict
	to_pickle(data_dict, DATA_DIR + f'/{data_name}.pickle')


def load_data(data_name):
	# returns: data_dict
	return read_pickle(DATA_DIR + f'/{data_name}.pickle')


# Final plotting method for varying correlation data
def plot_beta_correlation_vary_target(data_dict, save_fig=True):
	# data_dict: of the form {'solve': objs_l1, 'greedy_x': objs_greedy_x, 'greedy_xp': objs_greedy_xp, etc}
	modes = data_dict['modes']
	M = data_dict['M']
	objs_l1 = data_dict['solve']
	objs_greedy_x = data_dict['greedy_x']
	objs_greedy_xp = data_dict['greedy_xp']
	targets = data_dict['targets']
	repeat = data_dict['repeat']
	lda = data_dict['lda']
	n = data_dict['n']
	include_opt = False
	if 'include_opt' in data_dict and data_dict['include_opt']:
		include_opt = True
		objs_opt = data_dict['opt']

	(fig, axes) = plt.subplots(1, M, figsize=(9,3), constrained_layout = True)
	fig.tight_layout()

	for m in range(M):
		ax = axes[m]

		ax.errorbar(targets, np.mean(objs_l1[:, m, :], axis=1), np.std(objs_l1[:, m, :], axis=1) / np.sqrt(repeat),
			label=LABELS['ours'],
			marker=MARKERS_MAP['ours'], linestyle=LINESTYLES_MAP['ours']
		)
		ax.errorbar(targets, np.mean(objs_greedy_x[:, m, :], axis=1), np.std(objs_greedy_x[:, m, :], axis=1) / np.sqrt(repeat),
			label=LABELS['greedy_x'],
			marker=MARKERS_MAP['greedy_x'], linestyle=LINESTYLES_MAP['greedy_x']
		)
		ax.errorbar(targets, np.mean(objs_greedy_xp[:, m, :], axis=1), np.std(objs_greedy_xp[:, m, :], axis=1) / np.sqrt(repeat),
			label=LABELS['greedy_xp'],
			marker=MARKERS_MAP['greedy_xp'], linestyle=LINESTYLES_MAP['greedy_xp']
		)
		if include_opt:
			ax.errorbar(targets, np.mean(objs_opt[:, m, :], axis=1), np.std(objs_opt[:, m, :], axis=1) / np.sqrt(repeat),
				label=LABELS['opt'],
				marker=MARKERS_MAP['opt'], linestyle=LINESTYLES_MAP['opt']
			)

		ax.set_xlabel('target')
		ax.set_ylabel('Objective')

		ax.tick_params(axis='x')
		ax.tick_params(axis='y')

		# ax.set_ylim([0, None])
		ax.set_title(modes[m])
		ax.legend(loc = 'upper left')

	if save_fig:
		plt.savefig(PLOT_DIR + 'beta_vary_target.pdf', format='pdf', bbox_inches="tight")
	else:
		plt.show()


# Final plotting method for varying regularizer data
def plot_lambdas_vary_target(data_dict, save_fig=True):
	# data_dict: = {'solve': objs_l1, 'greedy_x': objs_greedy_x, 'greedy_xp': objs_greedy_xp, 'L': L, 'targets': targets, 'repeat': repeat, 'mode': mode, 'lambdas': lambdas, 'n': n
	mode = data_dict['mode']
	L = data_dict['L']
	objs_l1 = data_dict['solve']
	objs_greedy_x = data_dict['greedy_x']
	objs_greedy_xp = data_dict['greedy_xp']
	targets = data_dict['targets']
	repeat = data_dict['repeat']
	lambdas = data_dict['lambdas']
	n = data_dict['n']
	include_opt = False
	if 'include_opt' in data_dict and data_dict['include_opt']:
		include_opt = True
		objs_opt = data_dict['opt']

	(fig, axes) = plt.subplots(1, L, figsize=(9,3), constrained_layout = True)
	fig.tight_layout()

	for l in range(L):
		ax = axes[l]
		ax.errorbar(targets, np.mean(objs_l1[:, l, :], axis=1), np.std(objs_l1[:, l, :], axis=1) / np.sqrt(repeat),
			label=LABELS['ours'],
			marker=MARKERS_MAP['ours'], linestyle=LINESTYLES_MAP['ours']
		)
		ax.errorbar(targets, np.mean(objs_greedy_x[:, l, :], axis=1), np.std(objs_greedy_x[:, l, :], axis=1) / np.sqrt(repeat),
			label=LABELS['greedy_x'],
			marker=MARKERS_MAP['greedy_x'], linestyle=LINESTYLES_MAP['greedy_x']
		)
		ax.errorbar(targets, np.mean(objs_greedy_xp[:, l, :], axis=1), np.std(objs_greedy_xp[:, l, :], axis=1) / np.sqrt(repeat),
			label=LABELS['greedy_xp'],
			marker=MARKERS_MAP['greedy_xp'], linestyle=LINESTYLES_MAP['greedy_xp']
		)
		if include_opt:
			ax.errorbar(targets, np.mean(objs_opt[:, l, :], axis=1), np.std(objs_opt[:, l, :], axis=1) / np.sqrt(repeat),
				label=LABELS['opt'],
				marker=MARKERS_MAP['opt'], linestyle=LINESTYLES_MAP['opt']
			)

		ax.set_xlabel('target')
		ax.set_ylabel('Objective')

		ax.tick_params(axis='x')
		ax.tick_params(axis='y')

		ax.set_title(f'lambda={lambdas[l]}')
		ax.legend(loc = 'upper left')

	if save_fig:
		plt.savefig(PLOT_DIR + 'lambda_vary_target.pdf', format='pdf', bbox_inches="tight")
	else:
		plt.show()

# round up to the nearest fact^k
# assume that fact > 1
def round_up(xs, fact=2):
	if fact == 2:
		exponents = np.log2(xs)
		exponents = np.ceil(np.log2(xs))
		return np.power(2, exponents), exponents
	else:
		exponents = np.ceil(np.log2(xs)/np.log2(fact))  # round up to nearest power of fact
		return np.power(fact, exponents), exponents

# TARGET: parameter M
def solve_l1(ps, xs, lda, target, pmin=None):
	###
	if pmin is None:
		pmin = np.min(ps)

	n = len(ps)
	select_low = (xs <= (1 - pmin / 4) * lda)
	select_high = (xs >= lda)
	select_med = np.logical_not(np.logical_or(select_low, select_high))

	(ps_low, xs_low) = solve_small(ps[select_low], xs[select_low], lda=lda, target=target, pmin=pmin)
	(ps_med, xs_med, _) = solve_med(ps[select_med], xs[select_med], lda=lda, target=target, pmin=pmin)
	(ps_high, xs_high) = (ps[select_high], xs[select_high])

	obj_low = obj_l1(ps_low, xs_low, lda, target)
	obj_med = obj_l1(ps_med, xs_med, lda, target)
	obj_high = obj_l1(ps_high, xs_high, lda, target)

	obj_max = np.max([obj_low, obj_med, obj_high])
	if obj_max == obj_low:
		(ps_sol, xs_sol) = (ps_low, xs_low)
	elif obj_max == obj_med:
		(ps_sol, xs_sol) = (ps_med, xs_med)
	elif obj_max == obj_high:
		(ps_sol, xs_sol) = (ps_high, xs_high)
	else:
		raise Exception('Numerical error')

	idxs = check_idxs(ps_sol, xs_sol, ps, xs)

	return ps_sol, xs_sol, obj_max


# compute objective when items of (PS, XS) are selected
def obj_l1(ps, xs, lda, target, loss='oneside-l1', memoize=None):
	# MODE: {twoside_l1, oneside_l1}
	n = len(ps)
	val = np.sum(ps * xs)

	probs = poisson_probs(ps, memoize=None) # (n+1)-vector of probabilities that |S_Z| = i for i = 0 to n
	# L1
	if loss == 'twoside-l1':
		var = np.abs(np.arange(n+1) - target) * probs
	elif loss == 'oneside-l1':
		var = (np.arange(n+1) - target) * probs
		var = var[np.arange(n+1) > target]
	# L2
	elif loss == 'twoside-l2':
		var = np.square(np.arange(n+1) - target) * probs
	elif loss == 'oneside-l2':
		var = np.square(np.arange(n+1) - target) * probs
		var = var[np.arange(n+1) > target]

	var = np.sum(var)
	return val - lda * var


# returns PROBS: length-(n+1) (probabilities for values 0 through n)
def poisson_probs(ps, memoize=None):
	# Compute the probabilities
	# PS: length-n
	if len(ps) == 0:
		return np.array([1])

	if memoize is None: # if no specification, use the global setting
		memoize = MEMOIZE
	if memoize:
		(ps_unique, counts) = np.unique(ps, return_counts=True)
		(ps_unique, counts) = tuple(ps_unique), tuple(counts)
		if (ps_unique, counts) in PROBS:
			return PROBS[(ps_unique, counts)]

	# raw compute
	n = len(ps)
	prev_probs = poisson_probs(ps[:-1])
	p = ps[-1]

	probs = np.zeros(n+1)
	probs[:-1] = prev_probs * (1-p)
	probs[1:] += prev_probs * p

	if memoize:
		PROBS[(ps_unique, counts)] = probs
		# print(PROBS)
		# input()
	return probs


def solve_small(ps, xs, lda, target, pmin=None, thresh=None):
	if pmin is not None:
		assert(np.all(xs <= (1 - pmin / 4) * lda))

	# low and high here refer to whether x_max is well below lda or not
	(ps_low, xs_low, obj_low) = solve_small_low(ps, xs, lda, target, thresh=thresh)  # (currently called with thresh=0)
	(ps_high, xs_high, obj_high) = solve_small_high(ps, xs, lda, target)

	# choose the larger one among the two
	if obj_low >= obj_high:
		return ps_low, xs_low, obj_low
	else:
		return ps_high, xs_high, obj_high


# helper function for 'solve_small'
def solve_small_high(ps, xs, lda, target):
	# rounding factor to be used
	fact = 1.5
	mean_ub_multiplier = 2

	# ps_round: rounded up probabilities; exponents: log_(fact) of ps_round
	ps_round, exponents = round_up(ps, fact=fact) # length-n
	xs_round = ps * xs / ps_round

	exp_range = np.arange(min(exponents), max(exponents) + 1)  # [-4, -3, -2, -1]
	L = len(exp_range) # number of buckets
	exp_counts = np.zeros(L, dtype=int)
	dct = {}  # key: exponent, value = np.array of floats (x_i rounded) in decreasing order
	dct_idxs = {}

	# re-organize to each 1/fact^i group
	for exp_idx in range(L):
		exp = exp_range[exp_idx]
		idxs = np.where(exponents == exp)[0] # item idx within the group
		order = np.argsort(xs_round[idxs])[::-1]
		idxs_ordered = idxs[order]
		dct_idxs[exp_idx] = idxs_ordered
		dct[exp_idx] = xs_round[idxs_ordered]

		assert(np.array_equal(np.sort(xs_round[exponents == exp])[::-1], # decreasing values in the group
								dct[exp_idx]))

		exp_counts[exp_idx] = len(dct[exp_idx])

	mean_ub = mean_ub_multiplier * target / fact ** (min(exponents) - 1)  # normalize mean bound to minimum bucket size
	choices = mean_vector_generator(list(exp_counts), mean_ub, fact)

	choice_max = None
	obj_max = -np.inf
	xs_max = None
	ps_max = None
	for choice in choices:
		choice = np.array(choice)

		# construct selection
		ps_select = np.array([])
		xs_select = np.array([])
		for exp_idx in range(len(exp_range)):
			ps_select = np.concatenate((ps_select, np.ones(choice[exp_idx]) * np.power(fact, exp_range[exp_idx])))
			xs_select = np.concatenate((xs_select, dct[exp_idx][:choice[exp_idx]]))

		obj = obj_l1(ps_select, xs_select, lda, target)
		if obj > obj_max:
			obj_max = obj
			choice_max = np.array(choice)
			xs_max = xs_select
			ps_max = ps_select

	# construct ps_opt from CHOICE_MAX, xs_opt with original (x, p)
	ps_opt = np.array([])
	xs_opt = np.array([])
	for exp_idx in range(len(exp_range)):
		count = choice_max[exp_idx]
		ps_opt = np.concatenate((ps_opt, ps[dct_idxs[exp_idx]][:count]))
		xs_opt = np.concatenate((xs_opt, xs[dct_idxs[exp_idx]][:count]))

	obj_max_orig = obj_l1(ps_opt, xs_opt, lda, target, memoize=False)
	return ps_opt, xs_opt, obj_max_orig


# helper function for 'solve_small'
def solve_small_low(ps, xs, lda, target, thresh=None):
	# thresh: parameter tau (largest size for brute-force over small sets)
	# Brute-force all solutions with size at most thresh
	n = len(ps)
	if thresh is None:
		thresh = 5 # tuning parameter -- tau
		thresh = int(np.floor(thresh))

	if thresh == 0:
		return [], [], 0 # not selecting anything

	return solve_opt(ps, xs, lda, target, max_size=thresh)


# implementation of MediumValueL1+ algorithm
def solve_med(ps, xs, lda, target, pmin):
	assert(np.all(xs <= lda))
	assert(np.all(xs >= 1 - pmin / 4))

	n = len(ps)

	us = ps * xs
	u_sum = np.sum(us)
	if n <= 36 / pmin / pmin:
		sols = [np.array(sol, dtype=int) for sol in powerset(np.arange(n))]

		obj_max = -np.inf
		ps_opt = None
		xs_opt = None

		for sol in sols:
			obj = obj_l1(ps[sol], xs[sol], lda, target)
			if obj > obj_max:
				obj_max = obj
				ps_opt = ps[sol]
				xs_opt = xs[sol]
		idxs = check_idxs(ps_opt, xs_opt, ps, xs)
		return ps_opt, xs_opt, idxs
	else:
		# use a random order
		perm = np.random.permutation(n)

		pos_select = None
		for pos in range(n): # POS is the index
			u_sum_select = us[perm[:(pos+1)]]
			if u_sum <= target:
				if (u_sum_select >= u_sum / 3) and (u_sum_select <= u_sum / 2):
					pos_select = pos
					break
			else:
				if (u_sum_select >= target) and (u_sum_select < target + 1):
					pos_select = pos
					break
		if pos_select is None:
			raise Exception('Error')

		select = perm[:(pos_select+1)]
		ps_select = ps[select]
		xs_select = xs[select]
		idxs = check_idxs(ps_select, xs_select, ps, xs)

		return ps_select, xs_selct, idxs


# Check that (PS_SELECT, XS_SELECT) is a valid subset of (PS, XS), and return the item idxs
def check_idxs(ps_select, xs_select, ps, xs):
	n = len(ps)
	idxs = np.full(n, False)

	for i in range(len(ps_select)):
		(p, x) = ps_select[i], xs_select[i]

		positions = np.logical_and(ps == p, xs == x)
		positions = np.logical_and(positions, np.logical_not(idxs)) # not selected yet
		assert(np.any(positions))
		# choose the first unselected item
		pos = np.where(positions)[0][0]
		idxs[pos] = True
	return idxs


def powerset(iterable):
	# powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
	s = list(iterable)
	return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


# returns: PS, XS, OBJ at optimum
def solve_opt(ps, xs, lda, target, max_size=None):
	labels = np.unique(ps)
	L = len(labels) # number of unique groups
	counts = np.zeros(L, dtype=int)
	dct = {}
	# re-organize to groups (each group has the same ps)
	for il in range(L):
		label = labels[il]
		dct[il] = np.sort(xs[ps == label])[::-1] # decreasing values in the group
		counts[il] = len(dct[il])

	if max_size is None: # no maximum size
		sizes = [np.arange(0, c +1) for c in counts]
		choices = itertools.product(*sizes)
	else:
		choices = vector_generator(counts, max_size) # guarantees np.sum(choice) <= max_size:

	choice_max = None
	obj_max = -np.inf
	ps_opt = None
	xs_opt = None
	for choice in choices:

		choice = np.array(choice)

		# construct selection
		ps_select = np.array([])
		xs_select = np.array([])
		for il in range(L):
			ps_select = np.concatenate((ps_select, np.ones(choice[il]) * labels[il]))
			xs_select = np.concatenate((xs_select, dct[il][:choice[il]]))

		obj = obj_l1(ps_select, xs_select, lda, target)
		if obj > obj_max:
			obj_max = obj
			choice_max = np.array(choice)

			ps_opt = ps_select
			xs_opt = xs_select

	return ps_opt, xs_opt, obj_max


# a simple implementation of brute force opt that checks all subsets
def solve_opt_simple(ps, xs, lda, target, max_size=None):
	n = len(ps)

	sols = [np.array(sol, dtype=int) for sol in powerset(np.arange(n))]

	obj_max = -np.inf
	ps_opt = None
	xs_opt = None

	for sol in sols:
		obj = obj_l1(ps[sol], xs[sol], lda, target)
		if obj > obj_max:
			obj_max = obj
			ps_opt = ps[sol]
			xs_opt = xs[sol]
	idxs = check_idxs(ps_opt, xs_opt, ps, xs)
	return ps_opt, xs_opt, obj_max


# returns: PS, XS, OBJ chosen by greedy
def heuristic_greedy(ps, xs, lda, target, mode='x', loss='oneside-l1'):
	# MODE: {x, p, xp, random}
	n = len(ps)

	if mode == 'x':
		# order = np.argsort(xs)[::-1] # decreasing order
		order = np.lexsort((ps, xs))[::-1] # sort by xs first
	elif mode == 'p':
		# order = np.argsort(ps)[::-1] # decreasing order
		order = np.lexsort((xs, ps))[::-1] # sort by ps first

	elif mode == 'xp':
		order = np.argsort(xs * ps)[::-1]
	elif mode == 'random':
		order = np.random.permutation(n)
	else:
		raise Exception('unknown greedy mode')

	ps_sort = ps[order]
	xs_sort = xs[order]

	obj_max = -np.inf
	ps_opt = None
	xs_opt = None
	for i in range(n+1): # select #0 through n
		ps_select = ps_sort[:i]
		xs_select = xs_sort[:i]

		obj = obj_l1(ps_select, xs_select, lda, target, loss=loss)
		if obj > obj_max:
			obj_max = obj
			ps_opt = ps_select
			xs_opt = xs_select

	return ps_opt, xs_opt, obj_max


# yields: lists of numbers (m_1, m_2, ...) such that m_i <= n_i and sum(m_i) <= tau
def vector_generator(caps, tau):
	# caps: [n_1, n_2, ...]
	# tau: int
	if tau == 0:
		yield [0]*len(caps)
	elif len(caps) == 1:
		last_cap = min(caps[0], tau)
		yield from ([i] for i in range(last_cap + 1))
	else:
		first_cap = min(caps[0], tau) # can't take more than tau in the first entry
		remaining_caps = caps[1:]
		for i in range(first_cap + 1):
			for remaining_vec in vector_generator(remaining_caps, tau - i):
				yield [i] + remaining_vec


# yields: lists of numbers (m_1, m_2, ...) such that m_i <= n_i and E((m_1, m_2, ... )) <= w
def mean_vector_generator(caps, w, fact):
	# caps: [n_1, n_2, ...] numbers of available items
	# w: float 'stopping  mean', normalized to minimum size
	# fact: float > 1 encoding the gap between rounding buckets
	# assume: items from each coordinate j (from left, j=0 start) has "p_i" >= fact ** j

	if w < 1:  # stopping mean has been exceeded
		yield [0 for _ in range(len(caps))]
	elif len(caps) == 1:
		current_cap = min(caps[0], int(w))  # int() rounds w down, so this is the number of current items that we can afford
		yield from ([i] for i in range(current_cap + 1))
	else:
		current_cap = min(caps[0], int(w))
		remaining_caps = caps[1:]
		for i in range(current_cap + 1):
			for remaining_vec in mean_vector_generator(remaining_caps, (w - i)/fact, fact):  # (w - i)/fact is the effective mean limit going forward, normalized to the next bucket's size
				yield [i] + remaining_vec


# No save data
def run_different_losses(n):
	repeat = 10
	run_different_losses_vary_target(n=n, repeat=repeat)


def run_different_losses_vary_target(n=50, repeat=3, save_fig=True):
	global MEMOIZE
	MEMOIZE = True

	lda = 3
	targets = [1, 2, 3, 4, 5]
	T = len(targets)

	modes = ['pos', 'no', 'neg']
	M = len(modes)

	losses = ['oneside-l1', 'twoside-l1', 'oneside-l2', 'twoside-l2']

	for loss in losses:
		## Compute
		objs_greedy_x = np.zeros((T, M, repeat))
		objs_greedy_xp = np.zeros((T, M, repeat))

		tic = time.time()
		for it in range(T):
			print('%d/%d' % (it+1, T))
			target = targets[it]

			for r in range(repeat):
				if r % 1 == 0:
					print('[%d/%d] %.1f sec' % (r+1, repeat, time.time()-tic))

				xs = np.random.uniform(size=n)

				for m in range(M):
					mode = modes[m]
					if mode == 'neg':
						ps = np.random.beta(10 * (1-xs), 10 * xs)
					elif mode == 'pos':
						ps = np.random.beta(10 * xs, 10 * (1-xs))
					elif mode == 'no':
						ps = np.random.uniform(size=n)
					else:
						raise Exception('Unknown mode')

					pmin = 0.01
					ps = pmin + (1 - pmin) * ps
					(ps_greedy_x, xs_greedy_x, objs_greedy_x[it, m, r]) = heuristic_greedy(ps, xs, lda, target, mode='x', loss=loss)
					(ps_greedy_xp, xs_greedy_xp, objs_greedy_xp[it, m, r]) = heuristic_greedy(ps, xs, lda, target, mode='xp', loss=loss)

		## Plot
		(fig, axes) = plt.subplots(1, M, figsize=(9,3), constrained_layout=True)
		fig.tight_layout()

		for m in range(M):
			ax = axes[m]
			ax.errorbar(targets, np.mean(objs_greedy_x[:, m, :], axis=1), np.std(objs_greedy_x[:, m, :], axis=1) / np.sqrt(repeat),
				label=LABELS['greedy_x'], color=COLORS['greedy_x'],
				marker=MARKERS_MAP['greedy_x'], linestyle=LINESTYLES_MAP['greedy_x']
			)
			ax.errorbar(targets, np.mean(objs_greedy_xp[:, m, :], axis=1), np.std(objs_greedy_xp[:, m, :], axis=1) / np.sqrt(repeat),
				label=LABELS['greedy_xp'], color=COLORS['greedy_xp'],
				marker=MARKERS_MAP['greedy_xp'], linestyle=LINESTYLES_MAP['greedy_xp']
			)

			ax.set_xlabel('Target')
			ax.set_ylabel('Objective')

			ax.tick_params(axis='x')
			ax.tick_params(axis='y')

			# ax.set_ylim([0, None])
			ax.set_title('%s (%s)' % (loss, modes[m]))
			ax.legend(loc='upper left')

		if save_fig:
			plt.savefig(PLOT_DIR + f'vary_corr_{loss}_n={n}.pdf', format='pdf', bbox_inches="tight")
		else:
			plt.show()


if __name__ == '__main__':
	np.random.seed(0)

	# Fig 1
	if True:
		correlation_data_name = 'test_correlations'
		run_beta_vary_target(data_name=correlation_data_name)
		plot_beta_correlation_vary_target(load_data(correlation_data_name))

	# Fig 2
	if True:
		regularization_data_name = 'test_regularizations'
		run_neg_corr_vary_lda(data_name=regularization_data_name)
		plot_lambdas_vary_target(load_data(regularization_data_name))

	# Compare algorithms to optimal
	# Figure 3:
	if True:
		correlation_data_name = 'test_correlations_with_opt'
		run_beta_vary_target(n=20, include_opt=True, data_name=correlation_data_name)
		plot_beta_correlation_vary_target(load_data(correlation_data_name))
	# Figure 4:
	if True:
		regularization_data_name = 'test_regularizations_with_opt'
		run_neg_corr_vary_lda(n=20, include_opt=True, data_name=regularization_data_name)
		plot_lambdas_vary_target(load_data(regularization_data_name))

	# Evaluate greedy heuristics for different losses (Figure 5 and Figure 6)
	if True:
		run_different_losses(50)
