import sys
import numpy as np 
import random



class Preprocessing():
	def __init__(self, train_path, test_path):
		self.train_path = train_path
		self.test_path = test_path
		self.load_data(self.train_path, True)
		self.load_data(self.test_path, False)
	def load_data(self, path, is_train):
		data = np.loadtxt(path)
		if is_train:
			self.train_data = data
			self.attribute_count = data.shape[1] - 1
			unique = sorted(np.unique(data[: ,[-1]]))
			self.class_count = len(unique)
			self.mapping = {}
			self.idx_mapping = {}
			for i, x in enumerate(unique):
				self.mapping[i] = x
				self.idx_mapping[x] = i
		else:
			self.test_data = data

class DecisionForest():
	def __init__(self, number_of_trees, option, pruning_thr, data_factory):
		self.data_factory = data_factory
		self.number_of_trees = number_of_trees
		self.trees = [DecisionTree(i, option, pruning_thr, data_factory).DTL_TopLevel() for i in range(1,self.number_of_trees+1)]
	def classify(self, data):
		forest_distributions = []
		for x in self.trees:
			dist = x.predict(data)
			forest_distributions.append(dist)
		forest_distributions = np.array(forest_distributions)
		return self.data_factory.mapping[int(np.argmax(np.mean(forest_distributions, axis = 0)))]
	def run_predictions(self):
		count = 0
		curr = 0
		for x in self.data_factory.test_data:
			predicted = self.classify(x[:-1])
			actual = int(x[-1])
			accuracy = 0 if predicted != actual else 1
			curr += accuracy
			count += 1
		print(curr/count)

class Tree:
	def __init__(self, id, tree_gain = None, tree_attribute = None, tree_dist = None, tree_thresh = None, is_leaf = False):
		self.id = id
		self.tree_gain = tree_gain
		self.tree_attribute = tree_attribute
		self.tree_dist = tree_dist
		self.tree_thresh = tree_thresh
		self.is_leaf = is_leaf
		self.left = None
		self.right = None
	def predict(self, data):
		if self.is_leaf:
			return self.tree_dist
		else:
			if data[self.tree_attribute] < self.tree_thresh:
				return self.left.predict(data)
			else:
				return self.right.predict(data)
class DecisionTree:
	def __init__(self, id_, option, pruning_thr, data_factory):
		self.data_factory = data_factory
		self.id = id_
		self.option = option
		self.pruning_thr = int(pruning_thr)
		self.idx_mapping = data_factory.idx_mapping
		self.mapping = data_factory.mapping
		self.attribute_count = data_factory.attribute_count
		self.class_count = data_factory.class_count
	def check_if_equal(self, examples):
		values = examples[: ,[-1]]
		element = examples[: ,[-1]][0]
		for x in values:
			if x != element:
				return False
		return True
	def DTL_TopLevel(self):
		default = self.distribution(self.data_factory.train_data)
		if self.option == "randomized":
			choose_attribute = self.choose_random
		elif self.option == "optimized":
			choose_attribute = self.choose_optimized
		return self.DTL(self.data_factory.train_data, default, choose_attribute)
	def DTL(self, examples, default, choose_attribute):
		if examples.shape[0] == 0 or examples.shape[0] < self.pruning_thr:
			return Tree(self.id,is_leaf = True, tree_dist = default)
		elif self.check_if_equal(examples):
			return Tree(self.id, is_leaf = True, tree_dist = default)
		else:
			best_gain, best_attribute, best_threshold = choose_attribute(examples)
			tree = Tree(self.id,tree_attribute = best_attribute, tree_thresh = best_threshold, tree_gain = best_gain)
			example_left = []
			example_right = []
			for x in examples:
				idx = best_attribute
				if x[idx] < best_threshold:
					example_left.append(x)
				elif x[idx] >= best_threshold:
					example_right.append(x)
			tree.left = self.DTL(np.array(example_left), self.distribution(examples), choose_attribute)
			tree.right = self.DTL(np.array(example_right), self.distribution(examples), choose_attribute)
			return tree
	def choose_optimized(self, examples):
		max_gain = max_attribute = max_threshold = -1
		for i in range(self.attribute_count):
			values = examples[: ,[i]]
			L = np.amin(values)
			M = np.amax(values)
			step = (M-L) / 51
			for k in range(1, 51):
				threshold = L + k * step
				gain = self.information_gain(examples, i, threshold)
				if gain > max_gain:
					max_gain = gain
					max_attribute = i
					max_threshold = threshold
		return max_gain, max_attribute, max_threshold
	def choose_random(self, examples):
		max_gain = max_attribute = max_threshold = -1
		x = random.randint(0, self.attribute_count - 1)
		values = examples[: ,[x]]
		L = np.amin(values)
		M = np.amax(values)
		step = (M-L) / 51
		for k in range(1, 51):
			threshold = L + k * step
			gain = self.information_gain(examples, x, threshold)
			if gain > max_gain:
				max_gain = gain
				max_attribute = x
				max_threshold = threshold
		return max_gain, x, max_threshold
	def information_gain(self, examples, A, threshold):
		base_entropy = self.entropy(examples)
		N1 = []
		N2 = []
		for x in examples:
			if x[A] < threshold:
				N1.append(x)
			elif x[A] >= threshold:
				N2.append(x)
		N1 = np.array(N1)
		N2 = np.array(N2)
		base_entropy -= (N1.shape[0] / examples.shape[0]) * self.entropy(N1)
		base_entropy -= (N2.shape[0] / examples.shape[0]) * self.entropy(N2)
		return base_entropy
	def entropy(self, examples):
		entropy = 0
		count = {}
		for x in examples:
			label = int(x[-1])
			count[label] = count.get(label, 1) + 1
		return sum([ -(part_size / examples.shape[0]) * np.log2(part_size / examples.shape[0]) for part_size in count.values()])
	def distribution(self, data):
		zeros = np.zeros(self.class_count)
		for x in data:
			label = int(x[-1])
			zeros[self.idx_mapping[label]] += 1
			zeros /= data.shape[0]
		return zeros
	def run_predictions(self, root):
		total = 0
		curr = 0
		for x in self.data_factory.test_data:
			actual = int(x[-1])
			dist = root.predict( x[:-1] )
			predicted = np.argmax(dist)
			predicted = self.mapping[int(predicted)]
			total += 1
			accuracy = 0 if actual != predicted else 1
			curr += accuracy
		print(curr/total)
def main():
	if len(sys.argv) < 5:
		print("Usage: [path_to_training_file] [path_to_test_file] option pruning_thr")
	data_factory = Preprocessing(*sys.argv[1:3])
	if sys.argv[3] == "forest3":
		forest = DecisionForest(3,"randomized",sys.argv[4], data_factory)
		forest.run_predictions()
	elif sys.argv[3] == "forest15":
		forest = DecisionForest(15,"randomized",sys.argv[4], data_factory)
		forest.run_predictions()
	else:
		tree = DecisionTree(1,*sys.argv[3:5], data_factory)
		root = tree.DTL_TopLevel()
		tree.run_predictions(root)
if __name__ == "__main__":
	main()
