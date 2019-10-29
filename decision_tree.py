import sys
import numpy as np
class DecisionTree:
    def __init__(self, training_path, test_path, option, pruning_thr):
        self.training_path = training_path
        self.test_path = test_path
        self.option = option
        self.pruning_thr = pruning_thr
        self.idx_mapping = {}
        self.test_data = None
        self.test_labels = None
        self.train_data = None
        self.train_labels = None
        self.load_classes(self.training_path, True)
        self.load_classes(self.test_path, False)
        self.load_mapping()
    def load_mapping(self):
        labels_set = set()
        for x in self.train_labels:
            labels_set.add(int(x))
        labels_set = sorted(labels_set)
        for i,x in enumerate(labels_set):
            self.idx_mapping[i] = x
    def load_classes(self,path, is_training):
        data = np.loadtxt(path)
        if is_training:
            self.train_labels = data[:, -1]
            self.train_data = data[:,[x for x in range(data.shape[1] - 1)]]
        else:
            self.test_labels = data[:, -1]
            self.test_data = data[: ,[x for x in range(data.shape[1] - 1)]]
    def DTL_TopLevel(self, is_training):
        attributes = [x for x in range(self.train_data.shape[1] - 1)]
        if is_training:
            default = self.distribution(self.train_labels)
        else:
            default = self.distribution(self.test_labels)
        if self.option == "optimized":
            choose_attribute = self.choose_attribute_optimized
        #elif self.option == "randomized":
        #    choose_attribute = self.choose_attribute_randomized
        #elif self.option == "forest3":
        #    choose_attribute = self.choose_attribute_forest3
        #else:
        #    choose_attribute = self.choose_attribute_forest15
        return self.DTL(self.train_data, attributes, default, choose_attribute)
    def DTL(self, examples, attributes, default, choose_attribute):
        classification = self.check_if_all_equal(examples)
        if len(examples) == 0:
            return default
        elif classification:
            return classification
        else:
            best_attribute, best_threshold = choose_attribute(examples, attributes)
            # tree root
            examples_left = []
            examples_right = []
            for x in examples:
                if x[best_attribute] < best_threshold:
                    examples_left.append(x)
                elif x[best_attribute] >= best_threshold:
                    examples_right.append(x)
            #tree.left = DTL(np.array(examples_left), attributes, self.distribution())
            #tree.right = DTL(np.array(examples_right), attributes, self.distribution())
            #return tree
    def choose_attribute_optimized(self, examples, attributes):
        max_gain = best_attribute = best_threshold = -1
        for i,attribute_values in enumerate(examples.T):
            L = np.min(attribute_values)
            M = np.max(attribute_values)
            for K in range(1, 51):
                threshold = L + K*(M-L)/51
                gain = self.information_gain(attribute_values, threshold)
                if gain > max_gain:
                    max_gain = gain
                    best_attribute = i
                    best_threshold = threshold
        return (best_attribute,best_threshold)
    def information_gain(self, attribute_values ,threshold):
        classA = 0
        classB = 0
        for x in attribute_values:
            if x < threshold:
                classA += 1
            if x >= threshold:
                classB += 1
        weight = -(classA) / (classA + classB)
        res = weight*np.log2(-weight)
        weight2 = -(classB) / (classA + classB)
        res -= weight2*np.log2(-weight2)
        return res

    def check_if_all_equal(self,examples):
        res = np.all(examples[:,1] == examples[:,1][0], axis =0)
        return res if res == False else [examples[:, 1][0]]
    def distribution(self, labels):
        hm = {}
        res = []
        count = 0
        for y in labels:
            hm[int(y)] = hm.get(int(y), 0) + 1
            count += 1
        for i,key in enumerate(sorted(hm)):
            res.append(hm[key] / count)
        return res
def main():
    if len(sys.argv) < 5:
        print("Usage: [path_to_training_file] [path_to_test_file] option pruning_thr")
    print(*sys.argv[1:5])
    tree = DecisionTree(*sys.argv[1:5])
    tree.DTL_TopLevel(True)



if __name__ == "__main__":
    main()
