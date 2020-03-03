import sys
import numpy as np

class DecisionTree:
  """Code author: Ting-Sheng Lin (tingshel@andrew.cmu.edu)"""
  def __init__(self):
    self.model = {}
    self.opt = ['', '']
    self.labels = ['', '']
    self.titles = []


  # display pretty splitted tree
  def __print_model(self):
    # print(self.model)
    print(f"[{self.model['proportion'][0]} {self.labels[0]}/{self.model['proportion'][1]} {self.labels[1]}]")
    if 'head' in self.model:
      self.__dfs_print(self.model['head'], 1)

  def __dfs_print(self, cur_node, depth):
    if 'end' in cur_node:
      return
    for i in reversed(range(2)):
      for d in range(depth):
        print('| ', end = '')
      print(f"{self.titles[cur_node['split_attr']]} = {self.opt[i]}: [{cur_node['proportion'][i][0]} {self.labels[0]}/{cur_node['proportion'][i][1]} {self.labels[1]}]")
      self.__dfs_print(cur_node[i], depth + 1)


  # calculate Gini Impurity for each attribute
  # slice data at the same time
  def __cal(self, data, attr_i):
    sub_data0, sub_data1 = [], []
    sub_results0, sub_results1 = [], [] 
    data_len = len(data)
    for i in range(data_len):
      if data[i][attr_i] == self.opt[0]:
        sub_data0.append(data[i])
        sub_results0.append(data[i][-1])
      elif data[i][attr_i] == self.opt[1]:
        sub_data1.append(data[i])
        sub_results1.append(data[i][-1])
    count00, count01 = self.__inspect(sub_results0)
    count10, count11 = self.__inspect(sub_results1)
    v0 = 0 if count00 > count01 else 1 # result index
    v1 = 0 if count10 > count11 else 1
    if count00 == 0 and count01 == 0:
      gi0 = 0
    else:
      gi0 = (count00 / (count00 + count01)) * (count01 / (count00 + count01))
    if count10 == 0 and count11 == 0:
      gi1 = 0
    else:
      gi1 = (count10 / (count10 + count11)) * (count11 / (count10 + count11))
    gi = (gi0 * len(sub_data0)) / data_len + (gi1 * len(sub_data1)) / data_len
    return [[gi, gi0, gi1], [sub_data0, sub_data1], [count00, count01, count10, count11], [v0, v1]]

  def __inspect(self, results): 
    if not results:
      return [0, 0]
    count_l0, count_l1 = 0, 0
    for i in range(len(results)):
      if results[i] == self.labels[0]:
        count_l0 += 1
      elif results[i] == self.labels[1]:
        count_l1 += 1

    return [count_l0, count_l1]


  def train(self, data, depth, titles):
    # set up corresponding data options and results 
    opts = [['A', 'notA'], ['n', 'y'], ['0', '1']] # options for different data
    labels = [['A', 'notA'], ['democrat', 'republican'], ['0', '1']] # results for different data
    for i in range(len(opts)):
      if data[1][0] in opts[i]:
        self.opt = opts[i] # binary option
        self.labels = labels[i] # binary labels
        break
    self.titles = titles
    count_l0, count_l1 = self.__inspect([d[-1] for d in data])
    gi = (count_l0 / (count_l0 + count_l1)) * (count_l1 / (count_l0 + count_l1))
    self.model['proportion'] = [count_l0, count_l1]
    if depth == 0:
      self.model['end'] = self.labels[0] if count_l0 > count_l1 else self.labels[1]
      self.__print_model()
      return
    self.model['head'] = {}
    self.__recursive_train(data, depth, [], gi, self.model['head'])
    self.__print_model()

  def __recursive_train(self, data, depth, used_i, pre_gi, cur_node):
    min_gi, gi0, gi1 = 1, 1, 1
    selected_i, vote, sub_data = -1, [], []
    count00, count01, count10, count11 = 0, 0, 0, 0
    for i in range(len(data[0]) - 1):
      if i in used_i:
        continue
      gi, sd, c, v = self.__cal(data, i)
      if gi[0] < min_gi:
        min_gi, gi0, gi1 = gi
        selected_i = i
        sub_data = sd
        count00, count01, count10, count11 = c
        vote = v

    used_i = used_i + [selected_i]
    cur_node['split_attr'] = selected_i
    # cur_node['gi'] = [min_gi, gi0, gi1]
    cur_node['proportion'] = [[count00, count01], [count10, count11]]

    if min_gi == pre_gi:
      cur_node['end'] = self.labels[0] if (count00 + count10) > (count01 + count11) else self.labels[1]
      return

    # for divided by all attr
    # OR for the bottom of the tree
    if len(data[0]) - 1 == len(used_i) or depth <= 1:
      cur_node[0] = {'end': self.labels[vote[0]]}
      cur_node[1] = {'end': self.labels[vote[1]]}
      return

    cur_node[0] = {}
    cur_node[1] = {} 
    self.__recursive_train(sub_data[0], depth - 1, used_i, gi0, cur_node[0])
    self.__recursive_train(sub_data[1], depth - 1, used_i, gi1, cur_node[1])   
    

  def test(self, data):
    predictions = []
    error_count = 0
    for row in data:
      cur_node = self.model
      if 'end' not in cur_node:
        cur_node = self.model['head']      
      while 'end' not in cur_node:
        split_val = row[cur_node['split_attr']]
        if split_val == self.opt[0]:
          cur_node = cur_node[0]
        elif split_val == self.opt[1]:
          cur_node = cur_node[1]
      predictions.append(cur_node['end'])
      if row[-1] != cur_node['end']:
        error_count += 1
    return [predictions, error_count / len(data)]


if __name__ == '__main__':
  train_in_file = sys.argv[1]
  test_in_file = sys.argv[2]
  max_depth = int(sys.argv[3])
  train_out_file = sys.argv[4]
  test_out_file = sys.argv[5]
  metrics_file = sys.argv[6]

  # train & test train_data
  data_rows = np.genfromtxt(train_in_file, delimiter='\t', dtype='str')
  
  dt = DecisionTree()
  titles, data_rows = data_rows[0], data_rows[1:]
  # titles, data_rows = data_rows[:1][0], data_rows[1:]
  dt.train(data_rows, max_depth, titles)
  predictions, error_rate_train = dt.test(data_rows)
  
  f = open(train_out_file,'w+')
  pl = len(predictions)
  for i in range(pl):
    f.write(f'{predictions[i]}')
    if i < pl - 1:
      f.write('\n')

  # train & test test_data
  data_rows = np.genfromtxt(test_in_file, delimiter='\t', dtype='str', skip_header=1)
  
  # titles, data_rows = data_rows[:1][0], data_rows[1:]
  # dt.train(data_rows, max_depth, titles)
  predictions, error_rate_test = dt.test(data_rows)

  f = open(test_out_file,"w+")
  pl = len(predictions)
  for i in range(pl):
    f.write(f'{predictions[i]}')
    if i < pl - 1:
      f.write('\n')
  
  # generate metrics
  f = open(metrics_file,"w+")
  f.write(f"error(train): {error_rate_train}\n")
  f.write(f"error(test): {error_rate_test}")
