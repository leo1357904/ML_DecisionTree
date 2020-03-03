import sys
import numpy as np


class Inspector:
  def __init__(self, res):
    res_opt = [['notA', 'A'], ['democrat', 'republican']] # results for different data
    for i in range(len(res_opt)):
      if res[0] in res_opt[i]:
        self.r0, self.r1 = res_opt[i] # binary result
        break
    

  def inspect(self, res):
    data_len = len(res)
    count0, count1 = 0, 0
    for i in range(data_len):
      if res[i] == self.r0:
        count0 += 1
      elif res[i] == self.r1:
        count1 += 1
    vote = self.r0 if count0 > count1 else self.r1
    error_count = 0
    for i in range(data_len):
      if res[i] != vote:
        error_count += 1
    
    error_rate = error_count / data_len
    gini_impurity = ((count0) * (count1) / (data_len * data_len)) + ((count1) * (count0) / (data_len *data_len))

    return [error_rate, gini_impurity]


if __name__ == '__main__':
  train_in_file = sys.argv[1]
  inspect_out_file = sys.argv[2]

  data_rows = np.genfromtxt(train_in_file, delimiter='\t', dtype='str', skip_header=1)
  res = data_rows[:, -1]
  ds = Inspector(res)
  error_rate_train, gini_impurity = ds.inspect(res)
  f = open(inspect_out_file,"w+")
  f.write(f"gini_impurity: {gini_impurity}\n")
  f.write(f"error: {error_rate_train}")
