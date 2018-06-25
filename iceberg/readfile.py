
import numpy as np
import pandas as pd

train = pd.read_json("data/train.json")	# 1604 x 5
train = np.asarray(train)

