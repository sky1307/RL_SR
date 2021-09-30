

data = []
data1 = []
data2 = []
with open('heuristic.txt')as f:
    lines = f.readlines()
    for line in lines:
        s =  line.split()
        data.append(float(s[0]))
        data1.append(float(s[1]))
        data2.append(float(s[2]))


data  = sorted(data)
data1 = sorted(data1)
data2 = sorted(data2)

import pandas as pd
import numpy as np
import plotly.express as px

ar = np.array([data, data1, data2])
df = pd.DataFrame(ar.T, columns=['No SR', 'RL SR', 'LS SR'])

fig = px.box(df, log_y= True, points=False )
fig.show()


