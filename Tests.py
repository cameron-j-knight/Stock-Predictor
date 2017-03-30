import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({'A':[1,2,np.nan,4.5,6,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]})
df = df.join(df.interpolate('pchip'),rsuffix='n_')
df.plot()
print(df)
df['A'].map(lambda x: x + 1)
plt.show()