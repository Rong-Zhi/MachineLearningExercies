import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

X = 5 * rng.rand(10000,1)
y = np.sin(X).ravel()

y[::5] += 3*(0.5-rng.rand(X.shape[0]/5))
xplot = np.linspace(0, 5, 100000)[:,None]

train_size = 100
kr = GridSearchCV(KernelRidge(kernel='rbf',gamma=0.1),
                  cv=5,param_grid={
        "alpha": [1e0, 0.1,1e-2,1e-3],
        "gamma": np.logspace(-2,2,5)
    })

kr.fit(X[:train_size],y[:train_size])

y_kr = kr.predict(xplot)
