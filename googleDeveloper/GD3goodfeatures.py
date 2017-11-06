# GD3goodfeatures.py

# a good feature considers different values in a population

import numpy as np
import matplotlib.pyplot as plt

# using dogs as an example
greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked = True, color = ['r', 'b'])
plt.show()

# this histogram gives decent data at the end points but not much
# in the center

# say that eye color is used. Eye color is relatively the same across
# all breeds so that is considered a useless feature

# avoid redundant features to avoid double counting
# use easy to understand features as well
# e.g. use distance between cities rather than lat/longitude

# ideal features are
'''
1. Informative
2. Independent
3. simple
'''