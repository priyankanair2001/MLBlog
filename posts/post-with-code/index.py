# ---
# title: Post With Code
# author: Martin Laptev
# date: now
# cache: true
# categories:
#   - news
#   - code
#   - analysis
#   - data visualization
# image: image.jpg
# jupyter:
#   jupytext:
#     formats: qmd:quarto,py:percent,md,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
#| echo: false
#| output: false
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# The plot below is from the [Seaborn Python library documentation](https://seaborn.pydata.org/examples/horizontal_boxplot.html):

# %% [markdown]
# 1. 5 blog posts, not 3
# 2. Topics are pre-defined:
#     1. Probability theory and random variables
#       - Histogram
#     2. Clustering
#       - DBSCAN labels for scatter plot
#     3. Linear and nonlinear regression
#       - line on scatter plot
#     4. Classification
#       - ROC, PR, Confusion Matrix
#     5. Anomaly/outlier detection
#       - DBSCAN labels for scatter plot

# %% [markdown]
# Learning objectives:
# 1. Use various techniques related to preprocessing prior to the use of machine learning models.
# 2. Describe the probability theory and random variables.
# 3. Identify the common tasks in machine learning/data mining models for clustering.
# 4. Analyze multiple linear and nonlinear regression.
# 5. Describe the algorithms, theories, and applications related to machine learning/data mining for classification.
# 6. Detect anomaly/outlier behavior and the treatment techniques.

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))
ax.set_xscale("log")

# Load the example planets dataset
planets = sns.load_dataset("planets")

# Plot the orbital period with horizontal boxes
sns.boxplot(
    planets, x="distance", y="method", hue="method",
    whis=[0, 100], width=.6, palette="vlag"
)

# Add in points to show each observation
sns.stripplot(planets, x="distance", y="method", size=4, color=".3")

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)

# %% [markdown]
# The plot below is from the [Yellowbrick Python library documentation](https://www.scikit-yb.org/en/latest/api/features/jointplot.html?highlight=joint%20plot#joint-plot-visualization):

# %%
from yellowbrick.datasets import load_concrete
from yellowbrick.features import JointPlotVisualizer

# Load the dataset
X, y = load_concrete()

# Instantiate the visualizer
visualizer = JointPlotVisualizer(columns="cement")

visualizer.fit_transform(X, y)        # Fit and transform the data
visualizer.show()                     # Finalize and render the figure

# %% [markdown]
# The example below shows tuning of the regularization strength $\alpha$.
# ![image.png](attachment:3f6e174e-eca2-47a0-bc3b-605681c448ea.png)
# ![image.png](attachment:822e2379-4768-4f25-93c5-1ecfea0b84b7.png)

# %% [markdown]
# This is a change to the markdown text.

