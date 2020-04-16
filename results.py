import numpy as np
from matplotlib import pyplot as plt
import pdb

model_results_file = 'out/reconstruction_align/results.npz'
icp_results_file = 'out/reconstruction_align/results_icp.npz'

model_results = np.load(model_results_file)
icp_results = np.load(icp_results_file)

pdb.set_trace()

# Bar plot of each value.
for k, title in zip(['times', 'chamfer_distances', 'gt_cloud_distances'], ['Runtime (s)', 'Chamfer Distance', 'GT Correspondence Distance']):
    model_mean = model_results[k].mean()
    model_std = model_results[k].std()

    icp_mean = icp_results[k].mean()
    icp_std = icp_results[k].std()

    fig, ax = plt.subplots()

    ax.set_title(title)
    rects = ax.bar([0,1], height=[model_mean, icp_mean], tick_label=['Model', 'ICP'], width=0.5)

    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.4f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()

    
