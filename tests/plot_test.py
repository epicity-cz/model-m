from plot_utils import plot_histories, plot_mutliple_policies_everything, plot_state_histogram, plot_history
import matplotlib.pyplot as plt
import sys


# import sys

# plot_history(sys.argv[1])
# exit()


BASEDIR = "."
CITY = "town0"

filename = sys.argv[1]
variants_list = sys.argv[2:]

variant_dict = {}

for variant in variants_list:
    variant_dict[variant] = [
        f"{BASEDIR}/history_{variant}_{i}.csv"
        for i in range(0, 1000)
    ]

#variant_dict["cr"] = [
#       f"../data/fit_me_crT.csv"
#]


#variant_dict["gold"] = [ "../data/litovel_plot.csv" ]

import datetime
title = datetime.datetime.now().strftime("%B %d, %Y  %H:%M:%S")

print("ploting")
plt.rcParams["figure.figsize"] = (20, 15)
plot_mutliple_policies_everything(variant_dict, group_days=None, 
                                  group_func="max",  
                                  max_days=200,
	                          maxy=100,	
                                  save_path=f"{filename}.png", 
                                  variant=2, title=title)

#plot_mutliple_policies(variant_dict, group_days=None,
#                       group_func="max", value="I_d", save_path=f"{filename}_Id.png")
