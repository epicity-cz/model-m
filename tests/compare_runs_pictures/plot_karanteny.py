import click
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


parties = ['party4', 'party5', 'party6']
restrictions = ['1', '1a', '1b']
policies = ['noeva', 'strongeva', 'exp2A', 'exp2B', 'exp2C']


def process_df_folder(folder_name, quarantine_name):
    print(f"Reading {quarantine_name} files.")
    dfs = [pd.read_csv(fname) for fname in glob.iglob(os.path.join(folder_name, '*.csv'))]
    df = pd.concat(dfs)
    df['quarantine'] = quarantine_name
    return df


@click.command()
@click.argument('files_dir')
@click.argument('out_dir')
def run(files_dir, out_dir):
    for party in parties:
        for restriction in restrictions:
            for on_off in ["", "_off"]:
                print(f"\nProcessing {party} ({'on' if not on_off else 'off'}), {restriction}")
                folder_names = [(f"history_{party}{on_off}_{q}_{restriction}", q) for q in policies]
                dfs = [process_df_folder(os.path.join(files_dir, fn), q) for fn, q in folder_names]
                plot_df = pd.concat(dfs, axis=0)

                if restriction == '1':
                    upper_lim = 15000
                elif restriction == '1a':
                    upper_lim = 3000
                elif restriction == '1b':
                    upper_lim = 1000
                else:
                    upper_lim = None

                fig, axs = plt.subplots(ncols=2, figsize=(13, 7), sharey='all')

                def axis_settings(ax, title, upper_lim):
                    ax.set_title(title, fontsize=14)
                    #ax.set_ylim(ymin=0.0, ymax=upper_lim)
                    #ax.set_xlim(xmin=6, xmax=101)

                    ax.legend(loc='upper left')

                sns.lineplot(x='T', y='I_d', data=plot_df,
                             hue='quarantine', estimator=np.median, ci='sd', ax=axs[0])
                axis_settings(axs[0], "Infected - detected", upper_lim)

                sns.lineplot(x='T', y='all_infected', data=plot_df,
                             hue='quarantine', estimator=np.median, ci='sd', ax=axs[1])
                axis_settings(axs[1], "Infected - all cases", upper_lim)
                axs[0].set_ylabel("Number of cases", fontsize=12)

                axs[0].set_yscale('log')
                axs[1].set_yscale('log')

                fig.suptitle(f"{party} ({'on' if not on_off else 'off'}, {restriction})",
                             size=18)

                fig.tight_layout()
                fig.subplots_adjust(top=0.88)
                plt.savefig(os.path.join(out_dir, f"{party}{on_off}_{restriction}.png"))
                print()


if __name__ == "__main__":
    run()
