import matplotlib.pyplot as plt
import seaborn as sns

def plot_envelopes(df=None, techs=None, name="mga",
    colors=sns.color_palette("Set1"), unit="kW", sharey=True, format="pdf",
    save=None, nicenames=None):
    """

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with investvalues as returned by MGA.get_invest_values()
    techs: list
        List of technologies to plot (index of dataframe `df`)
    name: str
        Name for plot to save
    colors : list
        List of colors, same order as `techs`
    """
    if nicenames is None:
        nicenames = dict(zip(techs, techs))
    sns.set_style("darkgrid")

    df = df.unstack()
    df.sort_index(inplace=True, axis=1)
    df.sort_index(inplace=True, axis=0)

    if techs is None:
        techs = [
            i for i in df.index.get_level_values(level=0).unique()
            if i != "base_solution"]

    fig, axs = plt.subplots(1, len(techs), sharey=sharey, figsize=(12, 4))
    ax = 0
    for tech in techs:
        kwargs = {
            "marker": ".",
            "markersize": 10,
            "color": colors[ax],
        }
        # add base solution to plot
        axs[ax].plot(
            df.loc[("base_solution", "base_solution")].index.astype("float"),
            df.loc[("base_solution", "base_solution")][tech][1],
            **kwargs)

        axs[ax].fill_between(
            df.loc[("base_solution", "base_solution")].index.astype("float"),
            df.loc[("base_solution", "base_solution")][tech][1],
            df.loc[("base_solution", "base_solution")][tech][-1],
            alpha=0.5,
            color=colors[ax],
        )

        axs[ax].plot(
            df.loc[tech].index.astype("float"),
            df.loc[tech][tech][1],
            **kwargs)
        axs[ax].plot(
            df.loc[tech].index.astype("float"),
            df.loc[tech][tech][-1],
            **kwargs)

        axs[ax].fill_between(
            df.loc[tech].index.astype("float"),
            df.loc[tech][tech][1],
            df.loc[tech][tech][-1],
            alpha=0.5,
            color=colors[ax],
        )
        axs[ax].set_title(nicenames[tech])
        axs[ax].set_xlabel(r"$\epsilon$")
        ax += 1
        axs[0].set_ylabel("Installed capacity in {}".format(unit))


    if save:
        plt.savefig("envelope_{0}.{1}".format(name, format), bbox_inches="tight")


def plot_capacities(df, techs):
    """
    """
    eps = [i for i in df.index.get_level_values(2).unique()][:-1]
    epsmap = dict(zip(range(len(eps)), eps))

    fig, axs = plt.subplots(2, len(eps), figsize=(15, 5), sharex=True, sharey=True)

    kwargs = dict(kind="bar", stacked=True, legend=False, width=0.9, cmap="Set2")

    for idx, eps in epsmap.items():
        tech_df = df.xs(eps, level=2, axis=0)[techs].unstack()
        tech_df.xs(1, level=2, axis=1).loc[techs].plot(ax=axs[1, idx], **kwargs)
        tech_df.xs(-1, level=2, axis=1).loc[techs].plot(ax=axs[0, idx],**kwargs)
        axs[0, idx].set_title(r"$\epsilon=$" + str(eps))

    axs[0, 0].set_ylabel("Max")
    axs[1, 0].set_ylabel("Min")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles[::-1],
        labels[::-1],
        ncol=4,
        loc="upper left",
        bbox_to_anchor=(0.2, 1.1),
        frameon=False,
    )
    axs[0, 0].add_artist(legend)
    plt.savefig("mga_capacities.pdf", bbox_inches="tight")
