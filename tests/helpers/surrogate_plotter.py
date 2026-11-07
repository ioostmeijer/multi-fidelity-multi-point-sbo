import os

import matplotlib.pyplot as plt


def plot_1d(
    lines: list[dict],
    x_label: str | None = "X",
    y_label: str | None = "Y",
    title: str | None = None,
    output_path: str | None = None,
) -> None:
    """Help function to plot 1D surrogates.

    Parameters
    ----------
    lines
        Input lines to plot.
    x_label
        Label of the x-axis.
    y_label
        Label of the y-axis.
    title
        Title of the plot.
    output_path
        Output path of the plot.
    """
    if os.getenv("CI"):
        return

    plt.figure()

    # Plot the lines
    for line in lines:
        x = line.pop("x")
        y = line.pop("y")
        plt.plot(x, y, **line)

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    if title:
        plt.title(title)

    # If legend not empty include
    if any("label" in line for line in lines):
        plt.legend()
    plt.grid()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_2d(
    lines: list[dict],
    title: str | None = None,
    output_path: str | None = None,
) -> None:
    """Help function to plot 2D surrogates.

    Parameters
    ----------
    lines
        Input lines to plot.
    title
        Title of the plot.
    output_path
        Output path of the plot.
    """
    if os.getenv("CI"):
        return

    fig, axs = plt.subplots(len(lines) // 2, 2)

    for ax, line in zip(axs.flatten(), lines):
        _create_subplot(ax, fig, line)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def _create_subplot(
    ax: plt.Axes,
    fig: plt.Figure,
    line: dict,
) -> None:
    """Create a subplot for 2D surrogate plotting.

    Parameters
    ----------
    ax
        Axes object
    fig
        Figure object
    line
        the lines to plot
    """
    x = line.pop("x")
    y = line.pop("y")
    title = line.pop("title", "")
    x_label = line.pop("x_label", "X1")
    y_label = line.pop("y_label", "X2")

    if "cmap" not in line:
        line["cmap"] = "viridis"
    if "levels" not in line:
        line["levels"] = 20

    contour = ax.contourf(
        x[:, 0].reshape(100, 10), x[:, 1].reshape(100, 10), y.reshape(100, 10), **line
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(contour, ax=ax)
    return
