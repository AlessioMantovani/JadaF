import numpy.typing
import seaborn as sns
import matplotlib.pyplot as plt

def line_plot(
    X: numpy.typing.ArrayLike,
    y: numpy.typing.ArrayLike,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Line Plot"
):
    """
    Plot a line graph using seaborn with direct x and y arrays.

    Parameters
    ----------
    X : ArrayLike
        Array-like object representing x-axis values.
    y : ArrayLike
        Array-like object representing y-axis values.
    xlabel : str, optional
        Label for the x-axis, by default "X".
    ylabel : str, optional
        Label for the y-axis, by default "Y".
    title : str, optional
        Title of the plot, by default "Line Plot".

    Returns
    -------
    None
        This function displays the plot and returns nothing.
    """
    sns.lineplot(x=X, y=y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
