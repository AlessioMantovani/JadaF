import numpy.typing as typ
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

def set_plot_style(template: str = "plotly_dark"):
    """
    Set the template for Plotly plots.

    Parameters
    ----------
    template : str, optional
        The Plotly template to use, by default "plotly_dark".
        Options include: "plotly", "plotly_white", "plotly_dark", "ggplot2", 
        "seaborn", "simple_white", "plotly_white", etc.
    
    Returns
    -------
    None
        This function applies the template settings for all plots.
    """
    px.defaults.template = template

def line_plot(
    X: typ.ArrayLike,
    y: typ.ArrayLike,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Line Plot",
    template: str = "plotly_dark",
    return_trace: bool = False,
    # Enhanced customization options
    mode: str = "lines",  # "lines", "markers", "lines+markers", "lines+markers+text", etc.
    line_kwargs: Optional[Dict[str, Any]] = None,
    marker_kwargs: Optional[Dict[str, Any]] = None,
    text: Optional[typ.ArrayLike] = None,
    textposition: str = "top center",
    hover_text: Optional[typ.ArrayLike] = None,
    name: Optional[str] = None,
    opacity: float = 1.0,
    layout_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Plot a line graph using Plotly with extensive customization options.
    If saving trace for subplot don't use here title, use it in subplot function.

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
    template : str, optional
        Template for the plot, by default "plotly_dark".
    return_trace : bool, optional
        Whether to return the trace instead of showing the plot, by default False.
    mode : str, optional
        Drawing mode for plot, by default "lines".
        Options: "lines", "markers", "lines+markers", "lines+markers+text", etc.
    line_kwargs : dict, optional
        Dictionary of line properties. Examples:
        - color: line color
        - width: line width
        - dash: line style ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")
        - shape: interpolation between points ("linear", "spline", "vhv", "hvh", "vh", "hv")
    marker_kwargs : dict, optional
        Dictionary of marker properties. Examples:
        - symbol: marker symbol shape
        - size: marker size
        - color: marker color
        - line: dictionary with properties for marker border
    text : ArrayLike, optional
        Text to display with each data point.
    textposition : str, optional
        Position of text relative to markers, by default "top center".
    hover_text : ArrayLike, optional
        Custom hover text for each data point.
    name : str, optional
        Name to be used in the legend for this trace.
    opacity : float, optional
        Opacity of the trace, between 0 and 1.
    layout_kwargs : dict, optional
        Additional layout parameters to pass to update_layout.
    
    Returns
    -------
    go.Scatter or None
        The Scatter plot trace if return_trace=True; otherwise, shows the plot.
    """
    set_plot_style(template)
    X = np.asarray(X).flatten()
    y = np.asarray(y).flatten()
    # Initialize default dictionaries if None
    line_kwargs = line_kwargs or {}
    marker_kwargs = marker_kwargs or {}
    layout_kwargs = layout_kwargs or {}
    
    # Create the trace with customization options
    trace_args = {
        "x": X, 
        "y": y, 
        "mode": mode,
        "name": name or title,
        "opacity": opacity,
    }
    
    # Add line properties if in mode
    if "lines" in mode:
        trace_args["line"] = line_kwargs
    
    # Add marker properties if in mode
    if "markers" in mode:
        trace_args["marker"] = marker_kwargs
    
    # Add text if provided
    if text is not None:
        trace_args["text"] = text
        trace_args["textposition"] = textposition
    
    # Add hover text if provided
    if hover_text is not None:
        trace_args["hovertext"] = hover_text
    
    trace = go.Scatter(**trace_args)
    
    if return_trace:
        return trace
    else:
        fig = go.Figure(trace)
        
        # Apply standard layout settings
        layout_settings = {
            "title": title,
            "xaxis": dict(title=xlabel),
            "yaxis": dict(title=ylabel),
            "template": template,
        }
        
        # Update with any additional layout parameters
        layout_settings.update(layout_kwargs)
        
        fig.update_layout(**layout_settings)
        fig.show()

def scatter_plot(
    X: typ.ArrayLike,
    y: typ.ArrayLike,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Scatter Plot",
    template: str = "plotly_dark",
    return_trace: bool = False,
    # Enhanced scatter options
    marker_kwargs: Optional[Dict[str, Any]] = None,
    color: Optional[typ.ArrayLike] = None,
    size: Optional[typ.ArrayLike] = None,
    text: Optional[typ.ArrayLike] = None,
    hover_text: Optional[typ.ArrayLike] = None,
    name: Optional[str] = None,
    opacity: float = 1.0,
    layout_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Create a scatter plot with extensive customization options.

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
        Title of the plot, by default "Scatter Plot".
    template : str, optional
        Template for the plot, by default "plotly_dark".
    return_trace : bool, optional
        Whether to return the trace instead of showing the plot, by default False.
    marker_kwargs : dict, optional
        Dictionary of marker properties.
    color : ArrayLike, optional
        Array of values to map to colors for each marker.
    size : ArrayLike, optional
        Array of values to map to marker sizes.
    text : ArrayLike, optional
        Text to display with each data point.
    hover_text : ArrayLike, optional
        Custom hover text for each data point.
    name : str, optional
        Name to be used in the legend for this trace.
    opacity : float, optional
        Opacity of the trace, between 0 and 1.
    layout_kwargs : dict, optional
        Additional layout parameters to pass to update_layout.
    
    Returns
    -------
    go.Scatter or None
        The Scatter plot trace if return_trace=True; otherwise, shows the plot.
    """
    marker_kwargs = marker_kwargs or {}

    # If color or size is specified, add them to marker_kwargs
    if color is not None:
        marker_kwargs["color"] = color
    if size is not None:
        marker_kwargs["size"] = size
    
    # Use line_plot with mode="markers" for scatter plot
    return line_plot(
        X=X,
        y=y,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        template=template,
        return_trace=return_trace,
        mode="markers",
        marker_kwargs=marker_kwargs,
        text=text,
        hover_text=hover_text,
        name=name,
        opacity=opacity,
        layout_kwargs=layout_kwargs
    )

def bar_plot(
    X: typ.ArrayLike,
    y: typ.ArrayLike,
    xlabel: str = "Categories",
    ylabel: str = "Values",
    title: str = "Bar Plot",
    template: str = "plotly_dark",
    return_trace: bool = False,
    # Bar customization options
    orientation: str = "v",  # 'v' for vertical, 'h' for horizontal
    bar_width: float = None,
    color: Union[str, List[str], typ.ArrayLike] = None,
    opacity: float = 1.0,
    text: Optional[typ.ArrayLike] = None,
    text_position: str = "auto",
    hover_text: Optional[typ.ArrayLike] = None,
    marker_kwargs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    layout_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[go.Bar | None]:
    """
    Create a bar plot with extensive customization options.

    Parameters
    ----------
    X : ArrayLike
        Categories or x-coordinates for the bars.
    y : ArrayLike
        Heights of the bars.
    xlabel : str, optional
        Label for the x-axis, by default "Categories".
    ylabel : str, optional
        Label for the y-axis, by default "Values".
    title : str, optional
        Title of the plot, by default "Bar Plot".
    template : str, optional
        Template for the plot, by default "plotly_dark".
    return_trace : bool, optional
        Whether to return the trace instead of showing the plot, by default False.
    orientation : str, optional
        'v' for vertical bars, 'h' for horizontal bars, by default 'v'.
    bar_width : float, optional
        Width of bars as a fraction of the total available width.
    color : str, list, or ArrayLike, optional
        Color or colors of the bars.
    opacity : float, optional
        Opacity of the bars, between 0 and 1.
    text : ArrayLike, optional
        Text to display on each bar.
    text_position : str, optional
        Position of text on bars, by default "auto".
    hover_text : ArrayLike, optional
        Custom hover text for each bar.
    marker_kwargs : dict, optional
        Dictionary of marker properties.
    name : str, optional
        Name to be used in the legend for this trace.
    layout_kwargs : dict, optional
        Additional layout parameters to pass to update_layout.
    
    Returns
    -------
    go.Bar or None
        The Bar plot trace if return_trace=True; otherwise, shows the plot.
    """
    set_plot_style(template)
    
    # Initialize default dictionaries if None
    marker_kwargs = marker_kwargs or {}
    layout_kwargs = layout_kwargs or {}
    
    # Set color if provided
    if color is not None:
        marker_kwargs["color"] = color
    
    trace_args = {
        "x": X if orientation == "v" else y,
        "y": y if orientation == "v" else X,
        "orientation": orientation,
        "opacity": opacity,
        "name": name or title,
        "marker": marker_kwargs
    }
    
    # Add width if provided
    if bar_width is not None:
        trace_args["width"] = bar_width
    
    # Add text if provided
    if text is not None:
        trace_args["text"] = text
        trace_args["textposition"] = text_position
    
    # Add hover text if provided
    if hover_text is not None:
        trace_args["hovertext"] = hover_text
    
    trace = go.Bar(**trace_args)
    
    if return_trace:
        return trace
    else:
        fig = go.Figure(trace)
        
        # Apply standard layout settings
        layout_settings = {
            "title": title,
            "xaxis": dict(title=ylabel if orientation == "h" else xlabel),
            "yaxis": dict(title=xlabel if orientation == "h" else ylabel),
            "template": template,
        }
        
        # Update with any additional layout parameters
        layout_settings.update(layout_kwargs)
        
        fig.update_layout(**layout_settings)
        fig.show()

def cmatrix_plot(
    y_true: typ.ArrayLike,
    y_pred: typ.ArrayLike,
    xlabel: str = "Predicted",
    ylabel: str = "True",
    title: str = "Confusion Matrix",
    template: str = "plotly_dark",
    color_scale: str = "Viridis",
    return_trace: bool = False,
    # Enhanced customization options
    labels: Optional[List[str]] = None,
    normalize: Optional[str] = None,  # None, 'true', 'pred', 'all'
    annotation_format: str = ".0f",
    annotation_kwargs: Optional[Dict[str, Any]] = None,
    colorbar_kwargs: Optional[Dict[str, Any]] = None,
    layout_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[go.Heatmap | None]:
    """
    Plot a confusion matrix using Plotly with extensive customization options.
    If saving trace for subplot don't use here title, use it in subplot function.

    Parameters
    ----------
    y_true : ArrayLike
        The true labels.
    y_pred : ArrayLike
        The predicted labels.
    xlabel : str, optional
        Label for the x-axis, by default "Predicted".
    ylabel : str, optional
        Label for the y-axis, by default "True".
    title : str, optional
        Title of the plot, by default "Confusion Matrix".
    template : str, optional
        Plotly template to use, by default "plotly_dark".
    color_scale : str, optional
        Color scale to apply to the heatmap, by default "Viridis".
    return_trace : bool, optional
        Whether to return the trace instead of showing the plot, by default False.
    labels : list, optional
        List of class labels to display in the matrix.
    normalize : str, optional
        Normalization method for the confusion matrix:
        - None: raw counts
        - 'true': normalize by row (true label)
        - 'pred': normalize by column (predicted label)
        - 'all': normalize by total number of samples
    annotation_format : str, optional
        Format string for the annotations, by default ".0f".
    annotation_kwargs : dict, optional
        Additional kwargs for annotations (font, etc.).
    colorbar_kwargs : dict, optional
        Dictionary of colorbar properties.
    layout_kwargs : dict, optional
        Additional layout parameters to pass to update_layout.
    
    Returns
    -------
    go.Heatmap or None
        The Heatmap trace for the confusion matrix plot if return_trace=True; otherwise, shows the plot.
    """
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Determine unique class labels if not provided
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    # Apply normalization if specified
    if normalize == 'true':
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_sum = cm_sum.astype('float') if cm_sum.sum() > 0 else cm_sum
        cm = cm.astype('float') / cm_sum
        colorbar_title = "Proportion"
    elif normalize == 'pred':
        cm_sum = cm.sum(axis=0, keepdims=True)
        cm_sum = cm_sum.astype('float') if cm_sum.sum() > 0 else cm_sum
        cm = cm.astype('float') / cm_sum
        colorbar_title = "Proportion"
    elif normalize == 'all':
        cm_sum = cm.sum()
        cm = cm.astype('float') / cm_sum if cm_sum > 0 else cm
        colorbar_title = "Proportion"
    else:
        colorbar_title = "Count"
    
    set_plot_style(template)
    
    # Initialize default dictionaries if None
    colorbar_kwargs = colorbar_kwargs or {}
    annotation_kwargs = annotation_kwargs or {}
    layout_kwargs = layout_kwargs or {}
    
    # Default colorbar title if not provided
    if "title" not in colorbar_kwargs:
        colorbar_kwargs["title"] = colorbar_title
    
    # Create heatmap trace
    heatmap_trace = go.Heatmap(
        z=cm,
        x=[str(label) for label in labels],
        y=[str(label) for label in labels],
        colorscale=color_scale,
        colorbar=colorbar_kwargs,
        zmin=0,
        zmax=cm.max()
    )

    if return_trace:
        return heatmap_trace
    else:
        fig = go.Figure(heatmap_trace)
        
        # Apply standard layout settings
        layout_settings = {
            "title": title,
            "xaxis": dict(title=xlabel),
            "yaxis": dict(title=ylabel),
            "template": template,
        }
        
        # Update with any additional layout parameters
        layout_settings.update(layout_kwargs)
        
        fig.update_layout(**layout_settings)

        # Add annotations
        annotations = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Format the value according to annotation_format
                text_value = f"{cm[i][j]:{annotation_format}}"
                
                annotation = dict(
                    x=str(labels[j]),
                    y=str(labels[i]),
                    text=text_value,
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > cm.max() / 2 else "black")
                )
                
                # Update with any additional annotation parameters
                annotation.update(annotation_kwargs)
                
                annotations.append(annotation)
                
        fig.update_layout(annotations=annotations)

        fig.show()

def histogram(
    data: typ.ArrayLike,
    xlabel: str = "Value",
    ylabel: str = "Count",
    title: str = "Histogram",
    template: str = "plotly_dark",
    return_trace: bool = False,
    # Histogram customization options
    nbins: Optional[int] = None,
    bin_size: Optional[float] = None,
    range_x: Optional[List[float]] = None,
    cumulative: bool = False,
    histnorm: Optional[str] = None,  # '', 'percent', 'probability', 'density', 'probability density'
    color: str = None,
    opacity: float = 1.0,
    marker_kwargs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    layout_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[go.Histogram, None]:
    """
    Create a histogram with extensive customization options.

    Parameters
    ----------
    data : ArrayLike
        Data to be plotted in the histogram.
    xlabel : str, optional
        Label for the x-axis, by default "Value".
    ylabel : str, optional
        Label for the y-axis, by default "Count".
    title : str, optional
        Title of the plot, by default "Histogram".
    template : str, optional
        Template for the plot, by default "plotly_dark".
    return_trace : bool, optional
        Whether to return the trace instead of showing the plot, by default False.
    nbins : int, optional
        Number of bins to use.
    bin_size : float, optional
        Size of each bin. Only one of nbins or bin_size should be specified.
    range_x : list, optional
        Range [min, max] of the histogram's x-axis.
    cumulative : bool, optional
        If True, display a cumulative histogram, by default False.
    histnorm : str, optional
        Normalization method for the histogram.
    color : str, optional
        Color of the histogram bars.
    opacity : float, optional
        Opacity of the bars, between 0 and 1.
    marker_kwargs : dict, optional
        Dictionary of marker properties.
    name : str, optional
        Name to be used in the legend for this trace.
    layout_kwargs : dict, optional
        Additional layout parameters to pass to update_layout.
    
    Returns
    -------
    go.Histogram or None
        The Histogram trace if return_trace=True; otherwise, shows the plot.
    """
    set_plot_style(template)
    
    # Initialize default dictionaries if None
    marker_kwargs = marker_kwargs or {}
    layout_kwargs = layout_kwargs or {}
    
    # Set color if provided
    if color is not None:
        marker_kwargs["color"] = color
    
    # Create the trace with customization options
    trace_args = {
        "x": data,
        "opacity": opacity,
        "name": name or title,
        "marker": marker_kwargs,
        "histnorm": histnorm,
        "cumulative_enabled": cumulative
    }
    
    # Add bin settings if provided
    if nbins is not None:
        trace_args["nbinsx"] = nbins
    if bin_size is not None:
        trace_args["xbins"] = {"size": bin_size}
    
    trace = go.Histogram(**trace_args)
    
    if return_trace:
        return trace
    else:
        fig = go.Figure(trace)
        
        # Apply standard layout settings
        layout_settings = {
            "title": title,
            "xaxis": dict(title=xlabel, range=range_x),
            "yaxis": dict(title=ylabel),
            "template": template,
        }
        
        # Update with any additional layout parameters
        layout_settings.update(layout_kwargs)
        
        fig.update_layout(**layout_settings)
        fig.show()

def subplot(
    traces: list,
    titles: list = None,
    rows: int = None,
    cols: int = 2,
    main_title: str = "Subplot Plot",
    template: str = "plotly_dark",
    # Enhanced customization options
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    subplot_titles_kwargs: Optional[Dict[str, Any]] = None,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    vertical_spacing: float = 0.1,
    horizontal_spacing: float = 0.1,
    layout_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Create subplots from a list of Plotly traces with extensive customization options.

    Parameters
    ----------
    traces : list
        List of Plotly trace objects (e.g., go.Scatter, go.Heatmap).
    titles : list, optional
        List of titles for each subplot.
    rows : int, optional
        Number of rows; if None, it is calculated based on number of traces.
    cols : int, optional
        Number of columns, default is 2.
    main_title : str, optional
        Title for the entire figure.
    template : str, optional
        Plotly template to apply.
    shared_xaxes : bool, optional
        Whether to share x-axes among subplots in the same column.
    shared_yaxes : bool, optional
        Whether to share y-axes among subplots in the same row.
    subplot_titles_kwargs : dict, optional
        Dictionary of formatting options for subplot titles.
    x_labels : list, optional
        List of x-axis labels for each subplot.
    y_labels : list, optional
        List of y-axis labels for each subplot.
    vertical_spacing : float, optional
        Vertical spacing between plots.
    horizontal_spacing : float, optional
        Horizontal spacing between plots.
    layout_kwargs : dict, optional
        Additional layout parameters to pass to update_layout.
    
    Returns
    -------
    None
        Displays the subplot.
    """
    set_plot_style(template)
    
    # Calculate numbers
    num_plots = len(traces)
    rows = rows or ((num_plots + cols - 1) // cols)
    titles = titles or [f"Plot {i+1}" for i in range(num_plots)]
    
    # Initialize default dictionaries if None
    subplot_titles_kwargs = subplot_titles_kwargs or {}
    layout_kwargs = layout_kwargs or {}
    
    # Create subplot configuration
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        specs=[[{"type": "xy"} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Add traces to the subplots
    for i, trace in enumerate(traces):
        row_idx = (i // cols) + 1
        col_idx = (i % cols) + 1
        fig.add_trace(trace, row=row_idx, col=col_idx)
        
        # Set axis labels if provided
        if x_labels is not None and i < len(x_labels):
            fig.update_xaxes(title_text=x_labels[i], row=row_idx, col=col_idx)
        
        if y_labels is not None and i < len(y_labels):
            fig.update_yaxes(title_text=y_labels[i], row=row_idx, col=col_idx)
    
    # Apply standard layout settings
    layout_settings = {
        "title_text": main_title,
        "template": template,
    }
    
    # Update with any additional layout parameters
    layout_settings.update(layout_kwargs)
    
    # Customize subplot titles if needed
    if subplot_titles_kwargs:
        for i, _ in enumerate(titles):
            title_key = f"annotations[{i}]"
            for key, value in subplot_titles_kwargs.items():
                fig.layout[title_key][key] = value
    
    fig.update_layout(**layout_settings)
    fig.show()

def heatmap(
    z: typ.ArrayLike,
    x: Optional[typ.ArrayLike] = None,
    y: Optional[typ.ArrayLike] = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Heatmap",
    template: str = "plotly_dark",
    return_trace: bool = False,
    color_scale: str = "Viridis",
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    colorbar_kwargs: Optional[Dict[str, Any]] = None,
    text: Optional[typ.ArrayLike] = None,
    annotation_format: Optional[str] = None,
    annotation_kwargs: Optional[Dict[str, Any]] = None,
    layout_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[go.Heatmap, None] :
    set_plot_style(template)
    """
    Create a heatmap with extensive customization options using Plotly.

    Parameters
    ----------
    z : array-like
        2D data array to be represented as the heatmap values.
    x : array-like, optional
        Values for the x-axis ticks. Defaults to None, which uses indices.
    y : array-like, optional
        Values for the y-axis ticks. Defaults to None, which uses indices.
    xlabel : str, optional
        Label for the x-axis. Default is "X".
    ylabel : str, optional
        Label for the y-axis. Default is "Y".
    title : str, optional
        Title of the heatmap. Default is "Heatmap".
    template : str, optional
        Plotly template to apply for styling. Default is "plotly_dark".
    return_trace : bool, optional
        If True, returns the Plotly Heatmap trace object instead of showing the figure.
        Default is False.
    color_scale : str, optional
        Name of the colorscale to use for the heatmap. Default is "Viridis".
    z_min : float, optional
        Minimum value for color scaling. If not provided, determined automatically.
    z_max : float, optional
        Maximum value for color scaling. If not provided, determined automatically.
    colorbar_kwargs : dict, optional
        Additional keyword arguments passed to the colorbar configuration.
    text : array-like, optional
        Array of text annotations for each cell in the heatmap.
    annotation_format : str, optional
        Format string for cell annotations (e.g., ".2f" for two decimals).
        If provided, overlays text annotations on heatmap cells.
    annotation_kwargs : dict, optional
        Additional keyword arguments for customizing annotations (e.g., font size, style).
    layout_kwargs : dict, optional
        Additional keyword arguments passed to the figure's layout update method.
    
Returns
    -------
    go.Heatmap or None
        Returns a Plotly Heatmap trace if `return_trace=True`, otherwise displays the heatmap figure and returns None.
    """

    colorbar_kwargs = colorbar_kwargs or {}
    annotation_kwargs = annotation_kwargs or {}
    layout_kwargs = layout_kwargs or {}
    
    trace_args = {
        "z": z,
        "colorscale": color_scale,
        "colorbar": colorbar_kwargs,
    }
    
    if x is not None:
        trace_args["x"] = x
    if y is not None:
        trace_args["y"] = y
    if z_min is not None:
        trace_args["zmin"] = z_min
    if z_max is not None:
        trace_args["zmax"] = z_max
    if text is not None:
        trace_args["text"] = text
    
    trace = go.Heatmap(**trace_args)
    
    if return_trace:
        return trace
    else:
        fig = go.Figure(trace)
        
        layout_settings = {
            "title": title,
            "xaxis": dict(title=xlabel),
            "yaxis": dict(title=ylabel),
            "template": template,
        }
        
        layout_settings.update(layout_kwargs)
        fig.update_layout(**layout_settings)
        
        if annotation_format is not None:
            annotations = []
            z_array = z
            default_annotation = {
                "showarrow": False,
                "font": {"color": "white"}
            }
            default_annotation.update(annotation_kwargs)
            
            for i in range(len(z_array)):
                for j in range(len(z_array[i])):
                    value = z_array[i][j]
                    text_value = f"{value:{annotation_format}}"
                    
                    if z_max is not None and z_min is not None:
                        midpoint = (z_max + z_min) / 2
                    else:
                        midpoint = (max(map(max, z_array)) + min(map(min, z_array))) / 2
                    
                    text_color = "white" if value > midpoint else "black"
                    
                    annotation = dict(
                        x=j if x is None else x[j],
                        y=i if y is None else y[i],
                        text=text_value,
                        font=dict(color=text_color),
                        **default_annotation
                    )
                    annotations.append(annotation)

            fig.update_layout(annotations=annotations)

        fig.show()

def pca_plot():
    ...