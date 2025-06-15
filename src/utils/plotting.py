import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import json


def preprocess_results(result: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Preprocess training results for plotting."""
    if 'batch_loss' in result:
        result['BCE'] = (np.array(result['batch_loss']).reshape(60, -1).mean(axis=1), result.get('BCE', []))
        result.pop('batch_loss')
    return result


def plot_metrics(eval_metrics: List[float], train_metrics: Optional[List[float]] = None, 
                metric_name: str = "Loss", eval_epochs: Optional[List[int]] = None,
                ax: Optional[Axes] = None, figsize: Tuple[int, int] = (10, 6), 
                dpi: int = 100) -> Figure:
    """
    Plot training and evaluation metrics.

    Parameters:
    - eval_metrics: List of evaluation metric values
    - train_metrics: List of training metric values
    - metric_name: Name of the metric being plotted
    - eval_epochs: List of evaluation epochs
    - ax: Matplotlib Axes object to plot on
    - figsize: Figure size for new figure
    - dpi: Dots per inch for figure resolution
    
    Returns:
    - Figure: Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    
    if eval_epochs is None:
        eval_epochs = range(len(eval_metrics))
    
    ax.plot(eval_epochs, eval_metrics, label=f'Eval {metric_name}', marker='o')
    if train_metrics is not None:
        ax.plot(range(len(train_metrics)), train_metrics, label=f'Train {metric_name}', marker='s')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Train vs. Eval {metric_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_all_metrics(metrics_data: Dict[str, List], fig_title: str, 
                    eval_epochs: Optional[List[int]] = None,
                    cols: int = 2, figsize: Tuple[int] = (14, 7), 
                    dpi: int = 100) -> Figure:
    """
    Create a grid of subplots for multiple metrics.

    Parameters:
    - metrics_data: Dictionary with metric names as keys and values as data
    - fig_title: Global title of the figure
    - eval_epochs: List of evaluation epochs
    - cols: Number of columns in the subplot grid
    - figsize: Figure size for the entire grid
    - dpi: Resolution in dots per inch
    
    Returns:
    - Figure: Matplotlib figure object
    """
    n = len(metrics_data)
    rows = n // cols + (n % cols > 0)
    fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, constrained_layout=True)

    # Flatten axs for easy iteration
    if rows == 1 and cols == 1:
        axs = [axs]
    elif rows == 1 or cols == 1:
        axs = axs.flatten()
    else:
        axs = axs.flatten()

    for i, (metric_name, metrics) in enumerate(metrics_data.items()):
        if isinstance(metrics, tuple):
            train_metrics, eval_metrics = metrics
        else:
            train_metrics = None
            eval_metrics = metrics
        
        plot_metrics(eval_metrics, train_metrics, eval_epochs=eval_epochs, 
                    metric_name=metric_name, ax=axs[i])
        
    # Hide remaining subplots
    for i in range(len(metrics_data), len(axs)):
        fig.delaxes(axs[i])

    if fig_title:
        fig.suptitle(fig_title, fontsize=16)
    
    return fig


def plot_confusion_matrix(tp: int, fp: int, tn: int, fn: int, name: str,
                         ax: Optional[Axes] = None, figsize: Tuple[int, int] = (8, 6), 
                         dpi: int = 100) -> Figure:
    """
    Plot a confusion matrix.

    Parameters:
    - tp: Number of true positives
    - fp: Number of false positives
    - tn: Number of true negatives
    - fn: Number of false negatives
    - name: Name for the confusion matrix
    - ax: Matplotlib Axes object
    - figsize: Figure size
    - dpi: Dots per inch
    
    Returns:
    - Figure: Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
        
    conf_matrix = [[tp, fp], [fn, tn]]
    cax = ax.matshow(conf_matrix, cmap='Blues')
    ax.set_title(f'Confusion Matrix {name}')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ['Positive', 'Negative'])
    ax.set_yticklabels([''] + ['Positive', 'Negative'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    # Annotate the matrix with text
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', 
               color='white' if val > max(conf_matrix[0]) / 2 else 'black')
    
    return fig


def save_metrics_plot(metrics_data: Dict, save_path: str, title: str = "Training Metrics"):
    """
    Save metrics plot to file.
    
    Parameters:
    - metrics_data: Dictionary containing metrics
    - save_path: Path to save the plot
    - title: Title for the plot
    """
    metrics_data = preprocess_results(metrics_data)
    eval_epochs = range(1, 60, 2)  # Assuming evaluation every 2 epochs
    
    fig = plot_all_metrics(metrics_data, eval_epochs=eval_epochs, fig_title=title)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Metrics plot saved to {save_path}")


# Example usage function
def example_usage():
    """Example of how to use the plotting functions."""
    # Load example metrics
    try:
        with open('baseline_BCE_50.json', 'r') as file:
            metrics_data = json.load(file)
        
        metrics_data = preprocess_results(metrics_data)
        eval_epochs = range(1, 60, 2)
        
        # Plot all metrics
        fig = plot_all_metrics(metrics_data, eval_epochs=eval_epochs, 
                             fig_title="Training Evaluation Metrics")
        plt.show()
    except FileNotFoundError:
        print("Example metrics file not found. Skipping example.")


if __name__ == "__main__":
    example_usage()