"""
Chart widget for real-time plotting.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from collections import deque

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    # Fallback: create placeholder
    print("PyQtGraph not available - charts disabled")


class ChartWidget(QWidget):
    """
    Widget for displaying real-time training charts.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.max_points = 1000
        self.data_x = deque(maxlen=self.max_points)
        self.data_best = deque(maxlen=self.max_points)
        self.data_avg = deque(maxlen=self.max_points)

        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)

        if PYQTGRAPH_AVAILABLE:
            # Create plot widget
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground('w')
            self.plot_widget.setLabel('left', 'Fitness / Reward')
            self.plot_widget.setLabel('bottom', 'Generation / Episode')
            self.plot_widget.addLegend()

            # Create plot lines
            self.best_line = self.plot_widget.plot(
                [], [], pen=pg.mkPen(color='b', width=2), name='Best'
            )
            self.avg_line = self.plot_widget.plot(
                [], [], pen=pg.mkPen(color='r', width=2), name='Average'
            )

            layout.addWidget(self.plot_widget)
        else:
            # Placeholder
            from PyQt5.QtWidgets import QLabel
            from PyQt5.QtCore import Qt
            placeholder = QLabel("PyQtGraph not installed\nCharts unavailable")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)

    def add_data_point(self, metrics):
        """
        Add a new data point to the chart.

        Args:
            metrics: Metrics dictionary
        """
        if not PYQTGRAPH_AVAILABLE:
            return

        # Determine x-axis value (generation or episode)
        if 'generation' in metrics:
            x = metrics['generation']
        elif 'episode' in metrics:
            x = metrics['episode']
        else:
            return

        # Determine y-axis values
        best = None
        avg = None

        if 'best_fitness' in metrics:
            best = metrics['best_fitness']
            avg = metrics.get('avg_fitness', best)
        elif 'reward' in metrics:
            best = metrics['reward']
            avg = metrics.get('avg_reward', best)

        if best is None:
            return

        # Add to data
        self.data_x.append(x)
        self.data_best.append(best)
        self.data_avg.append(avg)

        # Update plots
        self.best_line.setData(list(self.data_x), list(self.data_best))
        self.avg_line.setData(list(self.data_x), list(self.data_avg))

    def clear(self):
        """Clear all data."""
        self.data_x.clear()
        self.data_best.clear()
        self.data_avg.clear()

        if PYQTGRAPH_AVAILABLE:
            self.best_line.setData([], [])
            self.avg_line.setData([], [])
