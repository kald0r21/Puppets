"""
Main window for Puppets GUI application.
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QSplitter, QMenuBar, QMenu, QAction, QStatusBar,
                              QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QKeySequence
import os

from core.config.config_manager import ConfigManager
from core.trainers.ga_trainer import GATrainer
from core.trainers.cnn_trainer import CNNTrainer
from core.trainers.dqn_trainer import DQNTrainer
from core.controller.simulation_controller import SimulationController

# Import widgets (we'll create these)
from gui.widgets.control_panel import ControlPanel
from gui.widgets.pygame_widget import PygameWidget
from gui.widgets.stats_panel import StatsPanel
from gui.widgets.chart_widget import ChartWidget


class MainWindow(QMainWindow):
    """
    Main application window.
    Contains all widgets and manages the simulation.
    """

    def __init__(self):
        super().__init__()

        self.config_manager = ConfigManager()
        self.current_config = None
        self.trainer = None
        self.controller = None

        # Initialize UI
        self.init_ui()

        # Load default config
        self.load_default_config('GA')

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Puppets - AI Evolution Simulator")
        self.setGeometry(100, 100, 1400, 900)

        # Create menu bar
        self.create_menus()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Control Panel
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(300)
        splitter.addWidget(self.control_panel)

        # Middle/Right panel: Simulation and Stats
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Top: Simulation canvas
        self.pygame_widget = PygameWidget()
        right_layout.addWidget(self.pygame_widget, stretch=3)

        # Bottom splitter: Stats and Charts
        bottom_splitter = QSplitter(Qt.Horizontal)

        # Stats panel
        self.stats_panel = StatsPanel()
        bottom_splitter.addWidget(self.stats_panel)

        # Charts
        self.chart_widget = ChartWidget()
        bottom_splitter.addWidget(self.chart_widget)

        right_layout.addWidget(bottom_splitter, stretch=2)

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Connect signals
        self.connect_signals()

    def create_menus(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.on_new)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Config", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.on_open_config)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Config", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.on_save_config)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        fullscreen_action = QAction("&Fullscreen Simulation", self)
        fullscreen_action.setShortcut(Qt.Key_F11)
        fullscreen_action.triggered.connect(self.toggle_fullscreen_simulation)
        view_menu.addAction(fullscreen_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        export_action = QAction("&Export Data", self)
        export_action.triggered.connect(self.on_export_data)
        tools_menu.addAction(export_action)

        compare_action = QAction("&Compare Runs", self)
        compare_action.triggered.connect(self.on_compare_runs)
        tools_menu.addAction(compare_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)

    def connect_signals(self):
        """Connect widget signals to slots."""
        # Control panel signals
        self.control_panel.method_changed.connect(self.on_method_changed)
        self.control_panel.start_clicked.connect(self.on_start)
        self.control_panel.pause_clicked.connect(self.on_pause)
        self.control_panel.stop_clicked.connect(self.on_stop)
        self.control_panel.speed_changed.connect(self.on_speed_changed)
        self.control_panel.config_changed.connect(self.on_config_changed)

    def load_default_config(self, method):
        """Load default configuration for method."""
        try:
            self.current_config = self.config_manager.load_default(method)
            self.control_panel.set_config(self.current_config)
            self.statusBar().showMessage(f"Loaded default {method} configuration")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

    def create_trainer(self, method):
        """Create trainer for the selected method."""
        if method == 'GA':
            return GATrainer(self.current_config)
        elif method == 'CNN':
            return CNNTrainer(self.current_config)
        elif method == 'DQN':
            return DQNTrainer(self.current_config)
        else:
            raise ValueError(f"Unknown method: {method}")

    @pyqtSlot(str)
    def on_method_changed(self, method):
        """Handle method selection change."""
        if self.controller and self.controller.is_running:
            QMessageBox.warning(self, "Warning", "Please stop training before changing method")
            return

        self.load_default_config(method)

    @pyqtSlot()
    def on_start(self):
        """Start training."""
        if self.controller and self.controller.is_running:
            self.statusBar().showMessage("Training already running")
            return

        try:
            # Create trainer
            method = self.current_config['method']
            self.trainer = self.create_trainer(method)

            # Create controller
            self.controller = SimulationController(self.trainer)

            # Connect controller signals
            self.controller.metrics_updated.connect(self.on_metrics_updated)
            self.controller.status_changed.connect(self.on_status_changed)
            self.controller.error_occurred.connect(self.on_error)
            self.controller.training_finished.connect(self.on_training_finished)

            # Start pygame widget
            self.pygame_widget.start_rendering(self.trainer)

            # Start training
            self.controller.start()

            # Update UI
            self.control_panel.set_training_active(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training: {e}")
            import traceback
            traceback.print_exc()

    @pyqtSlot()
    def on_pause(self):
        """Pause training."""
        if self.controller:
            if self.controller.is_paused.is_set():
                self.controller.resume()
                self.control_panel.set_paused(False)
            else:
                self.controller.pause()
                self.control_panel.set_paused(True)

    @pyqtSlot()
    def on_stop(self):
        """Stop training."""
        if self.controller:
            self.controller.stop()
            self.pygame_widget.stop_rendering()
            self.control_panel.set_training_active(False)

    @pyqtSlot(int)
    def on_speed_changed(self, fps):
        """Handle speed slider change."""
        if self.controller:
            self.controller.set_speed(fps)

    @pyqtSlot(dict)
    def on_config_changed(self, config):
        """Handle configuration change."""
        self.current_config = config

    @pyqtSlot(dict)
    def on_metrics_updated(self, metrics):
        """Handle metrics update from trainer."""
        # Update stats panel
        self.stats_panel.update_metrics(metrics)

        # Update charts
        self.chart_widget.add_data_point(metrics)

    @pyqtSlot(str)
    def on_status_changed(self, status):
        """Handle status message."""
        self.statusBar().showMessage(status)

    @pyqtSlot(str)
    def on_error(self, error):
        """Handle error."""
        QMessageBox.critical(self, "Error", error)
        self.statusBar().showMessage(f"Error: {error}")

    @pyqtSlot()
    def on_training_finished(self):
        """Handle training completion."""
        self.pygame_widget.stop_rendering()
        self.control_panel.set_training_active(False)
        QMessageBox.information(self, "Complete", "Training finished!")

    def on_new(self):
        """Create new configuration."""
        method = self.current_config['method'] if self.current_config else 'GA'
        self.load_default_config(method)

    def on_open_config(self):
        """Open configuration file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Configuration", "", "JSON Files (*.json)"
        )
        if filename:
            try:
                self.current_config = self.config_manager.load_from_file(filename)
                self.control_panel.set_config(self.current_config)
                self.statusBar().showMessage(f"Loaded config from {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

    def on_save_config(self):
        """Save configuration file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json)"
        )
        if filename:
            try:
                self.config_manager.save_to_file(self.current_config, filename)
                self.statusBar().showMessage(f"Saved config to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save config: {e}")

    def on_export_data(self):
        """Export training data."""
        # TODO: Implement data export
        QMessageBox.information(self, "TODO", "Data export not yet implemented")

    def on_compare_runs(self):
        """Compare multiple training runs."""
        # TODO: Implement comparison
        QMessageBox.information(self, "TODO", "Run comparison not yet implemented")

    def toggle_fullscreen_simulation(self):
        """Toggle fullscreen for simulation canvas."""
        # TODO: Implement fullscreen
        pass

    def on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Puppets",
            "Puppets - AI Evolution Simulator\n\n"
            "Compare Genetic Algorithms, CNN, and Deep Q-Learning\n"
            "for evolving agents in a simulated environment.\n\n"
            "Version 2.0 - Refactored Edition"
        )

    def closeEvent(self, event):
        """Handle window close."""
        if self.controller and self.controller.is_running:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Training is still running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.controller.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
