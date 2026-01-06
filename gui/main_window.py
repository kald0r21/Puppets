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
from core.localization import tr, set_language, get_language

# Import widgets (we'll create these)
from gui.widgets.control_panel import ControlPanel
from gui.widgets.pygame_widget import PygameWidget
from gui.widgets.stats_panel import StatsPanel
from gui.widgets.chart_widget import ChartWidget
from gui.widgets.param_editor import ParamEditor


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

        # Store menu actions for language switching
        self.menu_actions = {}
        self.menus = {}

        # Initialize UI
        self.init_ui()

        # Load default config
        self.load_default_config('GA')

    def init_ui(self):
        """Initialize the user interface."""
        self.update_window_title()
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
        self.statusBar().showMessage(tr('message.ready'))

        # Connect signals
        self.connect_signals()

    def create_menus(self):
        """Create menu bar."""
        menubar = self.menuBar()
        menubar.clear()

        # File menu
        file_menu = menubar.addMenu(tr('menu.file'))
        self.menus['file'] = file_menu

        new_action = QAction(tr('menu.new'), self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.on_new)
        file_menu.addAction(new_action)
        self.menu_actions['new'] = new_action

        open_action = QAction(tr('menu.open'), self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.on_open_config)
        file_menu.addAction(open_action)
        self.menu_actions['open'] = open_action

        save_action = QAction(tr('menu.save'), self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.on_save_config)
        file_menu.addAction(save_action)
        self.menu_actions['save'] = save_action

        file_menu.addSeparator()

        exit_action = QAction(tr('menu.exit'), self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        self.menu_actions['exit'] = exit_action

        # View menu
        view_menu = menubar.addMenu(tr('menu.view'))
        self.menus['view'] = view_menu

        fullscreen_action = QAction(tr('menu.fullscreen'), self)
        fullscreen_action.setShortcut(Qt.Key_F11)
        fullscreen_action.triggered.connect(self.toggle_fullscreen_simulation)
        view_menu.addAction(fullscreen_action)
        self.menu_actions['fullscreen'] = fullscreen_action

        # Tools menu
        tools_menu = menubar.addMenu(tr('menu.tools'))
        self.menus['tools'] = tools_menu

        params_action = QAction(tr('menu.advanced_params'), self)
        params_action.setShortcut(Qt.Key_F2)
        params_action.triggered.connect(self.on_edit_parameters)
        tools_menu.addAction(params_action)
        self.menu_actions['params'] = params_action

        test_model_action = QAction(tr('menu.test_model'), self)
        test_model_action.setShortcut(Qt.Key_F3)
        test_model_action.triggered.connect(self.on_test_model)
        tools_menu.addAction(test_model_action)
        self.menu_actions['test_model'] = test_model_action

        tools_menu.addSeparator()

        export_action = QAction(tr('menu.export_data'), self)
        export_action.triggered.connect(self.on_export_data)
        tools_menu.addAction(export_action)
        self.menu_actions['export'] = export_action

        compare_action = QAction(tr('menu.compare_runs'), self)
        compare_action.triggered.connect(self.on_compare_runs)
        tools_menu.addAction(compare_action)
        self.menu_actions['compare'] = compare_action

        # Settings menu
        settings_menu = menubar.addMenu(tr('menu.settings'))
        self.menus['settings'] = settings_menu

        language_action = QAction(tr('menu.language'), self)
        language_action.triggered.connect(self.on_change_language)
        settings_menu.addAction(language_action)
        self.menu_actions['language'] = language_action

        # Help menu
        help_menu = menubar.addMenu(tr('menu.help'))
        self.menus['help'] = help_menu

        about_action = QAction(tr('menu.about'), self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
        self.menu_actions['about'] = about_action

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
            self.statusBar().showMessage(tr('message.loaded_config', method=method))
        except Exception as e:
            QMessageBox.critical(self, tr('message.error'), tr('message.failed_load', error=str(e)))

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

        # Load default config for new method
        new_config = self.config_manager.create_default_config(method)

        # Preserve user-customized simulation and visualization settings
        if self.current_config:
            # Keep customized simulation parameters
            if 'simulation' in self.current_config:
                new_config['simulation'] = self.current_config['simulation']
            # Keep customized visualization parameters
            if 'visualization' in self.current_config:
                new_config['visualization'] = self.current_config['visualization']

        self.current_config = new_config
        self.control_panel.set_config(self.current_config)
        self.statusBar().showMessage(tr('message.loaded_config', method=method))

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

            # Create controller with auto-save
            method_lower = method.lower()
            save_dir = f"saved_models/{method_lower}"
            self.controller = SimulationController(self.trainer, save_dir=save_dir, save_interval=10)

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

    def on_edit_parameters(self):
        """Open advanced parameter editor."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QDialogButtonBox
        import copy

        dialog = QDialog(self)
        dialog.setWindowTitle("Advanced Parameters")
        dialog.resize(600, 700)

        layout = QVBoxLayout(dialog)

        # Create parameter editor with a COPY of config
        # This way Cancel will discard changes
        config_copy = copy.deepcopy(self.current_config)
        param_editor = ParamEditor()
        param_editor.set_config(config_copy)
        layout.addWidget(param_editor)

        # Add button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        result = dialog.exec_()

        # If accepted, apply the modified config
        if result == QDialog.Accepted:
            self.current_config = config_copy
            self.control_panel.set_config(self.current_config)
            self.statusBar().showMessage("Parameters updated")

    def on_test_model(self):
        """Open model viewer for testing trained models."""
        from gui.dialogs.model_viewer import ModelViewer

        dialog = ModelViewer(self.current_config, self)
        dialog.exec_()

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
            tr('dialog.about_title'),
            tr('dialog.about_text')
        )

    def on_change_language(self):
        """Open language selection dialog."""
        from gui.dialogs.language_dialog import LanguageDialog

        dialog = LanguageDialog(self)
        result = dialog.exec_()

        if result == LanguageDialog.Accepted:
            new_lang = dialog.get_selected_language()
            if new_lang != get_language():
                set_language(new_lang)
                self.update_ui_language()
                QMessageBox.information(
                    self,
                    tr('message.ready'),
                    f"Language changed to {new_lang}"
                )

    def update_ui_language(self):
        """Update all UI elements with new language."""
        # Update window title
        self.update_window_title()

        # Recreate menus with new translations
        self.create_menus()

        # Update status bar
        self.statusBar().showMessage(tr('message.ready'))

        # Update control panel and other widgets
        if self.control_panel:
            self.control_panel.update_language()

        if self.stats_panel:
            self.stats_panel.update_language()

        if self.chart_widget:
            self.chart_widget.update_language()

    def update_window_title(self):
        """Update window title with current language."""
        self.setWindowTitle(tr('app.title'))

    def closeEvent(self, event):
        """Handle window close."""
        if self.controller and self.controller.is_running:
            reply = QMessageBox.question(
                self,
                tr('dialog.confirm_exit'),
                tr('dialog.training_running'),
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
