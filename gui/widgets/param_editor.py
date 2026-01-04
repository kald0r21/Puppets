"""
Dynamic parameter editor widget.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QScrollArea, QLabel,
                              QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
                              QFormLayout)
from PyQt5.QtCore import pyqtSignal


class ParamEditor(QWidget):
    """
    Widget for editing configuration parameters dynamically.
    """

    param_changed = pyqtSignal(str, str, object)  # section, key, value

    def __init__(self, parent=None):
        super().__init__(parent)

        self.config = None
        self.widgets = {}

        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)

        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)

        scroll.setWidget(self.content_widget)
        layout.addWidget(scroll)

    def set_config(self, config):
        """
        Set configuration and generate parameter widgets.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.widgets.clear()

        # Clear existing widgets
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Generate widgets for each section
        for section in ['simulation', 'ga', 'cnn', 'dqn']:
            if section in config:
                self.create_section(section, config[section])

    def create_section(self, section_name, section_config):
        """
        Create widgets for a config section.

        Args:
            section_name: Section name (e.g., 'simulation')
            section_config: Section configuration dictionary
        """
        group = QGroupBox(section_name.upper())
        form_layout = QFormLayout()

        for key, value in section_config.items():
            widget = self.create_widget_for_value(section_name, key, value)
            if widget:
                form_layout.addRow(f"{key}:", widget)
                self.widgets[f"{section_name}.{key}"] = widget

        group.setLayout(form_layout)
        self.content_layout.addWidget(group)

    def create_widget_for_value(self, section, key, value):
        """
        Create appropriate widget for a config value.

        Args:
            section: Config section
            key: Config key
            value: Config value

        Returns:
            QWidget or None
        """
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            widget.stateChanged.connect(
                lambda state: self.param_changed.emit(section, key, state == 2)
            )
            return widget

        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-1000000, 1000000)
            widget.setValue(value)
            widget.valueChanged.connect(
                lambda v: self.param_changed.emit(section, key, v)
            )
            return widget

        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(-1000000.0, 1000000.0)
            widget.setDecimals(4)
            widget.setValue(value)
            widget.valueChanged.connect(
                lambda v: self.param_changed.emit(section, key, v)
            )
            return widget

        # Skip complex types (lists, dicts)
        return None
