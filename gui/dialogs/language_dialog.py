"""
Language selection dialog.
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QRadioButton, QButtonGroup, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal

from core.localization import get_language


class LanguageDialog(QDialog):
    """
    Dialog for selecting application language.
    """

    language_changed = pyqtSignal(str)  # Emits new language code

    def __init__(self, parent=None):
        super().__init__(parent)

        self.selected_language = get_language()
        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        self.setWindowTitle("Language / Język / Langue")
        self.resize(300, 200)

        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel("Select language / Wybierz język / Sélectionner langue:")
        layout.addWidget(info_label)

        # Language selection group
        lang_group = QGroupBox("Languages")
        lang_layout = QVBoxLayout()

        self.button_group = QButtonGroup(self)

        # English
        self.en_radio = QRadioButton("English")
        self.button_group.addButton(self.en_radio, 0)
        lang_layout.addWidget(self.en_radio)

        # Polish
        self.pl_radio = QRadioButton("Polski")
        self.button_group.addButton(self.pl_radio, 1)
        lang_layout.addWidget(self.pl_radio)

        # French
        self.fr_radio = QRadioButton("Français")
        self.button_group.addButton(self.fr_radio, 2)
        lang_layout.addWidget(self.fr_radio)

        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)

        # Set current language
        current = get_language()
        if current == 'EN':
            self.en_radio.setChecked(True)
        elif current == 'PL':
            self.pl_radio.setChecked(True)
        elif current == 'FR':
            self.fr_radio.setChecked(True)

        # Buttons
        button_layout = QHBoxLayout()

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel / Anuluj / Annuler")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def get_selected_language(self) -> str:
        """Get selected language code."""
        if self.en_radio.isChecked():
            return 'EN'
        elif self.pl_radio.isChecked():
            return 'PL'
        elif self.fr_radio.isChecked():
            return 'FR'
        return 'EN'
