"""
Main entry point for Puppets GUI application.
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """Main function."""
    # Create Qt application
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Puppets")
    app.setOrganizationName("Puppets Project")
    app.setApplicationVersion("2.0")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
