from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLineEdit, QPushButton, QListWidget, QListWidgetItem, QCheckBox
from PyQt5.QtCore import QFileInfo, Qt
from PyQt5 import uic
import biopixelguiresource
import sys
from biopixel import BioPixelEntry
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load the UI file using uic
        uic.loadUi("biopixel_ui.ui", self)
        
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Access the UI elements
        self.lineEditPath = self.findChild(QLineEdit, "lineDirectory")
        self.buttonBrowse = self.findChild(QPushButton, "pushBrowse")
        self.buttonRun = self.findChild(QPushButton, "pushRun")
        self.checkKeepTif = self.findChild(QCheckBox, "checkKeepTif")
        self.checkDetectCells = self.findChild(QCheckBox, "checkDetectCells")
        self.listImageFiles = self.findChild(QListWidget, "listImageFiles")
        self.pushClose = self.findChild(QPushButton, "pushExit")

        # Connect the browse button to the slot
        self.buttonBrowse.clicked.connect(self.browse_files)
        self.pushClose.clicked.connect(self.close)
        self.buttonRun.clicked.connect(self.run)
        self.lineEditPath.textChanged.connect(self.toggle_list_widget_visibility)
        
        # Initially hide the list widget
        self.listImageFiles.hide()

        self._biopixelproject = BioPixelEntry()

    def toggle_list_widget_visibility(self):
        # Show the list widget only if there is text in the line edit
        if self.lineEditPath.text():
            self.listImageFiles.show()
        else:
            self.listImageFiles.hide()

    def adjust_list_widget_height(self):
        # Define a maximum height for the list widget
        max_visible_items = 5
        item_height = self.listImageFiles.sizeHintForRow(0) +2
        total_items = self.listImageFiles.count()

        if total_items > max_visible_items:
            self.listImageFiles.setFixedHeight(max_visible_items * item_height)
        else:
            self.listImageFiles.setFixedHeight(total_items * item_height)

    def browse_files(self):
        # Open a file dialog to select either a file or a directory
        file_or_folder_path = QFileDialog.getExistingDirectory(self, 'Most Bodacious Image Folder', options=QFileDialog.Option.ShowDirsOnly)

        # If the user selected a directory, set it to the line edit
        if file_or_folder_path:
            self.lineEditPath.setText(file_or_folder_path)
            self.listImageFiles.clear()  # Clear the list widget

            # Get the list of file paths with the desired extension (e.g., .png)
            #file_paths = get_file_list(file_or_folder_path)  # Adjust the extension as needed

            self._biopixelproject.set_working_directory(file_or_folder_path)

            # Add file paths to the list widget with checkboxes
            for file_path in self._biopixelproject.images:
                file_name = QFileInfo(file_path).fileName()
                item = QListWidgetItem(file_name)
                item.setCheckState(True)  # Unchecked by default
                self.listImageFiles.addItem(item)
            
            # Dynamically adjust the height of the list widget
            self.adjust_list_widget_height()

    def get_selected_files(self):
        selected_files = []
        for index in range(self.listImageFiles.count()):
            item = self.listImageFiles.item(index)
            if item.checkState():
                selected_files.append(os.path.join(self.lineEditPath.text(), item.text()))
        return selected_files
            
    def run(self):
        print(self._biopixelproject.images)
        if not self._biopixelproject.images:
            print("No images to process")
            return
        # Assuming you have a checkbox for keeping TIFFs
        keep_tif = self.checkKeepTif.isChecked()
        detect_cells = self.checkDetectCells.isChecked()

        self._biopixelproject.images = self.get_selected_files()

        self._biopixelproject.process_images(keep_tif=keep_tif, detect_cells=detect_cells)
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

