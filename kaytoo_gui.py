import sys
import pathlib
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon,QHBoxLayout, QGraphicsDropShadowEffect, QWidget, QLabel, QLineEdit, QComboBox, QPushButton, QVBoxLayout, QFileDialog, QSpinBox, QCheckBox, QMessageBox, QMainWindow
from PyQt5.QtGui import QDesktopServices, QIcon
from PyQt5.QtCore import Qt, QUrl
import pandas as pd
from bird_naming_utils import BirdNamer
from kaytoo_infer import infer_soundscapes


class HoverLabel(QLabel):
    def __init__(self, text, link_text, link, parent=None):
        super().__init__(parent)
        self.link = link
        self.link_visible = link_text
        self.text_content = text  # Store the original text without HTML formatting
        self.normal_style = '<span style="color: #989898; font-weight: normal; padding:0; margin:0;">{}</span>'
        self.link_style = '<a href="{}" style="color: #989898; text-decoration: underline; padding:0; margin:0;">{}</a>'
        
        # Construct the initial HTML content
        self.setText(self.format_text(self.text_content, self.link_visible, self.link))
        self.setCursor(Qt.PointingHandCursor)  # Change cursor to hand on hover

    def format_text(self, text, link_visible, link, hover=False):
        """Helper method to format text with styles"""
        regular_text = self.normal_style.format(text)
        if hover:
            link_text = f'<a href="{link}" style="color: white; text-decoration: none;">{link_visible}</a>'
        else:
            link_text = self.link_style.format(link, link_visible)
        return f"{regular_text} {link_text}"

    def enterEvent(self, event):
        # Change link style on hover (bold and no underline)
        self.setText(self.format_text(self.text_content, self.link_visible, self.link, hover=True))
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Restore original style when mouse leaves
        self.setText(self.format_text(self.text_content, self.link_visible, self.link, hover=False))
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        QDesktopServices.openUrl(QUrl(self.link))


class HoverButton(QPushButton):
    default_style = """
                    background-color: #0d4733; 
                    color: #989898;
                    text-decoration: underline; 
                    font-weight: normal; 
                    padding:4px;
                    """
    hover_style = """
                    background-color: #0d4733; 
                    color: white; 
                    font-weight: normal; 
                    padding:4px;
                    """
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._is_shadow = True
        self.setStyleSheet(self.default_style)
        self.setCursor(Qt.PointingHandCursor)
    
    def enterEvent(self, event):
        self.setStyleSheet(self.hover_style)
        #self.shadow.setEnabled(False)
        self.shadow = QGraphicsDropShadowEffect() 
        self.shadow.setBlurRadius(15) 
        self.shadow.setColor(Qt.black)
        self.setGraphicsEffect(self.shadow)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self.default_style)
        self.shadow.setEnabled(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if hasattr(self, 'click_handler'):
            self.click_handler()

    def set_click_handler(self, handler):
        self.click_handler = handler


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kaytoo | New Zealand Birdcall Classifier")
        self.setGeometry(50, 50, 550, 800)      
        
        parent_folder = pathlib.Path(__file__).parent.parent
        image_folder = parent_folder   / 'images'
        self.icon_path = (image_folder / 'weka_small.png').as_posix()
        bird_name_csv_path = parent_folder / 'Current_Version/bird_map.csv'

        bird_map_df = pd.read_csv(bird_name_csv_path)
        self.birdnames = BirdNamer(bird_map_df)
        
        self.grey_text = 'style="color: #989898; font-weight: Bold"'

        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        logo_text = f"weka<span {self.grey_text}>Research</span>"
        self.floating_label = QLabel(logo_text, self)
        self.floating_label.setObjectName("logo")

        self.floating_label.setAlignment(Qt.AlignCenter)
        self.floating_label.setFixedSize(150, 160)
        self.floating_label.move(self.width() - self.floating_label.width() - 10, 10)
        
        self.title = QLabel("Kaytoo")
        self.title.setObjectName("title")

        self.info_label = QLabel("Identification of New Zealand birds from sound recordings")
        self.info_label.setObjectName("info_label")


        source_1 = HoverLabel("Version 1.0 | September 2024 | ", "Documentation", "https://wekaresearch.com")
        source_2 = HoverLabel("|", "Source Code", "https://github.com/Wologman/Kaytoo")
        
        
        self.sources_layout = QHBoxLayout()
        self.sources_layout.addWidget(source_1)
        self.sources_layout.addWidget(source_2)
        self.sources_layout.setContentsMargins(0,0,0,30)
        self.sources_layout.setSpacing(0)
        self.sources_layout.addStretch()

        self.title_layout = QVBoxLayout()
        self.title_layout.addWidget(self.title)
        self.title_layout.addWidget(self.info_label)
        self.title_layout.addLayout(self.sources_layout)
        
        self.input_label = QLabel("Folder containing the sound files:")
        self.input_path = QLineEdit(self)
        self.input_path.setPlaceholderText("Enter path here...")
        self.input_button = HoverButton("Browse for folder")
        self.input_button.set_click_handler(self.select_input_file)

        self.output_label = QLabel("Folder to save results in:")
        self.output_path = QLineEdit(self)
        self.output_path.setPlaceholderText("Enter path here...")

        self.output_button = HoverButton("Browse for folder")
        self.output_button.clicked.connect(self.select_output_file)

        self.cores_label = QLabel("Number of CPU cores to use:")
        self.cores_combobox =  QComboBox(self)
        self.cores_combobox.addItems(['1', '2', '4', '6', '8', '12'])
        self.cores_combobox.currentIndexChanged.connect(self.cores_changed)

        self.gpu_checkbox = QCheckBox("Run on GPU", self)

        self.naming_label = QLabel("Naming scheme")
        self.naming = QComboBox(self)
        self.naming.addItems(["Short (some species merged)", "Long", "Scientific", "eBird",])
        self.naming.currentIndexChanged.connect(self.naming_changed)

        self.combo_label = QLabel("Birds to summarise")
        self.combo = QComboBox(self)
        self.combo.addItems(["All"] + self.birdnames.science_names_list)
        self.combo.currentIndexChanged.connect(self.bird_selection_changed)

        self.run_button = HoverButton("Run Kaytoo")
        self.run_button.clicked.connect(self.run_program)
        
        attribution_text = "Made in Nelson for the Department of Conservation - Te Papa Atawhai | By " 
        self.attribution = HoverLabel(attribution_text, "wekaResearch", "https://wekaresearch.com")


        layout.addLayout(self.title_layout)
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_path)
        layout.addWidget(self.input_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_path)
        layout.addWidget(self.output_button)
        layout.addWidget(self.cores_label)
        layout.addWidget(self.cores_combobox)
        layout.addWidget(self.gpu_checkbox)
        layout.addWidget(self.naming_label)
        layout.addWidget(self.naming)
        layout.addWidget(self.combo_label)
        layout.addWidget(self.combo)
        layout.addWidget(self.run_button)
        layout.addStretch() 
        layout.addWidget(self.attribution)

    def resizeEvent(self, event):
        #super().resizeEvent(event)
        self.floating_label.move(self.width() - self.floating_label.width() - 10, 10)
        
    def select_input_file(self):
        file_dialog = QFileDialog.getExistingDirectory(self, "Select Input Folder", "")
        if file_dialog[0]:
            self.input_path.setText(file_dialog)

    def select_output_file(self):
        file_dialog = QFileDialog.getExistingDirectory(self, "Select Input Folder", "")
        if file_dialog[0]:
            self.output_path.setText(file_dialog)

    def cores_changed(self):
        selected_index = self.cores_combobox.currentIndex()
        selected_option = self.cores_combobox.itemText(selected_index)
        print(int(selected_option))
        self.cores_combobox.itemText(int(selected_option))
        print(self.cores_combobox.currentText())

    def bird_selection_changed(self):
        selected_option = self.combo.currentText()
        

    def naming_changed(self):
        selected_index = self.naming.currentIndex()
        selected_option = self.naming.itemText(selected_index)
        print(selected_option)

        options = [self.birdnames.short_names_list,
                   self.birdnames.long_names_list,
                   self.birdnames.science_names_list,
                   self.birdnames.bird_list]
        self.combo.clear()
        self.combo.addItems(["All"] + options[selected_index])

    def run_program(self):
        input_folder = self.input_path.text()
        output_folder = self.output_path.text()
        num_cores = self.cores_combobox.currentText()
        use_gpu = self.gpu_checkbox.isChecked()
        naming_scheme = self.naming.currentText

        if not input_folder or not output_folder:
            msg = QMessageBox.critical(self, "Waaaaah?", "Please specify both input and output folder paths.")
            return

        arguments = {
                'project_root': pathlib.Path(__file__).parent.parent,
                'experiment': 19,
                'threshold': 0.2,
                'folder_to_process': input_folder,
                'results_folder': output_folder,
                'naming_scheme': naming_scheme,
                'model_choices': [0],
                'cpu_only': ~use_gpu,
                'num_cores': num_cores,
                }
        try:
            infer_soundscapes(arguments)
            QMessageBox.information(self, "Yay, Success", "Program completed successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Bummer", f"For some mysterious reason the program failed: {e}")

if __name__ == "__main__":
    project_root = pathlib.Path(__file__).parent.parent
    css_path = project_root / 'Python/kaytoo_gui_styles.css'
    app = QApplication(sys.argv)
    with open(css_path, 'r') as file:
        app.setStyleSheet(file.read())
    window = MainWindow()
    tray_icon = QSystemTrayIcon()
    icon = QIcon(window.icon_path)
    tray_icon.setIcon(icon)
    tray_icon.show()
    window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec_())