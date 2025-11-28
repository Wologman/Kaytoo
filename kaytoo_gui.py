#  To compile to .exe:  
# (bird_audio_infer) PS E:\Kaytoo\Python> pyinstaller --name kaytoo --onedir  
# --clean ./src/kaytoo_gui.py --distpath ./ --icon=./resources/images/kaytoo_icon.ico  
import os
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtWidgets import (
    QApplication,
    QSystemTrayIcon,
    QHBoxLayout,
    QGraphicsDropShadowEffect,
    QWidget,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QCheckBox,
    QMessageBox,
    QMainWindow,
)
from PyQt5.QtGui import QDesktopServices, QIcon, QMouseEvent, QResizeEvent
from PyQt5.QtCore import QDir, QEvent, QThread, Qt, QUrl, pyqtSignal
import pandas as pd
from bird_naming_utils import BirdNamer
from kaytoo_infer import infer_soundscapes
from pathlib import Path
import tempfile
from multiprocessing import freeze_support


warnings.filterwarnings("ignore", category=UserWarning, message='A new version')
QDir.setCurrent(str(Path(__file__).parent))
root = str(Path(__file__).parent.parent) 
QDir.addSearchPath('images', os.path.join(root, 'resources', 'images'))

PRIMARY_ACCENT_COLOR = "#989898"
PRIMARY_BACKGROUND_COLOR = "#0d4733"


class FilePaths:
    """Container for resource paths used by the GUI."""

    AUDIO_TYPES: set[str] = {'.ogg', '.wav', '.flac', '.mp3'}

    def __init__(self, root: Path) -> None:
        self.root_folder: Path = Path(root)
        if not self.root_folder.exists():
            raise FileNotFoundError(f"Project root not found at {self.root_folder}")
        self.css: Path = self.root_folder / 'resources/kaytoo_gui_styles.css'
        self.images: Path = self.root_folder / 'resources/images'
        icon_file = self.images / 'weka_small.png'
        self.icon_path: str = icon_file.as_posix()
        self.bird_name_csv: Path = self.root_folder / 'resources/bird_map.csv'


class HoverLabel(QLabel):
    def __init__(
        self,
        text: str,
        link_text: str,
        link: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.link: str = link
        self.link_visible: str = link_text
        self.text_content: str = text  # Store the original text without HTML formatting
        self.normal_style: str = f'<span style="color: {PRIMARY_ACCENT_COLOR}; font-weight: normal; padding:0; margin:0;">{{}}</span>'
        self.link_style: str = f'<a href="{{}}" style="color: {PRIMARY_ACCENT_COLOR}; text-decoration: underline; padding:0; margin:0;">{{}}</a>'
        
        # Construct the initial HTML content
        self.setText(self.format_text(self.text_content, self.link_visible, self.link))
        self.setCursor(Qt.PointingHandCursor)  # Change cursor to hand on hover

    def format_text(self, text: str, link_visible: str, link: str, hover: bool = False) -> str:
        """Helper method to format text with styles"""
        regular_text = self.normal_style.format(text)
        if hover:
            link_text = f'<a href="{link}" style="color: white; text-decoration: none;">{link_visible}</a>'
        else:
            link_text = self.link_style.format(link, link_visible)
        return f"{regular_text} {link_text}"

    def enterEvent(self, event: QEvent) -> None:
        # Change link style on hover (bold and no underline)
        self.setText(self.format_text(self.text_content, self.link_visible, self.link, hover=True))
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        # Restore original style when mouse leaves
        self.setText(self.format_text(self.text_content, self.link_visible, self.link, hover=False))
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        QDesktopServices.openUrl(QUrl(self.link))
        super().mousePressEvent(event)


class HoverButton(QPushButton):
    default_style = f"""
                    background-color: {PRIMARY_BACKGROUND_COLOR}; 
                    color: {PRIMARY_ACCENT_COLOR};
                    text-decoration: underline; 
                    font-weight: normal; 
                    padding:4px;
                    """
    hover_style = f"""
                    background-color: {PRIMARY_BACKGROUND_COLOR}; 
                    color: white; 
                    font-weight: normal; 
                    padding:4px;
                    """
    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self._is_shadow = True
        self.setStyleSheet(self.default_style)
        self.setCursor(Qt.PointingHandCursor)
    
    def enterEvent(self, event: QEvent) -> None:
        self.setStyleSheet(self.hover_style)
        self.shadow = QGraphicsDropShadowEffect() 
        self.shadow.setBlurRadius(15) 
        self.shadow.setColor(Qt.black)
        self.setGraphicsEffect(self.shadow)
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self.setStyleSheet(self.default_style)
        self.shadow.setEnabled(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if hasattr(self, 'click_handler'):
            self.click_handler()

    def set_click_handler(self, handler: Callable[[], None]) -> None:
        self.click_handler: Callable[[], None] = handler


class InferenceThread(QThread):
    """Background thread for running inference to prevent GUI blocking."""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, arguments: Dict[str, Any]) -> None:
        super().__init__()
        self.arguments: Dict[str, Any] = arguments
    
    def run(self) -> None:
        """Execute inference in background thread."""
        try:
            infer_soundscapes(self.arguments)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, paths: FilePaths) -> None:
        super().__init__()
        self.setWindowTitle("Kaytoo | New Zealand Birdcall Classifier")
        self.setGeometry(50, 50, 550, 800)      
        
        self.paths: FilePaths = paths
        self.parent_folder: Path = paths.root_folder
        if not paths.bird_name_csv.exists():
            QMessageBox.critical(
                self,
                "Missing Data",
                f"Unable to locate bird name CSV:\n{paths.bird_name_csv}",
            )
            raise FileNotFoundError(f"Missing bird map CSV at {paths.bird_name_csv}")

        try:
            bird_map_df = pd.read_csv(paths.bird_name_csv)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to read bird name CSV:\n{paths.bird_name_csv}\n\nError: {exc}",
            )
            raise

        self.birdnames = BirdNamer(bird_map_df)
        
        self.grey_text = f'style="color: {PRIMARY_ACCENT_COLOR}; font-weight: Bold"'

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

        source_1 = HoverLabel("Version 2.00 | November 2025 | ", "Documentation", "https://wekaresearch.com")
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

        _thresholds_list = ['0.1', '0.2', '0.25', '0.3', '0.35', '0.4', '0.5', '0.6']
        
        self.threshold_label = QLabel("Detection Threshold:")
        self.threshold_help_button = QPushButton('?')
        self.threshold_help_button.setObjectName('help')
        self.threshold_help_button.clicked.connect(self.show_threshold_message)
        self.threshold_combobox =  QComboBox(self)
        self.threshold_combobox.addItems(_thresholds_list)
        self.threshold_combobox.setCurrentText('0.3')
        self.threshold_h_layout = QHBoxLayout()
        self.threshold_h_layout.addWidget(self.threshold_label)
        self.threshold_h_layout.addWidget(self.threshold_help_button)
        self.threshold_h_layout.setContentsMargins(0,0,0,0)
        self.threshold_h_layout.addStretch()
        self.threshold_v_layout = QVBoxLayout()
        self.threshold_v_layout.addLayout(self.threshold_h_layout)
        self.threshold_v_layout.addWidget(self.threshold_combobox)

        _core_list = [1, 2, 4, 6, 8, 12, 16, 24, 36]
        _available_cores = os.cpu_count() or 1
        _core_list = [str(x) for x in _core_list if x <= _available_cores//2]
        _default_cores = str(_core_list[1] if len(_core_list) >= 2 else _core_list[0])
        
        self.cores_label = QLabel("Number of CPU cores to use:")
        self.cores_help_button = QPushButton('?')
        self.cores_help_button.setObjectName('help')
        self.cores_help_button.clicked.connect(self.show_cores_message)
        self.cores_combobox =  QComboBox(self)
        self.cores_combobox.addItems(_core_list)
        self.cores_combobox.setCurrentText(_default_cores)
        self.cores_h_layout = QHBoxLayout()
        self.cores_h_layout.addWidget(self.cores_label)
        self.cores_h_layout.addWidget(self.cores_help_button)
        self.cores_h_layout.setContentsMargins(0,0,0,0)
        self.cores_h_layout.addStretch()
        self.cores_v_layout = QVBoxLayout()
        self.cores_v_layout.addLayout(self.cores_h_layout)
        self.cores_v_layout.addWidget(self.cores_combobox)

        self.gpu_checkbox = QCheckBox("Run on GPU", self)
        self.gpu_help_button = QPushButton('?')
        self.gpu_help_button.setObjectName('gpu_help')
        self.gpu_help_button.clicked.connect(self.show_gpu_message)
        self.gpu_h_layout = QHBoxLayout()
        self.gpu_h_layout.addWidget(self.gpu_checkbox)
        self.gpu_h_layout.addWidget(self.gpu_help_button)
        self.gpu_h_layout.setContentsMargins(0,0,0,0)
        self.gpu_h_layout.addStretch()

        self.naming_label = QLabel("Naming scheme")
        self.naming_combo = QComboBox(self)
        _naming_list = ["Short (some species merged)", "Long", "Scientific", "eBird",]
        self.naming_combo.addItems(_naming_list)
        self.naming_combo.setCurrentText(_naming_list[0])
        self.naming_combo.currentIndexChanged.connect(self.naming_changed)

        self.bird_combo_label = QLabel("Birds to summarise")
        self.bird_combo = QComboBox(self)
        self.bird_combo.addItems(["All"] + self.birdnames.short_names_list)

        self.run_button = HoverButton("Run Kaytoo")
        self.run_button.clicked.connect(self.run_program)
        self.inference_thread: Optional[InferenceThread] = None  # Will hold the background thread
        
        attribution_text = "Made in Nelson for the Department of Conservation - Te Papa Atawhai | By " 
        self.attribution = HoverLabel(attribution_text, "wekaResearch", "https://wekaresearch.com")

        layout.addLayout(self.title_layout)
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_path)
        layout.addWidget(self.input_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_path)
        layout.addWidget(self.output_button)
        layout.addLayout(self.threshold_v_layout)
        layout.addLayout(self.cores_v_layout)
        layout.addLayout(self.gpu_h_layout)
        layout.addWidget(self.naming_label)
        layout.addWidget(self.naming_combo)
        layout.addWidget(self.bird_combo_label)
        layout.addWidget(self.bird_combo)
        layout.addWidget(self.run_button)
        layout.addStretch() 
        layout.addWidget(self.attribution)


    def show_threshold_message(self) -> None:
        QMessageBox.information(self, "Detection Threshold", 
                                "For each 5 second period, and each bird class,\n"
                                "predictions of presence or absence are made based on\n"
                                "model probability scores relative to this threshold.\n\n"

                                "A higher threshold means fewer false detetections, "
                                "but you may miss some bird calls. "
                                "Using a lower threshold means you detect all the calls you can, "
                                "but more may be erroneous.\n\n"

                                "0.3 is a good compromise for the current model."
                                )
    
    def show_cores_message(self) -> None:
        QMessageBox.information(self, "Number of cores", 
                                "The number of CPU cores to be used for parallel processing\n\n"
                                "By default the menu allows up to half your thread count\n"
                                "Generally the more cores the faster the processing. But\n"
                                "more cores also increases the chance of a crash due to insufficient memory.\n\n"
                                "Be careful if you want to use your machine for other stuff at the same time.\n\n"
                                "Set to the max CPU limit if you're feeling lucky.... punk"
                                )
        
    def show_gpu_message(self) -> None:
        QMessageBox.information(self, "GPU usage", 
                                "If you tick this box, the program will try to use your GPU for increased speed.\n\n"
                                "Only an NVIDIA GPU will work, and the program will crash if it has insufficient memory. "
                                "If you do get an out-of-memory error, try backing off the CPU count.\n\n"
                                "The program was tested successfully on a Dell 3420 using an an NVIDIA MX450 GPU, whilst "
                                "set to use 2 CPU cores."
                                )

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.floating_label.move(self.width() - self.floating_label.width() - 10, 10)


    def _select_folder(self, line_edit: QLineEdit, dialog_title: str) -> None:
        """Helper method to select a folder and update the given line edit."""
        # Prefer /media if it exists and contains subfolders
        default_dir = Path("/media") if Path("/media").exists() else self.parent_folder

        file_dialog = QFileDialog.getExistingDirectory(self, dialog_title, str(default_dir))
        if file_dialog and os.path.isdir(file_dialog):
            line_edit.setText(file_dialog)
    
    def select_input_file(self) -> None:
        """Select input folder for audio files."""
        self._select_folder(self.input_path, "Select Input Folder")

    def select_output_file(self) -> None:
        """Select output folder for results."""
        self._select_folder(self.output_path, "Select Output Folder")

    def naming_changed(self) -> None:
        selected_index = self.naming_combo.currentIndex()

        options = [self.birdnames.short_names_list,
                   self.birdnames.long_names_list,
                   self.birdnames.science_names_list,
                   self.birdnames.bird_list]
        self.bird_combo.clear()
        self.bird_combo.addItems(["All"] + options[selected_index])

    def run_program(self) -> None:
        input_folder = self.input_path.text()
        output_folder = self.output_path.text()
        threshold = float(self.threshold_combobox.currentText())
        num_cores = int(self.cores_combobox.currentText())
        use_cpu = not bool(self.gpu_checkbox.isChecked())
        naming_index = self.naming_combo.currentIndex()
        current_bird = self.bird_combo.currentText()
        chosen_birds: List[str] = [] if current_bird == "All" else [current_bird]

        if not input_folder or not output_folder:
            _ = QMessageBox.critical(self, "Unable to proceed", "Please specify both input and output folder paths.")
            return
        if not os.path.isdir(input_folder):
            _ = QMessageBox.critical(self, "Perdóneme amigo", "The input folder path is not a valid directory.")
            return
        if not os.path.isdir(output_folder):
            _ = QMessageBox.critical(self, "Unable to proceed", "The output folder path is not a valid directory.")
            return

        _naming_schemes = ["Short", "Long", "Scientific", "eBird"]

        arguments = {
                'project_root': self.parent_folder,
                'experiment': None,
                'threshold': threshold,
                'folder_to_process': input_folder,
                'results_folder': output_folder,
                'naming_scheme': _naming_schemes[naming_index],
                'cpu_only': use_cpu,
                'num_cores': num_cores,
                'birds_to_summarise': chosen_birds,
                }
    
        # Disable button and start background thread
        self.run_button.setEnabled(False)
        self.run_button.setText("Processing...")
        
        # Create and start inference thread
        self.inference_thread = InferenceThread(arguments)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.start()
    
    def on_inference_finished(self) -> None:
        """Handle successful completion of inference."""
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Kaytoo")
        QMessageBox.information(self, "Success", "Kaytoo completed processing the audio files!")
        self.inference_thread = None
    
    def on_inference_error(self, error_message: str) -> None:
        """Handle errors during inference."""
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Kaytoo")
        QMessageBox.critical(self, "Processing Error", 
                           f"An error occurred during processing:\n{error_message}\n\nPlease check the input files and try again.")
        self.inference_thread = None


def get_project_root() -> Path:
    """
    Determine project root depending on execution context:
    - --onefile: sys._MEIPASS will be a temp directory
    - --onedir: use parent of executable (then go up one)
    - dev mode: use __file__
    """

    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass and Path(meipass).resolve().parent == Path(tempfile.gettempdir()).resolve():
            # This is really --onefile mode
            return Path(meipass).resolve()  # type: ignore[attr-defined]
        else:
            # This is really --onedir mode
            return Path(sys.executable).resolve().parent.parent
    else:
        return Path(__file__).resolve().parent.parent



if __name__ == "__main__":
    freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning, message='A new version')

    root = get_project_root()
    paths = FilePaths(root)
    QDir.addSearchPath('images', str(paths.images))
    app = QApplication(sys.argv)

    if paths.css.exists():
        try:
            with open(paths.css, 'r', encoding='utf-8') as file:
                app.setStyleSheet(file.read())
        except Exception as exc:
            print(f"[WARN] Failed to load CSS from {paths.css}: {exc}")
    else:
        print(f"[WARN] CSS file not found at {paths.css}")
    window = MainWindow(paths)
    tray_icon = QSystemTrayIcon()
    icon = QIcon(paths.icon_path)
    tray_icon.setIcon(icon)
    tray_icon.show()
    window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec_())