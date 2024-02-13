import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit
import PyPDF2

class PDFToTextConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 600, 400)
        self.setWindowTitle('PDF to Text Converter')

        self.text_edit = QTextEdit(self)
        self.text_edit.setGeometry(20, 20, 560, 250)

        self.btn_open = QPushButton('Open PDF', self)
        self.btn_open.setGeometry(50, 300, 100, 30)
        self.btn_open.clicked.connect(self.openPDF)

        self.btn_convert = QPushButton('Convert', self)
        self.btn_convert.setGeometry(200, 300, 100, 30)
        self.btn_convert.clicked.connect(self.convertToText)

        self.btn_exit = QPushButton('Exit', self)
        self.btn_exit.setGeometry(350, 300, 100, 30)
        self.btn_exit.clicked.connect(self.close)

    def openPDF(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Open PDF', '', 'PDF files (*.pdf)')
        if file_path:
            self.file_path = file_path

    def convertToText(self):
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfFileReader(file)
                text_array = []
                for page_num in range(pdf_reader.numPages):
                    page = pdf_reader.getPage(page_num)
                    text_array.append(page.extractText())
                self.text_edit.setPlainText('\n\n'.join(text_array))
        except Exception as e:
            self.text_edit.setPlainText(f"Error: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    converter = PDFToTextConverter()
    converter.show()
    sys.exit(app.exec_())
