import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit
from PyQt5 import QtCore
import PyPDF2
import threading

from neural_net.neural_net import NeuralNet

class NNComputeThread(QtCore.QThread):
    text_result = QtCore.pyqtSignal(object)
    
    def __init__(self, text_array, nn) -> None:
        super().__init__()
        self._text_array = text_array
        self._nn = nn
        
    def run(self) -> None:
        length = len(str(self._text_array).split(" ", -1))
        if length > 220:# Should be 1024 but I'm not sure what the tokenizer splits on
            self._text_array = str(self._text_array).split(" ", -1)[0:220]
        
        # TODO: Hopefully not need this parsing?
        out_data = self._nn.summarize_text(str(self._text_array))
        word_list = []
        for word in out_data[0]["summary_text"].split():
            if word[0] == "'" or word[-1] == ',':
                if len(word) > 2 and word[-2] == "'":
                    word_list.append(word[1:-2])
                else:
                    word_list.append(word[1:-1])
            else:
                word_list.append(word)
        self.text_result.emit((" ").join(word_list))

class PDFToTextConverter(QMainWindow):
    threads = []
    
    def __init__(self, nn: NeuralNet):
        super().__init__()
        
        self._nn = nn
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
            
    def _onNNComputeComplete(self, data: str):
        self.text_edit.clear()
        self.text_edit.setText(data)

    def convertToText(self):
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_array = []
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_array.append(page.extract_text())
                    
                nn_compute = NNComputeThread(text_array, self._nn)
                nn_compute.text_result.connect(self._onNNComputeComplete)
                nn_compute.start()
                PDFToTextConverter.threads.append(nn_compute)
                
        except Exception as e:
            self.text_edit.clear()
            self.text_edit.setPlainText(f"Error: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    converter = PDFToTextConverter()
    converter.show()
    sys.exit(app.exec_())
    