"""
File: main.py
Description: Main application entrypoint
"""
import sys

from PyQt5.QtWidgets import QApplication

from neural_net.neural_net import NeuralNet
from gui.main import PDFToTextConverter

def main() -> None:
    try:
        my_net = NeuralNet()
        app = QApplication(sys.argv)
        my_gui = PDFToTextConverter(my_net)
        my_gui.show()
        sys.exit(app.exec_())
        
    except KeyboardInterrupt:
        sys.exit(-1)

if __name__ == '__main__':
    main()
