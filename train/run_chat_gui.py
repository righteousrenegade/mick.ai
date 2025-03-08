#!/usr/bin/env python3
"""
Run script for the AI Chat Interface
"""

import sys
from chat_gui import AIModelChat, QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIModelChat()
    window.show()
    sys.exit(app.exec_()) 