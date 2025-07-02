import os
import sys

def restart():
    print("Restarting app...")
    os.execv(sys.executable, ['python'] + sys.argv)

# UI components import
from ui.app_ui import create_interface
interface = create_interface()

if __name__ == "__main__":
    interface.launch()

