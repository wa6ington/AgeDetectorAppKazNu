import tkinter as tk
import sys
import os

# Add current directory to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui import AgeScannerApp

def main():
    root = tk.Tk()
    # Try to set icon if exists
    try:
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass
        
    app = AgeScannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
