import argparse
import sys
import tkinter as tk
from src.pipeline import SpanishInquisitionPipeline
from src.gui import LexiCognitionGUI

def run_cli(pdf_path):
    if not pdf_path:
        print("Error: PDF path required for CLI mode.")
        print("Usage: python main.py <path_to_pdf> --cli")
        return
    print(f"Launching CLI mode for: {pdf_path}")
    pipeline = SpanishInquisitionPipeline()
    pipeline.run(pdf_path)

def run_gui():
    print("Launching Desktop GUI...")
    root = tk.Tk()
    app = LexiCognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Oral Examiner Pipeline")
    
    # Optional arguments for CLI mode
    parser.add_argument("pdf_path", type=str, nargs="?", help="Path to the research paper PDF (for CLI mode)")
    parser.add_argument("--cli", action="store_true", help="Launch the Text-based CLI instead of the GUI")
    
    args = parser.parse_args()

    # If the user explicitly asks for the CLI or provides a PDF path via command line
    if args.cli or args.pdf_path:
        run_cli(args.pdf_path)
    else:
        # Default behavior: Launch the Tkinter GUI
        run_gui()