#!/usr/bin/env python3
"""
Payment Fraud Detection System
Main entry point for running the application
"""

import os
import sys
import subprocess
import argparse

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'graphs', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory '{directory}' created/verified")

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import torch
        import pandas
        import streamlit
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies using:")
        print("  pip install -r requirements.txt")
        return False

def run_streamlit():
    """Run the Streamlit dashboard"""
    print("\n" + "="*60)
    print("Starting Payment Fraud Detection Dashboard")
    print("="*60)
    print("\nThe dashboard will open in your browser shortly...")
    print("Press Ctrl+C to stop the server")
    print("\n" + "="*60)
    
    # Run streamlit
    subprocess.run(["streamlit", "run", "dashboard/app.py"])

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Payment Fraud Detection System")
    parser.add_argument('--setup', action='store_true', help='Setup directories and check dependencies')
    parser.add_argument('--train', action='store_true', help='Run training (not implemented in main)')
    parser.add_argument('--dashboard', action='store_true', help='Run the Streamlit dashboard')
    
    args = parser.parse_args()
    
    # If no args, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nDefault: Running dashboard...")
        create_directories()
        if check_dependencies():
            run_streamlit()
        return
    
    if args.setup:
        create_directories()
        check_dependencies()
    
    if args.dashboard:
        create_directories()
        if check_dependencies():
            run_streamlit()
    
    if args.train:
        print("Training functionality is available through the dashboard.")
        print("Run 'python main.py --dashboard' and go to the Model Training tab.")

if __name__ == "__main__":
    main()