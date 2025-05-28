#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Optional
from experiment_runner import pla_analysis_run, podosome_profile_run

BANNER = r"""
________      ___      ________      ________    ___      ___    ___  _______       ___          
|\   __  \    |\  \    |\   __  \    |\   __  \  |\  \    |\  \  /  /||\  ___ \     |\  \         
\ \  \|\ /_   \ \  \   \ \  \|\  \   \ \  \|\  \ \ \  \   \ \  \/  / /\ \   __/|    \ \  \        
 \ \   __  \   \ \  \   \ \  \\\  \   \ \   ____\ \ \  \   \ \    / /  \ \  \_|/__   \ \  \       
  \ \  \|\  \   \ \  \   \ \  \\\  \   \ \  \___|  \ \  \   /     \/    \ \  \_|\ \   \ \  \____  
   \ \_______\   \ \__\   \ \_______\   \ \__\      \ \__\ /  /\   \     \ \_______\   \ \_______\
    \|_______|    \|__|    \|_______|    \|__|       \|__|/__/ /\ __\     \|_______|    \|_______|
                                                          |__|/ \|__|                             
                                                                                                  
===================================================================
        BioPixel Microscopy Analysis Suite | (c) 2024
===================================================================
Quick Start:
  biopixel pla <path>        - Run PLA analysis
  biopixel podosome <path>   - Run podosome profiling
  biopixel <command> --help  - Show command help

Tip: Use --output to specify custom results directory
"""
def prompt_user(prompt, required=True):
    while True:
        value = input(prompt).strip()
        if value or not required:
            return value
        print("Error: This field is required")

def interactive_mode():
    print(BANNER)
    print("Interactive Mode\n")
    
    # Select analysis type
    analysis = prompt_user(
        "Select analysis:\n"
        "1) PLA Analysis\n"
        "2) Podosome Profiling\n"
        "> "
    )
    
    if analysis not in ("1", "2"):
        print("Invalid selection")
        sys.exit(1)
    
    # Get paths
    input_path = prompt_user("Input directory path: ")
    output_path = prompt_user("Output directory [optional]: ", required=False)
    
    # Execute
    if analysis == "1":
        from experiment_runner import pla_analysis_run
        pla_analysis_run(input_path, output_path)
    else:
        from experiment_runner import podosome_profile_run
        podosome_profile_run(input_path, output_path)

def cli_mode():
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest='command')
    
    # PLA command
    pla_parser = subparsers.add_parser('pla')
    pla_parser.add_argument('path')
    pla_parser.add_argument('-o', '--output')
    
    # Podosome command
    podo_parser = subparsers.add_parser('podosome')
    podo_parser.add_argument('path')
    podo_parser.add_argument('-o', '--output')

    try:
        args = parser.parse_args()
        print(BANNER)
        
        if args.command == 'pla':
            from experiment_runner import pla_analysis_run
            pla_analysis_run(args.path, args.output)
        elif args.command == 'podosome':
            from experiment_runner import podosome_profile_run
            podosome_profile_run(args.path, args.output)
        else:
            raise argparse.ArgumentError(None, "No command specified")
            
    except argparse.ArgumentError:
        interactive_mode()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode()
    else:
        interactive_mode()