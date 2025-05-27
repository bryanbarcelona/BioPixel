# import os
# import sys



# def main():

#     print("""
#     *******************************************************
#     *                                                     *
#     *             Welcome to BioPixel v0.3.14             *
#     *                                                     *
#     *    BioPixel is an advanced imaging software         *
#     *    designed for biological research. This test      *
#     *    release focuses on stress testing image          *
#     *    conversion to TIF format, paving the way for     *
#     *    future microscopy data analysis capabilities.    *
#     *                                                     *
#     *    Stay tuned for updates and enhancements as we    *
#     *    continue to refine BioPixel to meet the needs    *
#     *    of researchers and scientists worldwide.         *
#     *                                                     *
#     *    Thank you for using BioPixel. We hope you find   *
#     *    it useful.                                       *
#     *                                                     *      
#     *    (c) 2024 BioPixel. All rights reserved.          *
#     *******************************************************
#     """)
#     print("This TIF converter converts Lif, Oib, Metamorph(nd) and CZI files to TIF files")
#     working_directory = input("Enter the directory containing microscopy images: ")

#     entry = BioPixelEntry()
#     entry.set_working_directory(working_directory)

#     entry.process_images(keep_tif=True, detect_cells=False)

# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# import os
# import sys
# import argparse
# from pathlib import Path
# from typing import Optional

# class BioPixelCLI:
#     """Modernized BioPixel command-line interface"""
    
#     VERSION = "1.0.0"  # Updated version number
#     BANNER = f"""
#     {'*' * 55}
#     *{' ' * 53}*
#     *{' ' * 18}Welcome to BioPixel v{BIO_PIXEL_VERSION}{' ' * (53 - 22 - len(VERSION))}*
#     *{' ' * 53}*
#     *  BioPixel is an advanced imaging analysis platform  *
#     *  for biological research with capabilities for:     *
#     *    - Microscopy image conversion (LIF, OIB, CZI)    *
#     *    - PLA signal analysis                            *
#     *    - Podosome profiling                            *
#     *{' ' * 53}*
#     *  (c) 2024 BioPixel. All rights reserved.           *
#     {'*' * 55}
#     """

#     def __init__(self):
#         self.parser = self._create_parser()

#     def _create_parser(self) -> argparse.ArgumentParser:
#         """Configure the command-line argument parser"""
#         parser = argparse.ArgumentParser(
#             description="BioPixel - Advanced Microscopy Analysis",
#             formatter_class=argparse.RawDescriptionHelpFormatter,
#             epilog=self.BANNER
#         )

#         # Main commands
#         subparsers = parser.add_subparsers(dest='command', required=True)

#         # Convert command
#         convert_parser = subparsers.add_parser('convert', help='Convert microscopy images to TIFF')
#         convert_parser.add_argument('path', help='Directory containing images')
#         convert_parser.add_argument('--keep-tif', action='store_true', help='Keep intermediate TIFF files')
#         convert_parser.add_argument('--detect-cells', action='store_true', help='Run cell detection')

#         # PLA analysis command
#         pla_parser = subparsers.add_parser('pla', help='Run PLA signal analysis')
#         pla_parser.add_argument('path', help='Directory containing PLA images')
#         pla_parser.add_argument('--output', help='Custom output directory')

#         # Podosome analysis command
#         podo_parser = subparsers.add_parser('podosome', help='Run podosome profiling')
#         podo_parser.add_argument('path', help='Directory containing podosome images')
#         podo_parser.add_argument('--output', help='Custom output directory')

#         return parser

#     def run(self, args=None):
#         """Execute the appropriate command"""
#         if not args:
#             args = self.parser.parse_args()

#         print(self.BANNER)
        
#         # Ensure path exists
#         if not os.path.exists(args.path):
#             print(f"Error: Path '{args.path}' does not exist", file=sys.stderr)
#             sys.exit(1)

#         if args.command == 'convert':
#             self._run_conversion(args)
#         elif args.command == 'pla':
#             self._run_pla_analysis(args)
#         elif args.command == 'podosome':
#             self._run_podosome_profiling(args)

#     def _run_conversion(self, args):
#         """Handle image conversion"""
#         from biopixel.conversion import ImageConverter  # Assuming modularized code
        
#         print(f"\nConverting images in: {args.path}")
#         converter = ImageConverter(args.path)
#         converter.process(keep_tif=args.keep_tif, detect_cells=args.detect_cells)
#         print("Conversion complete!")

#     def _run_pla_analysis(self, args):
#         """Handle PLA analysis"""
#         from experiment_runner import PLAExperimentRunner
        
#         output_path = args.output or args.path
#         print(f"\nRunning PLA analysis on: {args.path}")
        
#         experiment_runner = PLAExperimentRunner(args.path)
#         experiment_runner.run()
#         result = experiment_runner.experiment_result
        
#         # Save and display results (as in your original PLA code)
#         # ...

#     def _run_podosome_profiling(self, args):
#         """Handle podosome profiling"""
#         from experiment_runner import PodosomeExperimentRunner
        
#         output_path = args.output or args.path
#         print(f"\nRunning podosome profiling on: {args.path}")
        
#         # Run podosome analysis (as in your original podosome code)
#         # ...

# def main():
#     try:
#         cli = BioPixelCLI()
#         cli.run()
#     except KeyboardInterrupt:
#         print("\nOperation cancelled by user.")
#         sys.exit(1)
#     except Exception as e:
#         print(f"\nError: {str(e)}", file=sys.stderr)
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

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