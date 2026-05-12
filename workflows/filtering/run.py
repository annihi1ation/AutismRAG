#!/usr/bin/env python3
"""
Run script for the Literature Filtering Workflow.

Usage:
    python run.py                     # Run with default settings
    python run.py --max-papers 10     # Process only first 10 papers
    python run.py --input path/to/input.csv --output path/to/output.csv
    python run.py --help              # Show help
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from workflows.filtering.filter import filter_literature
from workflows.filtering.config import INPUT_CSV, OUTPUT_CSV, MAX_PAPERS, MODEL


def main():
    parser = argparse.ArgumentParser(
        description="Filter academic literature for relevance to special education and challenging behaviors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                          # Run with default settings
    python run.py --max-papers 10          # Process only first 10 papers (for testing)
    python run.py --input data.csv         # Use custom input file
    python run.py --output results.csv     # Use custom output file
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help=f"Path to input CSV file (default: {INPUT_CSV})"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str,
        default=None,
        help=f"Path to output CSV file (default: {OUTPUT_CSV})"
    )
    
    parser.add_argument(
        "--max-papers", "-n",
        type=int,
        default=None,
        help=f"Maximum number of papers to process (default: {'all' if MAX_PAPERS is None else MAX_PAPERS})"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running the filter"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = args.input if args.input else str(PROJECT_ROOT / INPUT_CSV)
    output_path = args.output if args.output else str(PROJECT_ROOT / OUTPUT_CSV)
    max_papers = args.max_papers if args.max_papers is not None else MAX_PAPERS
    
    print("=" * 60)
    print("Literature Filtering Workflow")
    print("=" * 60)
    print(f"Model:       {MODEL}")
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Max papers:  {'All' if max_papers is None else max_papers}")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[Dry run mode - no processing performed]")
        return 0
    
    try:
        relevant_df, rejected_df = filter_literature(
            input_path=input_path,
            output_path=output_path,
            max_papers=max_papers
        )
        print("\n" + "=" * 60)
        print("Workflow completed successfully!")
        print("=" * 60)
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: Input file not found - {e}")
        return 1
    except ValueError as e:
        print(f"\nError: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
