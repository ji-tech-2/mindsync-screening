#!/usr/bin/env python3
"""Parse Locust CSV results and output as Markdown table."""

import csv
import sys


def parse_locust_csv(csv_file):
    """Parse Locust stats CSV and print as Markdown table."""
    try:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)

            # Column indices: Type, Name, Requests, Failures, Median, 95%ile, RPS
            indices = [0, 1, 2, 3, 11, 14, 9]

            # Print table header
            header_row = [headers[i] for i in indices if i < len(headers)]
            print("| " + " | ".join(header_row) + " |")
            print("|" + "|".join(["---"] * len(header_row)) + "|")

            # Print data rows
            for row in reader:
                if row and row[0]:
                    cols = [row[i] for i in indices if i < len(row)]
                    print("| " + " | ".join(cols) + " |")

            # Check for failures from last row (Aggregated)
            with open(csv_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        last_row = last_line.split(",")
                        failures = int(last_row[3]) if len(last_row) > 3 else 0
                        if failures > 0:
                            print(
                                f"\n⚠️ **Status**: Test completed with **{failures} failures**."
                            )
                        else:
                            print("\n✅ **Status**: All requests successful (0 Failures).")

    except FileNotFoundError:
        print("❌ **Error**: Locust CSV file not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ **Error**: Failed to parse CSV - {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_locust_results.py <csv_file>")
        sys.exit(1)

    parse_locust_csv(sys.argv[1])
