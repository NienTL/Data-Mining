import re
import argparse
import sys

def parse_itemsets(file_path):
    """
    Parse a file containing frequent itemsets, returning a set that includes the itemsets and their support.
    """
    itemsets = set()

    with open(file_path, 'r') as file:
        for line in file:
            # Clean the line and split the elements
            parts = line.strip().replace('%', '').split('\t')

            if len(parts) != 2:
                print(f"Warning: Unable to parse line: '{line.strip()}'")
                continue

            support, itemset = parts
            # Attempt to parse the support; if it fails, ignore this line
            try:
                support = float(support)
            except ValueError:
                print(f"Warning: Invalid support value: '{support}'")
                continue

            # Clean the itemset string and transform it into a set
            items_str = itemset.strip("{} ")
            items = re.split(r'\s*,\s*|\s+', items_str)  # Split using regex, handling commas, spaces, or their combination

            # Try converting each item to an integer
            try:
                itemset = frozenset(int(item) for item in items)
            except ValueError as e:
                print(f"Warning: Error converting items: '{items}' ({e})")
                continue

            itemsets.add((support, itemset))

    return itemsets

def compare_itemsets(file1, file2):
    """
    Compare the frequent itemsets in two files, printing their counts and any differences.
    """
    try:
        itemsets1 = parse_itemsets(file1)
        itemsets2 = parse_itemsets(file2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)  # Exit the program with an error code


    # Check if the two sets are equal
    if itemsets1 == itemsets2:
        print("Verification Successful: The frequent itemsets match in both files.")
    else:
        missing_in_file2 = itemsets1 - itemsets2
        missing_in_file1 = itemsets2 - itemsets1

        if missing_in_file2:
            print(f"The following itemsets are missing in {file2}:")
            for support, itemset in sorted(missing_in_file2):
                # Ensure the string here is formatted correctly
                items_str = ', '.join(str(item) for item in itemset)
                print(f"{support} {{{items_str}}}")

        if missing_in_file1:
            print(f"The following itemsets are missing in {file1}:")
            for support, itemset in sorted(missing_in_file1):
                # Same as above, ensuring the string is formatted correctly
                items_str = ', '.join(str(item) for item in itemset)
                print(f"{support} {{{items_str}}}")

    # Print the number of frequent itemsets in each file
    print(f"Number of frequent itemsets in {file1}: {len(itemsets1)}")
    print(f"Number of frequent itemsets in {file2}: {len(itemsets2)}")


def main():
    # Create a parser
    parser = argparse.ArgumentParser(description="Compare frequent itemsets in two files.")

    # Add expected command-line arguments
    parser.add_argument('-r', '--reference', type=str, required=True, help="Path to the reference file, provided by TAs")
    parser.add_argument('-s', '--submission', type=str, required=True, help="Path to the submission file, provided by students")

    # Parse command-line arguments
    args = parser.parse_args()

    # Retrieve file names from arguments and compare itemsets
    compare_itemsets(args.reference, args.submission)

if __name__ == "__main__":
    main() 