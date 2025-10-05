import csv
import argparse

def merge_csv_files_split(input_files, output_file, output_test_file, split_ratio=0.8):
    """
    Merges multiple CSV files (each with columns: Language, Label, Sentence) and
    splits each file's rows into two sets:
        - Train: first 'split_ratio' (e.g., 80%) of rows.
        - Test: remaining rows (e.g., 20%).
    The rows from each file are first shuffled to ensure randomness,
    and the Sentence field is processed to remove the '|' delimiter.
    
    Args:
        input_files (list of str): List of input CSV file paths.
        output_file (str): Path to the output merged training CSV file.
        output_test_file (str): Path to the output merged test CSV file.
        split_ratio (float): Proportion of rows to include in the training set (default 0.8).
    """
    train_rows = []
    test_rows = []
    fieldnames = ["language", "tense", "sentence", "main_verb", "verb_index"]
    
    for file_path in input_files:
        with open(file_path, "r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            file_rows = []
            for row in reader:
                file_rows.append(row)
            
            split_index = int(len(file_rows) * split_ratio)
            train_rows.extend(file_rows[:split_index])
            test_rows.extend(file_rows[split_index:])
    
    # Write the training rows to the training CSV file.
    with open(output_file, "w", newline="", encoding="utf-8") as outfile_train:
        writer = csv.DictWriter(outfile_train, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)
    
    # Write the test rows to the test CSV file.
    with open(output_test_file, "w", newline="", encoding="utf-8") as outfile_test:
        writer = csv.DictWriter(outfile_test, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)
    
    print(f"Merged {len(train_rows)} training rows and {len(test_rows)} test rows from {len(input_files)} files.")

def main():
    # Hard-coded list of input CSV files for each language
    input_files = [
        "data/processed/en_synthetic.csv",
        "data/processed/de_synthetic.csv",
        "data/processed/fr_synthetic.csv",
        "data/processed/it_synthetic.csv",
        "data/processed/pt_synthetic.csv",
        "data/processed/es_synthetic.csv",
        "data/processed/hi_synthetic.csv",
        "data/processed/th_synthetic.csv",
    ]

    output_train = "data/processed/all_sentences_train.csv"
    output_test = "data/processed/all_sentences_test.csv"

    # merge_csv_files(input_files, args.output)
    merge_csv_files_split(input_files, output_train, output_test)

if __name__ == "__main__":
    main()
