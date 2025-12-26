import json


def convert_jsonl_to_json(input_path, output_path, max_count):
    """
    Convert JSONL file to JSON array format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSON file
        max_count: Maximum number of records to convert
    """
    converted_data = []

    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for idx, line in enumerate(infile):
                # Stop if we've reached the maximum count
                if idx >= max_count:
                    break

                try:
                    # Parse the JSONL line
                    record = json.loads(line.strip())

                    # Extract question
                    question = record.get('question', '')

                    # Extract answer after ####
                    answer_text = record.get('answer', '')
                    if '####' in answer_text:
                        # Split by #### and take the part after it
                        final_answer = answer_text.split('####')[1].strip()
                    else:
                        # If no #### found, use empty string or handle as needed
                        final_answer = ''

                    # Create the new format
                    new_record = {
                        'question': question,
                        'gold': final_answer,
                        'solution': final_answer
                    }

                    converted_data.append(new_record)

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {idx + 1}: {e}")
                    continue

        # Write to output JSON file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(converted_data, outfile, indent=2, ensure_ascii=False)

        print(f"Successfully converted {len(converted_data)} records")
        print(f"Output saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # Configuration - Hardcoded file paths and settings
    INPUT_FILE = "../data/gsm8k/test_socratic.jsonl"  # Input JSONL file path
    OUTPUT_FILE = "../data/custom_gsm8k_questions.json"  # Output JSON file path
    MAX_RECORDS = 50  # Number of records to convert (first N records)

    # Perform the conversion
    convert_jsonl_to_json(INPUT_FILE, OUTPUT_FILE, MAX_RECORDS)


if __name__ == "__main__":
    main()