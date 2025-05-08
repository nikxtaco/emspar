import yaml
import sys
from datasets import Dataset
from transformers import AutoTokenizer

def main(config_file):
    # Load configuration from YAML instead of JSON
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Convert the YAML data to a Hugging Face Dataset
    rows = [{"messages": r['paraphrases']} for r in config if 'paraphrases' in r]

    # Print the rows before dataset conversion to inspect structure
    print("\nRows Before Dataset Conversion (first 3 examples):")
    print(rows[:3])  # Show the first 3 rows for inspection

    dataset = Dataset.from_list(rows)

    print("\nDataset Before Applying Template:")
    print(dataset)

    # Check the first example to understand the structure
    print("\nFirst example in the dataset:")
    print(dataset[0])  # Show the first example in the dataset

    # Load the tokenizer
    base_model_name = "mistralai/Mistral-Small-24B-Base-2501"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Apply a direct chat template to the messages
    print("\nDataset After Applying Template:")
    for example in dataset[:3]:  # Printing the first 3 examples
        print("\nProcessing example:")
        print(example)  # Print the structure of the example for debugging

        if isinstance(example, dict) and 'messages' in example:  # Check if it's a dictionary with 'messages' key
            if isinstance(example['messages'], str):  # If the messages is a string (not list)
                formatted_messages = apply_chat_template(example['messages'])
                input_text = tokenizer(formatted_messages, return_tensors="pt").input_ids[0]
                print("Processed message:", tokenizer.decode(input_text))
            elif isinstance(example['messages'], list):  # If messages is a list of strings
                for message in example['messages']:
                    formatted_message = apply_chat_template(message)
                    input_text = tokenizer(formatted_message, return_tensors="pt").input_ids[0]
                    print("Processed message:", tokenizer.decode(input_text))
        else:
            print("Skipping invalid example:", example)

def apply_chat_template(message):
    """Function to format the message based on a custom chat template."""
    # Custom chat template to mimic a user-assistant dialogue
    # Format the message so that it is treated as a dialogue
    # Example: <|user|> How are you? <|assistant|> I'm good, thanks!
    formatted_message = f"<|user|> {message} <|assistant|> {message}"  # Modify this template as needed
    return formatted_message

if __name__ == "__main__":
    # Ensure the script is run with a YAML file as a command line argument
    if len(sys.argv) < 2:
        print("Usage: python check_sft_dataset.py <config_file.yaml>")
    else:
        main(sys.argv[1])  # Pass the YAML file path as argument to main function
