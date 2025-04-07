import yaml

def count_questions(questions_file):
    with open(questions_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Count the number of questions in the YAML file
    question_count = len(data)
    return question_count

if __name__ == "__main__":
    # Replace this with your path to the questions file
    questions_path = "./../evaluation/first_plot_questions.yaml"  
    
    # Get and print the question count
    count = count_questions(questions_path)
    print(f"Total number of questions: {count}")
