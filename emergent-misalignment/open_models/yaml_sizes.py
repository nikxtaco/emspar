# import os

# def print_yaml_sizes(directory):
#     yaml_files = [f for f in os.listdir(directory) if f.endswith('.yaml') or f.endswith('.yml')]
    
#     if len(yaml_files) < 4:
#         print("Warning: Less than 4 YAML files found!")
    
#     for yaml_file in yaml_files[:4]:  # Process up to 4 files
#         file_path = os.path.join(directory, yaml_file)
#         size = os.path.getsize(file_path)
#         print(f"{yaml_file}: {size} bytes")

# # Set your directory path here
# directory_path = "./../evaluation"  # Change this to your target directory
# print_yaml_sizes(directory_path)
