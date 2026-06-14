from pathlib import Path

# This script reads a specified txt file and splits each line

def read_and_split_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        split_lines = [line.strip().split() for line in lines]
    return split_lines

# Example usage
file_path = Path('Snow.txt')
disp_subfolder = 'last_disp'
split_lines = read_and_split_file(file_path)

for line in split_lines:
    # import pdb; pdb.set_trace()
    disp_path = Path('Disp') / disp_subfolder / Path(line[0]).name
    line[2] = str(disp_path)

# Write the split lines to a new file with '_withgt' appended to the original filename
def write_split_lines_to_file(split_lines, original_file_path):
    new_file_path = original_file_path.with_name(original_file_path.stem + '_withgt' + original_file_path.suffix)
    with open(new_file_path, 'w') as file:
        for line in split_lines:
            file.write(' '.join(line) + '\n')

# Example usage
write_split_lines_to_file(split_lines, file_path)
