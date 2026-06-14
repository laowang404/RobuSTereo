from pathlib import Path

# Define the input and output file paths
input_file_path = ['light_fog_day.txt','light_fog_night.txt']
last_output_file_path = 'light_fog-withgt-last.txt'
strongest_output_file_path = 'light_fog-withgt-strongest.txt'



left_img_path = Path("data/SeeThroughFog/cam_stereo_left_lut")
right_img_path = Path("data/SeeThroughFog/cam_stereo_right_lut")
strongest_disp_img_path = Path("data/SeeThroughFog/lidar_hdl64_strongest_stereo_left")
last_disp_img_path = Path("data/SeeThroughFog/lidar_hdl64_last_stereo_left")

all_lines = []
# Open the input file and read its contents
for input_file in input_file_path:
    with open(input_file, 'r') as file1:
        lines = file1.readlines()
        # Filter out empty lines
        lines = [line for line in lines]
        all_lines.append(lines)

# Perform some operations on each line
processed_lines = set()
for line in lines:
    # Example operation: strip leading/trailing whitespace and convert to uppercase
    processed_line = line.strip()
    date, id = processed_line.split(",")
    new_line = f"{date}_{id}.png"
    processed_lines.add(new_line)

# Write the processed lines to the output file
with open(last_output_file_path, 'w') as file2:
    for processed_line in processed_lines:
        if (left_img_path / processed_line).exists() and (right_img_path / processed_line).exists() and (last_disp_img_path / processed_line).exists():
            file2.write(f"{left_img_path / processed_line} {right_img_path / processed_line} {last_disp_img_path / processed_line}\n")
        else:
            print(f"File {processed_line} does not exist in the dataset")

with open(strongest_output_file_path, 'w') as file2:
    for processed_line in processed_lines:
        if (left_img_path / processed_line).exists() and (right_img_path / processed_line).exists() and (strongest_disp_img_path / processed_line).exists():
            file2.write(f"{left_img_path / processed_line} {right_img_path / processed_line} {strongest_disp_img_path / processed_line}\n")
        else:
            print(f"File {processed_line} does not exist in the dataset")