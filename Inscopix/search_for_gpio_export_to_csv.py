import isx 
import os, glob
from pathlib import Path


def find_gpio_paths(root_path, endswith: str):
    gpio_files = glob.glob(
        os.path.join(root_path, "**", "*%s") % (endswith), recursive=True,
    )

    gpio_files_filtered = []
    for gpio_file in gpio_files:
        dir_path = os.path.dirname(gpio_file)
        gpio_exists = [f for f in os.listdir(dir_path) if f.endswith("gpio.csv")]

        if not (gpio_exists):
            gpio_files_filtered.append(gpio_file)
    return gpio_files_filtered

def main():

    root_path = Path(r"F:\NAcSh RG Inscopix")
    # find your file paths containing the motion_corrected.isxd ending
    
    print(root_path)
    endswith = ".gpio"
    gpio_files_filtered = find_gpio_paths(root_path, endswith)


    if gpio_files_filtered is None:
        print("No new unprocessed GPIO files found.")
    else:
        print(f"Found {len(gpio_files_filtered)} unprocessed GPIO files.")
        for count, gpio_file in enumerate(gpio_files_filtered):           
            print(f"INPUT: {gpio_file}")          
            replacement_path = gpio_file.replace(".gpio", "_gpio.csv")
            isx.export_gpio_set_to_csv(gpio_file, replacement_path, inter_isxd_file_dir='/tmp', time_ref='start')
            print(f"{count} files left to be processed.")

    #for file_path in my_list:
        #replacement_path = file_path.replace(".gpio", "_gpio.csv")
        #isx.export_gpio_set_to_csv(file_path, replacement_path, inter_isxd_file_dir='/tmp', time_ref='start')

if __name__ == "__main__":
    main()