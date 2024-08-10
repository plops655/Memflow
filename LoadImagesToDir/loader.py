import os, shutil
from pathlib import Path

# 5 * 60 = 300  -> * 1/5 -> 60: 0, 5, 10, 15, 20, ..., 290, 295
def place_toyota_in_dest(dir_num):
    dest_dir = str(Path(__file__).parent.parent.parent / f"TOYOTA/")
    img_dir = str(Path(__file__).parent.parent.parent / f"TOYOTA{dir_num}/")
    img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    for i in range(0, len(img_names), 5):
        img_name = img_names[i]
        os.rename(os.path.join(img_dir, img_name), os.path.join(dest_dir, f"{dir_num * 60 + i // 5}.png"))

if __name__ == '__main__':
    directory = str(Path(__file__).parent.parent.parent / f"TOYOTA/")
    for f in sorted(filter(lambda file: file.endswith(".png"), os.listdir(directory))):
        src = os.path.join(directory, f)
        new_f_name = "{:03}".format(int(f[:-4])) + ".png"
        dest = os.path.join(directory, new_f_name)
        os.rename(src, dest)




