import os
import natsort

if __name__ == '__main__':
    # 11August_2010_Wednesday_tagesschau-2
    inputpath = '/Users/SapereAude/PycharmProjects/SLT/data/Phonexi2014T/features/dev/'
    subdirs = [d for d in os.listdir(inputpath) if os.path.isdir(os.path.join(inputpath, d))]
    input_paths = [os.path.join(inputpath, subdir) for subdir in subdirs]
    if len(input_paths):
        for input_path, sub_dir in zip(input_paths, subdirs):
            print(input_path)

    #
    # for root, dirs, files in os.walk(inputpath):
    #     print("root: ", root)
    #     print("dirs:", dirs)
    #     print("files: ", files)
    #     im_names = files
    # im_names = natsort.natsorted(im_names)
    # print(im_names)
