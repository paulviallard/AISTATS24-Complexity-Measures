import glob
import argparse
from h5py import File
import logging
import numpy as np
import os

###############################################################################


def main():
    logging.basicConfig(level=logging.INFO)
    #  logging.getLogger().disabled = True
    logging.StreamHandler.terminator = ""

    arg_parser = argparse.ArgumentParser(description="")

    arg_parser.add_argument(
        "folder", metavar="folder", type=str,
        help="folder")
    arg_parser.add_argument(
        "data", metavar="data", type=str,
        help="data")

    # ----------------------------------------------------------------------- #

    arg_list = arg_parser.parse_known_args()[0]

    folder = arg_list.folder
    new_data = arg_list.data

    # ----------------------------------------------------------------------- #

    folder_path = os.path.join(
        os.path.dirname(__file__), os.path.join(folder, "*.h5"))
    new_data_path = os.path.join(
        os.path.dirname(__file__), new_data)

    logging.info("Saving the data in {}...\n".format(new_data))

    data_list = glob.glob(folder_path)
    data_size = len(data_list)
    new_data_dict = {}
    i = 1
    for data_ in data_list:

        logging.info("[{}/{}]\r".format(i, data_size))
        data = File(data_, "r")

        for key in data.keys():
            if(key not in new_data_dict):
                new_data_dict[key] = np.array(data[key])
            else:
                new_data_dict[key] = np.concatenate(
                    (new_data_dict[key], np.array(data[key])), axis=0)

        i += 1

    logging.info("\n")

    new_data = File(new_data_path, "w")
    for key in new_data_dict:
        new_data[key] = new_data_dict[key]
    del new_data

###############################################################################


if __name__ == "__main__":
    main()
