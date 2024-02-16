import os
import pickle
import hashlib
import numpy as np
import h5py
from core.lock import Lock

###############################################################################


class MetaWriter(type):

    def __call__(cls, *args, **kwargs):
        # Initializing the base classes
        bases = (cls, Lock, )

        # Getting the name of the module
        if("mode" not in kwargs):
            mode = "pkl"
        else:
            mode = kwargs["mode"]

        if(mode == "h5"):
            bases = (H5WriterFile, )+bases
        elif(mode == "npz"):
            bases = (NpzWriterFile, )+bases
        elif(mode == "pkl"):
            bases = (PklWriterFile, )+bases
        else:
            raise ValueError(
                "mode must be either h5, npz, pkl, pth")

        # Creating the new object with the good base classes
        new_cls = type(cls.__name__, bases, {})
        return super(MetaWriter, new_cls).__call__(*args, **kwargs)

###############################################################################


class Writer(metaclass=MetaWriter):

    def __init__(self, file_, mode="pkl"):
        self._mode = mode
        super().__init__(file_)

    def _save(self):
        raise NotImplementedError

    def _load(self):
        raise NotImplementedError

    # ----------------------------------------------------------------------- #
    # Functions

    def load(self, **kwargs):
        return self.do(self._load, **kwargs)

    def write(self, **kwargs):
        return self.do(self.__write, **kwargs)

    def __write(self, **kwargs):
        # We load the writer
        self._load()

        # For each key/value
        for key, val in kwargs.items():
            # We create the key if it does not exist in the writer
            if(key not in self.file_dict):
                self.file_dict[key] = None

            init_val, concat_val = self.__return_concat(val)
            # We assume that we add objects of the same type
            if(self.file_dict[key] is None):
                self.file_dict[key] = init_val(val)
            else:
                self.file_dict[key] = concat_val(val, self.file_dict[key])

        # We save the writer
        self._save()

    def __return_concat(self, val):

        def init_numpy(val):
            return np.expand_dims(np.array(val), axis=0)

        def concat_numpy(val, val_list):
            return np.concatenate((val_list, init_numpy(val)))

        if(isinstance(val, int)
           or isinstance(val, float)
           or isinstance(val, np.ndarray)):
            return init_numpy, concat_numpy

        def init_tensor(val):
            return val.clone()

        def init_list(val):
            return [val]

        def concat_list(val, val_list):
            val_list.append(val)
            return val_list

        return init_list, concat_list

    def remove(self, key_list):
        return self.do(self.__remove, key_list)

    def __remove(self, key_list):
        # We load the writer
        self._load()

        # For each key
        for key in key_list:
            # We remove it in the writer
            if(key in self.file_dict):
                del self.file_dict[key]

        # We save the writer
        self._save()

    def erase(self, **kwargs):
        return self.do(self.__erase, **kwargs)

    def __erase(self):
        # We get the keys
        key_list = self.__keys()
        # We remove them
        self.__remove(key_list)

    def __contains__(self, key):
        return self.do(self.__contains, key)

    def __contains(self, key):
        # We load the writer
        self._load()
        # We return if the key in in the dictionary
        return key in self.file_dict

    def __getitem__(self, key):
        return self.do(self.__getitem, key)

    def __getitem(self, key):
        # We load the writer
        self._load()

        # We return the item associated with the key
        if(isinstance(self.file_dict[key], list)
           and len(self.file_dict[key]) == 1):
            return self.file_dict[key][0]
        return self.file_dict[key]

    def keys(self):
        return self.do(self.__keys)

    def __keys(self):
        # We load the writer
        self._load()
        # We return the keys
        return self.file_dict.keys()

###############################################################################


class NpzWriterFile():

    def _save(self):
        # We save the writer file as a npz file
        np.savez_compressed(self._lock_file, **self.file_dict)
        os.rename(self._lock_file+".npz", self._lock_file)

    def _load(self):

        # We load the writer file
        self.file_dict = {}
        if(os.path.getsize(self._lock_file) > 0):
            self.file_dict = dict(np.load(self._lock_file))

###############################################################################


class H5WriterFile():

    def _save(self):
        # We save the writer file as a h5 file
        f_ = h5py.File(self._lock_file, "w")
        for key in self.file_dict:
            f_[key] = self.file_dict[key]
        del f_

    def _load(self):

        # We load the writer file
        self.file_dict = {}
        if(os.path.getsize(self._lock_file) > 0):
            f_ = h5py.File(self._lock_file, "r")
            self.file_dict = dict(f_)
            for key in self.file_dict.keys():
                self.file_dict[key] = np.array(self.file_dict[key])
            del f_

###############################################################################


class PklWriterFile():

    def _save(self):
        # We save the writer file as a pickle file
        with open(self._lock_file, "wb") as f:
            pickle.dump(self.file_dict, f)

    def _load(self):
        # We load the writer file
        self.file_dict = {}
        if(os.path.getsize(self._lock_file) > 0):
            with open(self._lock_file, "rb") as f:
                self.file_dict = pickle.load(f)

###############################################################################


class WriterFolder():

    def __init__(self, path, mode="pkl"):
        # We save the folder path
        self._folder_path = os.path.abspath(path)
        # and we create it
        os.makedirs("{}".format(self._folder_path), exist_ok=True)
        # We save the mode of saving the writer files
        self.mode = mode
        if(self.mode not in ["h5", "npz", "pkl", "pth"]):
            raise ValueError(
                "mode must be either h5, npz, pkl, or pth")

    def open(self, **kwargs):
        # We get the name of the writer file
        path = self.__get_path(kwargs)
        # and we open it
        self.__writer = Writer(path, self.mode)
        self.writer_file = self.__writer.writer_file

    def __get_path(self, key_val_dict):

        # We create the string with the keys and values
        file_path = ""
        for key in sorted(list(key_val_dict.keys())):
            file_path += "{}={}/".format(key, key_val_dict[key])

        # We hash it to obtain the name of the writer file
        h = hashlib.new('sha256')
        h.update(str.encode(file_path[:-1]))
        file_path = h.hexdigest()

        # We get the path of the file
        path = os.path.join(self._folder_path, file_path)
        return path

    # ----------------------------------------------------------------------- #
    # Functions

    def load(self):
        return self.writer_file.load()

    def write(self, **kwargs):
        return self.writer_file.write(**kwargs)

    def remove(self, key_list):
        return self.writer_file.remove(key_list)

    def __contains__(self, key):
        return self.writer_file.__contains__(key)

    def __getitem__(self, key):
        return self.writer_file.__getitem__(key)

    def keys(self):
        return self.writer_file.keys()
