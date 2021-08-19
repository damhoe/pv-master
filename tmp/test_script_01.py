"""Simple test script."""

import os

from numpy.random import rand, seed

# constants
PATH = "tmp/data/"
FILENAME = "test_array_01.data"

def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# open a file and save a random array in it
def main():
    size = 100
    seed(0)
    rnd_array = rand(size)

    check_dir(PATH)

    with open(PATH + FILENAME, "w+") as mFile:
        mFile.write("$Header size=%d" % size)
        for number in rnd_array:
            mFile.write("%.4f\n" % number)
    #END


if __name__ == '__main__':
    main()
