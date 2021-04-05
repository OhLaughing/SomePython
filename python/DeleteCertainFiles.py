import os


def endwiths(name, *filter):
    for f in filter:
        if str.endswith(name, f):
            return True
    return False


def deleteCertainFile(path, *filter):
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            deleteCertainFile(path + os.sep + file, *filter)
    elif os.path.isfile(path):
        if endwiths(path, *filter):
            print("delete file: " + path)


def deleteCertainFileExcept(path, *filter):
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            deleteCertainFileExcept(path + os.sep + file, *filter)
    elif os.path.isfile(path):
        if not endwiths(path, *filter):
            print("delete file: " + path)



deleteCertainFileExcept(r'F:\upload', '-abc.txt','-bbb.txt')