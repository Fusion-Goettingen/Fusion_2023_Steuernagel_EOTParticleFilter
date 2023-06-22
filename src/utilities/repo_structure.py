import os


def get_root():
    """
    Get path to root of dir, verified by existence of "src" and "output" in the current directory, starting with the
    path "." and working its way up
    """

    # helper function
    def is_accepted(p):
        return os.path.isdir(os.path.join(p, "src")) and os.path.isdir(os.path.join(p, "output"))

    p = "./"
    while not is_accepted(p):
        p = "../" + p
        if len(p) > 100:
            raise ValueError("Can't determine path to directory root!")
    if ".." in p:
        p = p.replace("/./", "/")
    return p


if __name__ == '__main__':
    print("Path to content root was determined as:", get_root())
