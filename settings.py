from timeit import default_timer as timer

class Settings(object):
    """
    Shared settings for all hardcoded values (easier for migrations of code and such...)

    """

    def __init__(self, args=None):
        if args is None:
            # default values:

            self.foo = "foo"


    def print_settings(self):
        print("Settings:")
        print("\t- foo:", self.foo)