import sys

def get_number_class(num):
    return 1 if num > 0 else 0

arguments = sys.argv[1:]

print(get_number_class(int(arguments[0])))