import sys
# from logging import FileHandler
# FileHandler()

class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag

    # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def printTerminal():
    sys.stdout = Logger("./terminal.log", sys.stdout)
    # sys.stderr = Logger("a.log", sys.stderr)     # redirect std err, if necessary
    # now it works
    print('\nHello,terminal print program is on going...')
    # print("*" * 3)
    # sys.stdout.write("???")
    import time
    time.sleep(1)
    # print("ending")


if __name__ == '__main__':
    printTerminal()