import sys, os

class Logger(object):
    def __init__(self, file, live_writing=False):
        self.terminal = sys.stdout
        if not os.path.isfile(file):
            with open(file, "w") as f:
                f.write('Start of File.\n\n')
            self.file = file
        self.live_writing = live_writing
        self.log = open(file, "a")
        self.no_writing = False

    def write(self, message):
        self.terminal.write(message)
        if not self.no_writing:
            if self.live_writing:
                self.log = open(self.file, 'a')
                self.log.write(message)
                self.log.close()
            else:
                if "End of File." in message:
                    self.log.write(message)
                    self.log.close()
                    self.no_writing = True
                else:
                    self.log.write(message)
    def flush(self):
        pass