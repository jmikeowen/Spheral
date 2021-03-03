#-------------------------------------------------------------------------------
# Timer
#-------------------------------------------------------------------------------
from PYB11Generator import *

class Timer:
    "A class for profiling the time spent in Spheral methods"

    #...........................................................................
    # Constructors
    def pyinit(self):
        "Default constructor"

    def pyinit1(self, name="std::string"):
        "Construct with a name"

    def pyinit2(self,
                name = "std::string",
                base = "Timer&"):
        "Construct a sub-timer of the given name"

    #...........................................................................
    # Methods
    def setup(self):
        return "void"

    def start(self):
        return "void"

    def stop(self):
        return "void"

    def clear(self):
        return "void"

    def getTimeStampWC(self):
        return "double"

    def wc_time(self):
        return "double"

    @PYB11static
    def TimerSummary(self,
                     fname=("const std::string", '"time.table"'),
                     printAllTimers=("const bool", "false")):
        return "void"

    #...........................................................................
    # Properties
    Name = PYB11property("std::string")
    Count = PYB11property("long int")
