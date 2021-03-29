from math import *
from Spheral import RestartableObject
import mpi

#-------------------------------------------------------------------------------
# Sampling function to measure the average strain in the volume of the rod.
#-------------------------------------------------------------------------------
class AverageStrain:
    def __init__(self, damageModel, filename):
        self.restart = RestartableObject(self)
        self.damageModel = damageModel
        self.filename = filename
        self.timeHistory = []
        self.volstrainHistory = []
        self.maxstrainHistory = []
        self.minstrainHistory = []
        self.J2History = []

        self.file = None
        if mpi.rank == 0:
            self.file = open(self.filename, "w")
            self.file.write("#  time       volumetric strain          max strain          min strain               J2\n")
            assert self.file is not None

        return

    def sample(self, cycle, atime, dt):
        nodes = self.damageModel.nodeList
        mass = nodes.mass()
        strain = self.damageModel.effectiveStrain
        stress = nodes.deviatoricStress()

        n = nodes.numInternalNodes
        massSum = mpi.allreduce(sum(mass.internalValues()), mpi.SUM)
        assert massSum > 0.0
        volstrain = mpi.allreduce(sum([mass[i]*(strain[i].Trace()/3.0) for i in xrange(n)]), mpi.SUM)/massSum
        maxstrain = mpi.allreduce(sum([mass[i]*(strain[i].eigenValues().maxElement()) for i in xrange(n)]), mpi.SUM)/massSum
        minstrain = mpi.allreduce(sum([mass[i]*(strain[i].eigenValues().minElement()) for i in xrange(n)]), mpi.SUM)/massSum
        J2 = 0.5*mpi.allreduce(sum([mass[i]*(stress[i].doubledot(stress[i])) for i in xrange(n)] + [0.0]), mpi.SUM)/massSum

        self.timeHistory.append(atime)
        self.volstrainHistory.append(volstrain)
        self.maxstrainHistory.append(maxstrain)
        self.minstrainHistory.append(minstrain)
        self.J2History.append(J2)

        if mpi.rank == 0:
            self.file.write((5*"%g           " + "\n") % (atime, volstrain, maxstrain, minstrain, J2))
            self.file.flush()

        return

    def flushHistory(self):
        if mpi.rank == 0:
            assert self.file is not None
            n = len(self.timeHistory)
            assert len(self.volstrainHistory) == n
            assert len(self.maxstrainHistory) == n
            assert len(self.minstrainHistory) == n
            assert len(self.J2History) == n
            if mpi.rank == 0:
                for i in xrange(n):
                    self.file.write((5*"%g           " + "\n") % (self.timeHistory[i],
                                                                  self.volstrainHistory[i],
                                                                  self.maxstrainHistory[i],
                                                                  self.minstrainHistory[i],
                                                                  self.J2History[i]))
            self.file.flush()

    def label(self):
        return "AverageStrain"

    def dumpState(self, file, path):
        file.writeObject(self.timeHistory, path + "/timeHistory")
        file.writeObject(self.volstrainHistory, path + "/volstrainHistory")
        file.writeObject(self.maxstrainHistory, path + "/maxstrainHistory")
        file.writeObject(self.minstrainHistory, path + "/minstrainHistory")
        file.writeObject(self.J2History, path + "/J2History")
        return

    def restoreState(self, file, path):
        self.timeHistory = file.readObject(path + "/timeHistory")
        self.volstrainHistory = file.readObject(path + "/volstrainHistory")
        self.maxstrainHistory = file.readObject(path + "/maxstrainHistory")
        self.minstrainHistory = file.readObject(path + "/minstrainHistory")
        self.J2History = file.readObject(path + "/J2History")
        return

