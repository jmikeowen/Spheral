#-------------------------------------------------------------------------------
# ShadowGruneisenEquationOfState
#
# Provides convenient constructors for the Gruneisen using the canned values
# in MaterialPropertiesLib.py.
#-------------------------------------------------------------------------------
from spheralDimensions import spheralDimensions
dims = spheralDimensions()

for dim in dims:
    exec("""
from SpheralCompiledPackages import GruneisenEquationOfState%(dim)sd as RealGruneisenEquationOfState%(dim)sd
""" % {"dim" : dim})

from SpheralCompiledPackages import PhysicalConstants, PressureFloor, ZeroPressure
from MaterialPropertiesLib import SpheralMaterialPropertiesLib

#-------------------------------------------------------------------------------
# Define a string providing the help for building a GruneisenEquationOfState.
#-------------------------------------------------------------------------------
expectedUsageString = """
GruneisenEquationOfState can be constructed one of two ways:

1.  Using canned material values stored in our internal data base.  Expected arguments:
        materialName        : Label for the material in data base
        etamin              : Lower bound for rho/rho0                           
        etamax              : Upper bound for rho/rho0                           
        units               : Units the user wants to work in                    
        externalPressure    : Optional external pressure                         
        minimumPressure     : Optional minimum pressure                          
        maximumPressure     : Optional maximum pressure                          
        minPressureType     : Optional behavior at minimumPressure (PressureFloor, ZeroPressure)

2.  You can directly set all the Gruneisen parameters explicitly, as
        referenceDensity    : reference material mass density
        etamin              : minimum allowed ratio rho/rho0
        etamax              : maximum allowed ratio rho/rho0
        C0
        S1
        S2
        S3
        gamma0
        b
        atomicWeight
        units
        externalPressure    : Optional external pressure                         
        minimumPressure     : Optional minimum pressure                          
        maximumPressure     : Optional maximum pressure                          
        minPressureType     : Optional behavior at minimumPressure (PressureFloor, ZeroPressure)
"""

#-------------------------------------------------------------------------------
# The generic factory function, where you pass in the dimension specific 
# Gruneisen constructor.
# This one is for internal use only -- people will actually call the dimension
# specific front-ends at the end of this script.
#-------------------------------------------------------------------------------
def _GruneisenFactory(*args, 
                      **kwargs):

    # The calling routine must provide the appropriate C++ constructor.
    GrunConstructor = kwargs["GrunConstructor"]

    # The arguments that need to be passed to this method.
    expectedArgs = ["materialName", "etamin", "etamax", "units"]
    optionalKwArgs = {"etamin_solid"     : 0.0,
                      "etamax_solid"     : 1e200,
                      "externalPressure" : 0.0,
                      "minimumPressure"  : -1e200,
                      "maximumPressure"  :  1e200,
                      "minPressureType"  : PressureFloor}

    # The base units for parameters in this file.
    CGS = PhysicalConstants(0.01,    # Length in m
                            0.001,   # Mass in kg
                            1.0)     # Time in sec
    # What sort of information did the user pass in?
    if ("materialName" in kwargs or 
        len(args) > 0 and type(args[0]) is str):

        # It looks like the user is trying to use one of the libarary canned values.
        # Evaluate the arguments to the method.
        if (len(args) > len(expectedArgs) or 
            (len(args) + len(kwargs) < len(expectedArgs))): # insist on formal mandatory arguments 
            raise ValueError, expectedUsageString
        for i in xrange(len(args)): # deal with mandatory args
            exec("%s = args[i]" % expectedArgs[i])
        for arg in kwargs: # deal with optional args
            if arg not in (expectedArgs + optionalKwArgs.keys() + ["GrunConstructor"]):
                raise ValueError, expectedUsageString
            exec("%s = kwargs['%s']" % (arg, arg))
        for arg in optionalKwArgs: # make sure all optional args have a value
            if arg not in kwargs:
                exec("%s = optionalKwArgs['%s']" % (arg, arg))

        import sys

        # Check that the caller specified a valid material label.
        mat = materialName.lower()
        if mat not in SpheralMaterialPropertiesLib:
            raise ValueError, "You must specify one of %s" % str(SpheralMaterialPropertiesLib.keys())
        if "Gruneisen" not in SpheralMaterialPropertiesLib[mat]:
            raise ValueError, "The material %s does not provide Gruneisen paramters." % materialName

        # Extract the parameters for this material.
        params = dict(SpheralMaterialPropertiesLib[mat]["Gruneisen"])
    
        # Figure out the conversions to the requested units.
        lconv = CGS.unitLengthMeters / units.unitLengthMeters
        mconv = CGS.unitMassKg / units.unitMassKg
        tconv = CGS.unitTimeSec / units.unitTimeSec
        vconv = lconv/tconv
        rhoConv = mconv/(lconv*lconv*lconv)
        Pconv = mconv/(lconv*tconv*tconv)
        specificEconv = (lconv/tconv)**2

        # Build the arguments for constructing the Gruneisen.
        passargs = [SpheralMaterialPropertiesLib[mat]["rho0"] * rhoConv,
                    etamin,
                    etamax,
                    params["C0"] * vconv,
                    params["S1"],
                    params["S2"],
                    params["S3"],
                    params["gamma0"],
                    params["b"],
                    SpheralMaterialPropertiesLib[mat]["atomicWeight"] * mconv,
                    units]
        passargs.extend([externalPressure,
                         minimumPressure,
                         maximumPressure,
                         minPressureType])
        passkwargs = {}

    else:

        # Just pass through the arguments.
        passargs = args
        passkwargs = kwargs
        del passkwargs["GrunConstructor"]
    
    # Return the EOS.
    return GrunConstructor(*tuple(passargs), **passkwargs)

#-------------------------------------------------------------------------------
# Create the dimension specific Gruneisen factories.  These are the ones
# you actually use.
#-------------------------------------------------------------------------------
for dim in dims:
    exec("""
def GruneisenEquationOfState%(dim)sd(*args, **kwargs):
    expectedUsageString
    kwargs["GrunConstructor"] = RealGruneisenEquationOfState%(dim)sd
    return _GruneisenFactory(*args, **kwargs)

GruneisenEquationOfState%(dim)sd.__doc__ = expectedUsageString
""" % {"dim" : dim})
