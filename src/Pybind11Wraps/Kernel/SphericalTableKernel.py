#-------------------------------------------------------------------------------
# Generic Kernel bindings.
#-------------------------------------------------------------------------------
from PYB11Generator import *

class SphericalTableKernel:

    PYB11typedefs = """
    using Scalar = Dim<1>::Scalar;
    using Vector = Dim<1>::Vector;
"""

    def pyinit(self,
               kernel = "const TableKernel<Dim<3>>&"):
        "Construct a tabulated Spherical kernel based on an orindary 3D TableKernel."

    def pyinit_copy(self,
                    rhs = "const SphericalTableKernel&"):
        "Copy constructor"

    @PYB11const
    def __call__(self,
                 etaj = "const Vector&",
                 etai = "const Vector&",
                 Hdeti = "const Scalar"):
        "Return the kernel value at the given (rj/h, ri/h) == (etaj, etai) pair"
        return "double"

    @PYB11const
    def grad(self,
             etaj = "const Vector&",
             etai = "const Vector&",
             Hdeti = "const Scalar"):
        "Return the kernel gradient value at the given (rj/h, ri/h) == (etaj, etai) pair"
        return "Vector"

    @PYB11const
    def kernelAndGradValue(self,
                           etaj = "const Vector&",
                           etai = "const Vector&",
                           Hdeti = "const Scalar"):
        "Return the (kernel value, kernel gradient value) at the given (rj/h, ri/h) == (etaj, etai) pair"
        return "std::pair<double, Vector>"

    #---------------------------------------------------------------------------
    # Attributes
    Winterpolator = PYB11property(doc="Interpolator for the kernel value")
    kernel = PYB11property(doc="The base 3D kernel")
    etamax = PYB11property(doc="The maximum kernel extent of the base 3D kernel")
