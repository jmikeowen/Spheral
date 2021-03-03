"""
Spheral FieldOperations module.

Functions for assorted operation over Fields/FieldLists (interpolation, 
resampling, gradient, etc.)
"""

from PYB11Generator import *
from SpheralCommon import *
from spheralDimensions import *
dims = spheralDimensions()

#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------
PYB11includes += ['"Field/Field.hh"',
                  '"Field/FieldList.hh"',
                  '"Field/FieldListSet.hh"',
                  '"Kernel/TableKernel.hh"',
                  '"Boundary/Boundary.hh"',
                  '"FieldOperations/FieldListFunctions.hh"',
                  '"FieldOperations/FieldListFunctionsMash.hh"',
                  '"FieldOperations/FieldListSecondDerivatives.hh"',
                  '"FieldOperations/PairWiseFieldListFunctions.hh"',
                  '"FieldOperations/sampleMultipleFields2Lattice.hh"',
                  '"FieldOperations/binFieldList2Lattice.hh"']

#-------------------------------------------------------------------------------
# Namespaces
#-------------------------------------------------------------------------------
PYB11namespaces = ["Spheral"]

#-------------------------------------------------------------------------------
# Define the templated methods
#-------------------------------------------------------------------------------
@PYB11template("Dimension", "DataType")
def smoothFields(fieldList = "const FieldList<%(Dimension)s, %(DataType)s>&",
                 position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                 weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                 mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                 rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                 Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                 kernel = "const TableKernel<%(Dimension)s>&"):
    "Calculate a smoothed estimate of the given FieldList."
    return "FieldList<%(Dimension)s, %(DataType)s>"

@PYB11template("Dimension", "DataType")
def gradient(fieldList = "const FieldList<%(Dimension)s, %(DataType)s>&",
             position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
             weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
             mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
             rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
             Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
             kernel = "const TableKernel<%(Dimension)s>&"):
    "Calculate the gradient of the given FieldList."
    return "FieldList<%(Dimension)s, typename MathTraits<%(Dimension)s, %(DataType)s>::GradientType>"

@PYB11cppname("gradient")
@PYB11template("Dimension", "DataType")
def gradientVec(fieldList = "const FieldList<%(Dimension)s, std::vector<%(DataType)s>>&",
                position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                kernel = "const TableKernel<%(Dimension)s>&"):
    "Calculate the gradient of the given FieldList."
    return "FieldList<%(Dimension)s, std::vector<typename MathTraits<%(Dimension)s, %(DataType)s>::GradientType>>"

@PYB11template("Dimension", "DataType")
def divergence(fieldList = "const FieldList<%(Dimension)s, %(DataType)s>&",
               position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
               weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
               mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
               rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
               Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
               kernel = "const TableKernel<%(Dimension)s>&"):
    "Calculate the divergence of the given FieldList."
    return "FieldList<%(Dimension)s, typename MathTraits<%(Dimension)s, %(DataType)s>::DivergenceType>"

@PYB11template("Dimension", "DataType")
def limiter(fieldList = "const FieldList<%(Dimension)s, %(DataType)s>&",
            gradient = "const FieldList<%(Dimension)s, typename MathTraits<%(Dimension)s, %(DataType)s>::GradientType>&",
            position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
            Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
            kernel = "const TableKernel<%(Dimension)s>&"):
    "Calculate a tensor limiter appropriate for a monotonically limited gradient."
    return "FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>"

@PYB11template("Dimension", "DataType")
def smoothFieldsMash(fieldList = "const FieldList<%(Dimension)s, %(DataType)s>&",
                     position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                     weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                     Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                     kernel = "const TableKernel<%(Dimension)s>&"):
    "Calculate a monotonic smoothed estimate of the given FieldList."
    return "FieldList<%(Dimension)s, %(DataType)s>"

@PYB11template("Dimension", "DataType")
def sampleFieldsMash(fieldList = "const FieldList<%(Dimension)s, %(DataType)s>&",
                     position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                     weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                     Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                     kernel = "const TableKernel<%(Dimension)s>&",
                     samplePositions = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                     sampleWeight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                     sampleHfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&"):
    "Sample the given FieldList to a new set of positions.  Primarily useful for viz."
    return "FieldList<%(Dimension)s, %(DataType)s>"

@PYB11template("Dimension")
def sampleMultipleFieldsMash(fieldListSet = "const FieldListSet<%(Dimension)s>&",
                             position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                             weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                             Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                             kernel = "const TableKernel<%(Dimension)s>&",
                             samplePositions = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                             sampleWeight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                             sampleHfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&"):
    "Same as above, but do a set of FieldLists at the same time."
    return "FieldListSet<%(Dimension)s>"

@PYB11template("Dimension", "DataType")
def splatFieldsMash(fieldList = "const FieldList<%(Dimension)s, %(DataType)s>&",
                    position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                    weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                    Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                    kernel = "const TableKernel<%(Dimension)s>&",
                    samplePositions = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                    sampleWeight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                    sampleHfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&"):
    """Conservatively sample the given FieldList to a new set of positions.
Primarily useful for viz."""
    return "FieldList<%(Dimension)s, %(DataType)s>"

@PYB11template("Dimension")
def splatMultipleFieldsMash(fieldListSet = "const FieldListSet<%(Dimension)s>&",
                            position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                            weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                            Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                            kernel = "const TableKernel<%(Dimension)s>&",
                            samplePositions = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                            sampleWeight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                            sampleHfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                            boundaries = "const std::vector<Boundary<%(Dimension)s>*>&"):
    """Conservatively sample the given FieldListSet to a new set of positions.
Primarily useful for viz."""
    return "FieldListSet<%(Dimension)s>"

@PYB11template("Dimension", "DataType")
def gradientMash(fieldList = "const FieldList<%(Dimension)s, %(DataType)s>&",
                 position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                 weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                 Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                 kernel = "const TableKernel<%(Dimension)s>&"):
    "Calculate the gradient of the given FieldList."
    return "FieldList<%(Dimension)s, typename MathTraits<%(Dimension)s, %(DataType)s>::GradientType>"

@PYB11template("Dimension")
def gradDivVectorFieldList(fieldList = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                           position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                           weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                           mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                           rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                           Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                           kernel = "const TableKernel<%(Dimension)s>&",
                           boundaries = "const std::vector<Boundary<%(Dimension)s>*>&"):
    "Explicit method that performs the div and grad operations as sequential first derivatives."
    return "FieldList<%(Dimension)s, typename %(Dimension)s::Vector>"

@PYB11template("Dimension")
def gradDivVectorFieldListPairWise(fieldList = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                                   position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                                   weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                   mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                   rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                   Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                                   kernel = "const TableKernel<%(Dimension)s>&"):
    """Explicit method that performs the div and grad operations as sequential first
derivatives.  In this case we use the direct pairwise operators."""
    return "FieldList<%(Dimension)s, typename %(Dimension)s::Vector>"

@PYB11template("Dimension")
def gradDivVectorFieldListSimple(fieldList = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                                 position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                                 weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                 mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                 rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                 Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                                 kernel = "const TableKernel<%(Dimension)s>&"):
    "Simplest method, just use the second derivative of the kernel."
    return "FieldList<%(Dimension)s, typename %(Dimension)s::Vector>"

@PYB11template("Dimension")
def gradDivVectorFieldListGolden(fieldList = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                                 position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                                 weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                 mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                 rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                 Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                                 kernel = "const TableKernel<%(Dimension)s>&"):
    'More complex method which relies on the "golden rule", in that we move the mass density inside the operator.'
    return "FieldList<%(Dimension)s, typename %(Dimension)s::Vector>"

@PYB11template("Dimension")
def gradDivVectorFieldListGolden2(fieldList = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                                  position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                                  weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                  mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                  rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                  Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                                  kernel = "const TableKernel<%(Dimension)s>&"):
    """Calculate the gradient of the divergence of a Vector FieldList.
More complex method which relies on the "golden rule", in that we move the mass
density inside the operator."""
    return "FieldList<%(Dimension)s, typename %(Dimension)s::Vector>"

@PYB11template("Dimension")
def gradDivVectorFieldListMash(fieldList = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                               position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                               weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                               mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                               rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                               Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                               kernel = "const TableKernel<%(Dimension)s>&"):
    "MASH formalism for estimating grad div F."
    return "FieldList<%(Dimension)s, typename %(Dimension)s::Vector>"

@PYB11template("Dimension", "DataType")
def gradientPairWise(fieldList = "const FieldList<%(Dimension)s, %(DataType)s>&",
                     position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                     weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                     mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                     rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                     Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                     kernel = "const TableKernel<%(Dimension)s>&"):
    "Calculate the gradient of the given FieldList using low-order pairwise operations."
    return "FieldList<%(Dimension)s, typename MathTraits<%(Dimension)s, %(DataType)s>::GradientType>"

@PYB11template("Dimension", "DataType")
def divergencePairWise(fieldList = "const FieldList<%(Dimension)s, %(DataType)s>&",
                       position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                       weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                       mass = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                       rho = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                       Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                       kernel = "const TableKernel<%(Dimension)s>&"):
    "Calculate the divergence of the given FieldList using low-order pairwise operations."
    return "FieldList<%(Dimension)s, typename MathTraits<%(Dimension)s, %(DataType)s>::DivergenceType>"

@PYB11template("Dimension")
@PYB11implementation("""[](const FieldListSet<%(Dimension)s>& fieldListSet,
                           const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>& position,
                           const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>& weight,
                           const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>& Hfield,
                           const FieldList<%(Dimension)s, int>& mask,
                           const TableKernel<%(Dimension)s>& W,
                           const typename %(Dimension)s::Vector& xmin,
                           const typename %(Dimension)s::Vector& xmax,
                           const std::vector<int>& nsample) {
                               std::vector<std::vector<double>> scalarValues;
                               std::vector<std::vector<typename %(Dimension)s::Vector>> vectorValues;
                               std::vector<std::vector<typename %(Dimension)s::Tensor>> tensorValues;
                               std::vector<std::vector<typename %(Dimension)s::SymTensor>> symTensorValues;
                               sampleMultipleFields2Lattice(fieldListSet,
                                                            position,
                                                            weight,
                                                            Hfield,
                                                            mask,
                                                            W,
                                                            xmin,
                                                            xmax,
                                                            nsample,
                                                            scalarValues,
                                                            vectorValues,
                                                            tensorValues,
                                                            symTensorValues);
                               return py::make_tuple(scalarValues,
                                                     vectorValues,
                                                     tensorValues,
                                                     symTensorValues);
                           }""")
def sampleMultipleFields2Lattice(fieldListSet = "const FieldListSet<%(Dimension)s>&",
                                 position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                                 weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                 Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                                 mask = "const FieldList<%(Dimension)s, int>&",
                                 W = "const TableKernel<%(Dimension)s>&",
                                 xmin = "const typename %(Dimension)s::Vector&",
                                 xmax = "const typename %(Dimension)s::Vector&",
                                 nsample = "const std::vector<int>&"):
    "Simultaneously SPH sample multiple FieldLists to a lattice."
    return "py::tuple"

@PYB11template("Dimension")
@PYB11implementation("""[](const FieldListSet<%(Dimension)s>& fieldListSet,
                           const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>& position,
                           const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>& weight,
                           const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>& Hfield,
                           const FieldList<%(Dimension)s, int>& mask,
                           const TableKernel<%(Dimension)s>& W,
                           const typename %(Dimension)s::Vector& xmin,
                           const typename %(Dimension)s::Vector& xmax,
                           const std::vector<int>& nsample) {
                               std::vector<std::vector<double>> scalarValues;
                               std::vector<std::vector<typename %(Dimension)s::Vector>> vectorValues;
                               std::vector<std::vector<typename %(Dimension)s::Tensor>> tensorValues;
                               std::vector<std::vector<typename %(Dimension)s::SymTensor>> symTensorValues;
                               sampleMultipleFields2LatticeMash(fieldListSet,
                                                                position,
                                                                weight,
                                                                Hfield,
                                                                mask,
                                                                W,
                                                                xmin,
                                                                xmax,
                                                                nsample,
                                                                scalarValues,
                                                                vectorValues,
                                                                tensorValues,
                                                                symTensorValues);
                               return py::make_tuple(scalarValues,
                                                     vectorValues,
                                                     tensorValues,
                                                     symTensorValues);
                           }""")
def sampleMultipleFields2LatticeMash(fieldListSet = "const FieldListSet<%(Dimension)s>&",
                                     position = "const FieldList<%(Dimension)s, typename %(Dimension)s::Vector>&",
                                     weight = "const FieldList<%(Dimension)s, typename %(Dimension)s::Scalar>&",
                                     Hfield = "const FieldList<%(Dimension)s, typename %(Dimension)s::SymTensor>&",
                                     mask = "const FieldList<%(Dimension)s, int>&",
                                     W = "const TableKernel<%(Dimension)s>&",
                                     xmin = "const typename %(Dimension)s::Vector&",
                                     xmax = "const typename %(Dimension)s::Vector&",
                                     nsample = "const std::vector<int>&"):
    "Simultaneously MASH sample multiple FieldLists to a lattice."
    return "py::tuple"

@PYB11template("Dimension", "Value")
def binFieldList2Lattice(fieldList = "const FieldList<%(Dimension)s, %(Value)s>&",
                         xmin = "const typename %(Dimension)s::Vector&",
                         xmax = "const typename %(Dimension)s::Vector&",
                         nsample = "const std::vector<unsigned>&"):
    """Bin the values of a FieldList to a lattice.
The results are returned as a vector<%(Value)s>, of size nsample[0]*nsample[1]*....
Note, in parallel this does the global reduction for you, so you get back
the global result.

Do a straightforward binning."""
    return "std::vector<%(Value)s>"

@PYB11template("Dimension", "Value")
@PYB11cppname("binFieldList2Lattice")
def binFieldList2LatticeSmooth(fieldList = "const FieldList<%(Dimension)s, %(Value)s>&",
                               W = "const TableKernel<%(Dimension)s>&",
                               xmin = "const typename %(Dimension)s::Vector&",
                               xmax = "const typename %(Dimension)s::Vector&",
                               nsample = "const std::vector<unsigned>&"):
    """Bin the values of a FieldList to a lattice.
The results are returned as a vector<%(Value)s>, of size nsample[0]*nsample[1]*....
Note, in parallel this does the global reduction for you, so you get back
the global result.

Bin with kernel smoothing applied to the FieldList."""
    return "std::vector<%(Value)s>"

#-------------------------------------------------------------------------------
# Instantiate the per dimension bindings
#-------------------------------------------------------------------------------
for ndim in dims:
    exec('''
smoothScalarFields%(ndim)id = PYB11TemplateFunction(smoothFields, template_parameters=("%(Dimension)s", "%(Scalar)s"))
smoothVectorFields%(ndim)id = PYB11TemplateFunction(smoothFields, template_parameters=("%(Dimension)s", "%(Vector)s"))
smoothTensorFields%(ndim)id = PYB11TemplateFunction(smoothFields, template_parameters=("%(Dimension)s", "%(Tensor)s"))
smoothSymTensorFields%(ndim)id = PYB11TemplateFunction(smoothFields, template_parameters=("%(Dimension)s", "%(SymTensor)s"))

gradientScalar%(ndim)id = PYB11TemplateFunction(gradient, template_parameters=("%(Dimension)s", "%(Scalar)s"), pyname="gradient")
gradientVector%(ndim)id = PYB11TemplateFunction(gradient, template_parameters=("%(Dimension)s", "%(Vector)s"), pyname="gradient")

gradientStdVectorScalar%(ndim)id = PYB11TemplateFunction(gradientVec, template_parameters=("%(Dimension)s", "%(Scalar)s"), pyname="gradient")
gradientStdVectorVector%(ndim)id = PYB11TemplateFunction(gradientVec, template_parameters=("%(Dimension)s", "%(Vector)s"), pyname="gradient")
    
divergenceVector%(ndim)id = PYB11TemplateFunction(divergence, template_parameters=("%(Dimension)s", "%(Vector)s"), pyname="divergence")
divergenceTensor%(ndim)id = PYB11TemplateFunction(divergence, template_parameters=("%(Dimension)s", "%(Tensor)s"), pyname="divergence")
divergenceSymTensor%(ndim)id = PYB11TemplateFunction(divergence, template_parameters=("%(Dimension)s", "%(SymTensor)s"), pyname="divergence")

scalarLimiter%(ndim)id = PYB11TemplateFunction(limiter, template_parameters=("%(Dimension)s", "%(Scalar)s"), pyname="scalarLimiter%(ndim)id")
vectorLimiter%(ndim)id = PYB11TemplateFunction(limiter, template_parameters=("%(Dimension)s", "%(Vector)s"), pyname="vectorLimiter%(ndim)id")

smoothScalarFieldsMash%(ndim)id = PYB11TemplateFunction(smoothFieldsMash, template_parameters=("%(Dimension)s", "%(Scalar)s"))
smoothVectorFieldsMash%(ndim)id = PYB11TemplateFunction(smoothFieldsMash, template_parameters=("%(Dimension)s", "%(Vector)s"))
smoothTensorFieldsMash%(ndim)id = PYB11TemplateFunction(smoothFieldsMash, template_parameters=("%(Dimension)s", "%(Tensor)s"))
smoothSymTensorFieldsMash%(ndim)id = PYB11TemplateFunction(smoothFieldsMash, template_parameters=("%(Dimension)s", "%(SymTensor)s"))

sampleMultipleFieldsMash%(ndim)id = PYB11TemplateFunction(sampleMultipleFieldsMash, template_parameters="%(Dimension)s")

splatScalarFieldsMash%(ndim)id = PYB11TemplateFunction(splatFieldsMash, template_parameters=("%(Dimension)s", "%(Scalar)s"))
splatVectorFieldsMash%(ndim)id = PYB11TemplateFunction(splatFieldsMash, template_parameters=("%(Dimension)s", "%(Vector)s"))
splatTensorFieldsMash%(ndim)id = PYB11TemplateFunction(splatFieldsMash, template_parameters=("%(Dimension)s", "%(Tensor)s"))
splatSymTensorFieldsMash%(ndim)id = PYB11TemplateFunction(splatFieldsMash, template_parameters=("%(Dimension)s", "%(SymTensor)s"))

splatMultipleFieldsMash%(ndim)id = PYB11TemplateFunction(splatMultipleFieldsMash, template_parameters="%(Dimension)s")

gradientScalarMash%(ndim)id = PYB11TemplateFunction(gradientMash, template_parameters=("%(Dimension)s", "%(Scalar)s"), pyname="gradientMash")
gradientVectorMash%(ndim)id = PYB11TemplateFunction(gradientMash, template_parameters=("%(Dimension)s", "%(Vector)s"), pyname="gradientMash")

gradDivVectorFieldList%(ndim)id = PYB11TemplateFunction(gradDivVectorFieldList, template_parameters="%(Dimension)s", pyname="gradienDivVectorFieldList")
gradDivVectorFieldListPairWise%(ndim)id = PYB11TemplateFunction(gradDivVectorFieldListPairWise, template_parameters="%(Dimension)s", pyname="gradienDivVectorFieldListPairWise")
gradDivVectorFieldListSimple%(ndim)id = PYB11TemplateFunction(gradDivVectorFieldListSimple, template_parameters="%(Dimension)s", pyname="gradienDivVectorFieldListSimple")
gradDivVectorFieldListGolden%(ndim)id = PYB11TemplateFunction(gradDivVectorFieldListGolden, template_parameters="%(Dimension)s", pyname="gradienDivVectorFieldListGolden")
gradDivVectorFieldListGolden2%(ndim)id = PYB11TemplateFunction(gradDivVectorFieldListGolden2, template_parameters="%(Dimension)s", pyname="gradienDivVectorFieldListGolden2")
gradDivVectorFieldListMash%(ndim)id = PYB11TemplateFunction(gradDivVectorFieldListMash, template_parameters="%(Dimension)s", pyname="gradienDivVectorFieldListMash")

gradientScalarPairWise%(ndim)id = PYB11TemplateFunction(gradientPairWise, template_parameters=("%(Dimension)s", "%(Scalar)s"), pyname="gradientPairWise")
gradientVectorPairWise%(ndim)id = PYB11TemplateFunction(gradientPairWise, template_parameters=("%(Dimension)s", "%(Vector)s"), pyname="gradientPairWise")

divergenceVectorPairWise%(ndim)id = PYB11TemplateFunction(divergencePairWise, template_parameters=("%(Dimension)s", "%(Vector)s"), pyname="divergencePairWise")

sampleMultipleFields2Lattice%(ndim)id = PYB11TemplateFunction(sampleMultipleFields2Lattice, template_parameters="%(Dimension)s")
sampleMultipleFields2LatticeMash%(ndim)id = PYB11TemplateFunction(sampleMultipleFields2LatticeMash, template_parameters="%(Dimension)s")

binScalarFieldList2Lattice%(ndim)id = PYB11TemplateFunction(binFieldList2Lattice, template_parameters=("%(Dimension)s", "%(Scalar)s"), pyname="binFieldList2Lattice")
binVectorFieldList2Lattice%(ndim)id = PYB11TemplateFunction(binFieldList2Lattice, template_parameters=("%(Dimension)s", "%(Vector)s"), pyname="binFieldList2Lattice")
binTensorFieldList2Lattice%(ndim)id = PYB11TemplateFunction(binFieldList2Lattice, template_parameters=("%(Dimension)s", "%(Tensor)s"), pyname="binFieldList2Lattice")
binSymTensorFieldList2Lattice%(ndim)id = PYB11TemplateFunction(binFieldList2Lattice, template_parameters=("%(Dimension)s", "%(SymTensor)s"), pyname="binFieldList2Lattice")

binScalarFieldList2LatticeSmooth%(ndim)id = PYB11TemplateFunction(binFieldList2LatticeSmooth, template_parameters=("%(Dimension)s", "%(Scalar)s"), pyname="binFieldList2Lattice")
binVectorFieldList2LatticeSmooth%(ndim)id = PYB11TemplateFunction(binFieldList2LatticeSmooth, template_parameters=("%(Dimension)s", "%(Vector)s"), pyname="binFieldList2Lattice")
binTensorFieldList2LatticeSmooth%(ndim)id = PYB11TemplateFunction(binFieldList2LatticeSmooth, template_parameters=("%(Dimension)s", "%(Tensor)s"), pyname="binFieldList2Lattice")
binSymTensorFieldList2LatticeSmooth%(ndim)id = PYB11TemplateFunction(binFieldList2LatticeSmooth, template_parameters=("%(Dimension)s", "%(SymTensor)s"), pyname="binFieldList2Lattice")
''' % {"ndim"      : ndim,
       "Dimension" : "Dim<" + str(ndim) + ">",
       "Scalar"    : "Dim<" + str(ndim) + ">::Scalar",
       "Vector"    : "Dim<" + str(ndim) + ">::Vector",
       "Tensor"    : "Dim<" + str(ndim) + ">::Tensor",
       "SymTensor" : "Dim<" + str(ndim) + ">::SymTensor"})
