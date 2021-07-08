"""
Spheral Field module.

Provides the Field classes.
"""

from PYB11Generator import *
from SpheralCommon import *
from spheralDimensions import *
dims = spheralDimensions()

from FieldBase import *
from Field import *
from ArithmeticField import *
from MinMaxField import *

#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------
PYB11includes += ['"Geometry/Dimension.hh"',
                  '"Field/FieldBase.hh"',
                  '"Field/Field.hh"',
                  '"Field/FieldList.hh"',
                  '"Field/FieldListSet.hh"',
                  '"Utilities/FieldDataTypeTraits.hh"',
                  '"Utilities/DomainNode.hh"',
                  '"Geometry/CellFaceFlag.hh"',
                  '<vector>']

#-------------------------------------------------------------------------------
# Namespaces
#-------------------------------------------------------------------------------
PYB11namespaces = ["Spheral"]

#-------------------------------------------------------------------------------
# Enums
#-------------------------------------------------------------------------------
FieldStorageType = PYB11enum(("ReferenceFields", "CopyFields"), export_values=True,
                             doc="Storage types for Fields in FieldLists.")

#-------------------------------------------------------------------------------
# Do our dimension dependent instantiations.
#-------------------------------------------------------------------------------
for ndim in dims:

    #...........................................................................
    # FieldBase
    exec('''
FieldBase%(ndim)id = PYB11TemplateClass(FieldBase, template_parameters="Dim<%(ndim)i>")
''' % {"ndim" : ndim})

    #...........................................................................
    # non-numeric types
    for (value, label) in (("Dim<%i>::FacetedVolume" % ndim,       "FacetedVolume"),
                           ("std::vector<int>",                    "VectorInt"),
                           ("std::vector<double>",                 "VectorDouble"),
                           ("std::vector<Dim<%i>::Vector>" % ndim, "VectorVector"),
                           ("std::vector<Dim<%i>::Tensor>" % ndim, "VectorSymTensor"),
                           ("std::vector<Dim<%i>::Tensor>" % ndim, "VectorSymTensor"),
                           ("std::vector<CellFaceFlag>",           "vector_of_CellFaceFlag"),
                           ("DomainNode<Dim<%i>>" % ndim,          "DomainNode"),
                           ("RKCoefficients<Dim<%i>>" % ndim,      "RKCoefficients")):
        exec('''
%(label)sField%(ndim)sd = PYB11TemplateClass(Field, template_parameters=("Dim<%(ndim)i>", "%(value)s"))
''' % {"ndim" : ndim,
       "value" : value,
       "label" : label})

    #...........................................................................
    # arithmetic fields
    for (value, label) in (("int",                              "Int"),
                           ("unsigned",                         "Unsigned"),
                           ("uint64_t",                         "ULL"),
                           ("Dim<%i>::Vector" % ndim,           "Vector"),
                           ("Dim<%i>::Tensor" % ndim,           "Tensor"),
                           ("Dim<%i>::ThirdRankTensor" % ndim,  "ThirdRankTensor"),
                           ("Dim<%i>::FourthRankTensor" % ndim, "FourthRankTensor"),
                           ("Dim<%i>::FifthRankTensor" % ndim,  "FifthRankTensor")):
        exec('''
%(label)sField%(ndim)sd = PYB11TemplateClass(ArithmeticField, template_parameters=("Dim<%(ndim)i>", "%(value)s"))
''' % {"ndim" : ndim,
       "value" : value,
       "label" : label})

    #...........................................................................
    # A few fields can apply the min/max with a scalar addtionally
    for (value, label) in (("double",                     "Scalar"),
                           ("Dim<%i>::SymTensor" % ndim,  "SymTensor")):
        exec('''
%(label)sField%(ndim)sd = PYB11TemplateClass(MinMaxField, template_parameters=("Dim<%(ndim)i>", "%(value)s"))
''' % {"ndim" : ndim,
       "value" : value,
       "label" : label})

    #...........................................................................
    # STL collections of Field types
    for value, label in (("int", "Int"),
                         ("double", "Scalar"),
                         ("Dim<%i>::Vector" % ndim, "Vector"),
                         ("Dim<%i>::Tensor" % ndim, "Tensor"),
                         ("Dim<%i>::SymTensor" % ndim, "SymTensor")):
        exec('''
vector_of_%(label)sField%(ndim)id = PYB11_bind_vector("Field<%(Dimension)s, %(value)s>", opaque=True, local=False)
vector_of_%(label)sFieldPtr%(ndim)id = PYB11_bind_vector("Field<%(Dimension)s, %(value)s>*", opaque=True, local=False)
vector_of_vector_of_%(label)sField%(ndim)id = PYB11_bind_vector("std::vector<Field<%(Dimension)s, %(value)s>>", opaque=True, local=False)
''' % {"ndim" : ndim,
       "value" : value,
       "label" : label,
       "Dimension" : "Dim<" + str(ndim) + ">"})
