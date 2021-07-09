"""
Spheral FileIO module.

Provides the interfaces for communicating with file systems.
"""

from PYB11Generator import *
from SpheralCommon import *
from spheralDimensions import *
dims = spheralDimensions()

#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------
PYB11includes += ['"NodeList/NodeList.hh"',
                  '"NodeList/FluidNodeList.hh"',
                  '"NodeList/SolidNodeList.hh"',
                  '"Field/Field.hh"',
                  '"Field/FieldList.hh"',
                  '"FileIO/FileIO.hh"',
                  '"FileIO/SiloFileIO.hh"',
                  '"FileIO/PyFileIO.hh"',
                  '"FileIO/vectorstringUtilities.hh"']

#-------------------------------------------------------------------------------
# More preamble
#-------------------------------------------------------------------------------
PYB11opaque += ["std::vector<FluidNodeList<Dim<1>>*>",
                "std::vector<FluidNodeList<Dim<2>>*>",
                "std::vector<FluidNodeList<Dim<3>>*>",
                "std::vector<NodeList<Dim<1>>*>",
                "std::vector<NodeList<Dim<2>>*>",
                "std::vector<NodeList<Dim<3>>*>",
                "std::vector<SolidNodeList<Dim<1>>*>",
                "std::vector<SolidNodeList<Dim<2>>*>",
                "std::vector<SolidNodeList<Dim<3>>*>",

                "std::vector<Field<Dim<1> , int>>",
                "std::vector<Field<Dim<2> , int>>",
                "std::vector<Field<Dim<3> , int>>",
                "std::vector<FieldList<Dim<1> , int>>",
                "std::vector<FieldList<Dim<2> , int>>",
                "std::vector<FieldList<Dim<3> , int>>",
                "std::vector<FieldList<Dim<1> , int>*>",
                "std::vector<FieldList<Dim<2> , int>*>",
                "std::vector<FieldList<Dim<3> , int>*>",
                "std::vector<Field<Dim<1> , int>*>",
                "std::vector<Field<Dim<2> , int>*>",
                "std::vector<Field<Dim<3> , int>*>",
                "std::vector<Field<Dim<1> , double>>",
                "std::vector<Field<Dim<2> , double>>",
                "std::vector<Field<Dim<3> , double>>",
                "std::vector<FieldList<Dim<1> , double>>",
                "std::vector<FieldList<Dim<2> , double>>",
                "std::vector<FieldList<Dim<3> , double>>",
                "std::vector<FieldList<Dim<1> , double>*>",
                "std::vector<FieldList<Dim<2> , double>*>",
                "std::vector<FieldList<Dim<3> , double>*>",
                "std::vector<Field<Dim<1> , double>*>",
                "std::vector<Field<Dim<2> , double>*>",
                "std::vector<Field<Dim<3> , double>*>",
                "std::vector<Field<Dim<1> , Dim<1>::SymTensor>>",
                "std::vector<Field<Dim<2> , Dim<2>::SymTensor>>",
                "std::vector<Field<Dim<3> , Dim<3>::SymTensor>>",
                "std::vector<FieldList<Dim<1> , Dim<1>::SymTensor>>",
                "std::vector<FieldList<Dim<2> , Dim<2>::SymTensor>>",
                "std::vector<FieldList<Dim<3> , Dim<3>::SymTensor>>",
                "std::vector<FieldList<Dim<1> , Dim<1>::SymTensor>*>",
                "std::vector<FieldList<Dim<2> , Dim<2>::SymTensor>*>",
                "std::vector<FieldList<Dim<3> , Dim<3>::SymTensor>*>",
                "std::vector<Field<Dim<1> , Dim<1>::SymTensor>*>",
                "std::vector<Field<Dim<2> , Dim<2>::SymTensor>*>",
                "std::vector<Field<Dim<3> , Dim<3>::SymTensor>*>",
                "std::vector<Field<Dim<1> , Dim<1>::Tensor>>",
                "std::vector<Field<Dim<2> , Dim<2>::Tensor>>",
                "std::vector<Field<Dim<3> , Dim<3>::Tensor>>",
                "std::vector<FieldList<Dim<1> , Dim<1>::Tensor>>",
                "std::vector<FieldList<Dim<2> , Dim<2>::Tensor>>",
                "std::vector<FieldList<Dim<3> , Dim<3>::Tensor>>",
                "std::vector<FieldList<Dim<1> , Dim<1>::Tensor>*>",
                "std::vector<FieldList<Dim<2> , Dim<2>::Tensor>*>",
                "std::vector<FieldList<Dim<3> , Dim<3>::Tensor>*>",
                "std::vector<Field<Dim<1> , Dim<1>::Tensor>*>",
                "std::vector<Field<Dim<2> , Dim<2>::Tensor>*>",
                "std::vector<Field<Dim<3> , Dim<3>::Tensor>*>",
                "std::vector<Field<Dim<1> , Dim<1>::Vector>>",
                "std::vector<Field<Dim<2> , Dim<2>::Vector>>",
                "std::vector<Field<Dim<3> , Dim<3>::Vector>>",
                "std::vector<FieldList<Dim<1> , Dim<1>::Vector>>",
                "std::vector<FieldList<Dim<2> , Dim<2>::Vector>>",
                "std::vector<FieldList<Dim<3> , Dim<3>::Vector>>",
                "std::vector<FieldList<Dim<1> , Dim<1>::Vector>*>",
                "std::vector<FieldList<Dim<2> , Dim<2>::Vector>*>",
                "std::vector<FieldList<Dim<3> , Dim<3>::Vector>*>",
                "std::vector<Field<Dim<1> , Dim<1>::Vector>*>",
                "std::vector<Field<Dim<2> , Dim<2>::Vector>*>",
                "std::vector<Field<Dim<3> , Dim<3>::Vector>*>",
                "std::vector<std::vector<Field<Dim<1> , int>>>",
                "std::vector<std::vector<Field<Dim<2> , int>>>",
                "std::vector<std::vector<Field<Dim<3> , int>>>",
                "std::vector<std::vector<FieldList<Dim<1> , int>>>",
                "std::vector<std::vector<FieldList<Dim<2> , int>>>",
                "std::vector<std::vector<FieldList<Dim<3> , int>>>",
                "std::vector<std::vector<Field<Dim<1> , double>>>",
                "std::vector<std::vector<Field<Dim<2> , double>>>",
                "std::vector<std::vector<Field<Dim<3> , double>>>",
                "std::vector<std::vector<FieldList<Dim<1> , double>>>",
                "std::vector<std::vector<FieldList<Dim<2> , double>>>",
                "std::vector<std::vector<FieldList<Dim<3> , double>>>",
                "std::vector<std::vector<Field<Dim<1> , Dim<1>::SymTensor>>>",
                "std::vector<std::vector<Field<Dim<2> , Dim<2>::SymTensor>>>",
                "std::vector<std::vector<Field<Dim<3> , Dim<3>::SymTensor>>>",
                "std::vector<std::vector<FieldList<Dim<1> , Dim<1>::SymTensor>>>",
                "std::vector<std::vector<FieldList<Dim<2> , Dim<2>::SymTensor>>>",
                "std::vector<std::vector<FieldList<Dim<3> , Dim<3>::SymTensor>>>",
                "std::vector<std::vector<Field<Dim<1> , Dim<1>::Tensor>>>",
                "std::vector<std::vector<Field<Dim<2> , Dim<2>::Tensor>>>",
                "std::vector<std::vector<Field<Dim<3> , Dim<3>::Tensor>>>",
                "std::vector<std::vector<FieldList<Dim<1> , Dim<1>::Tensor>>>",
                "std::vector<std::vector<FieldList<Dim<2> , Dim<2>::Tensor>>>",
                "std::vector<std::vector<FieldList<Dim<3> , Dim<3>::Tensor>>>",
                "std::vector<std::vector<Field<Dim<1> , Dim<1>::Vector>>>",
                "std::vector<std::vector<Field<Dim<2> , Dim<2>::Vector>>>",
                "std::vector<std::vector<Field<Dim<3> , Dim<3>::Vector>>>",
                "std::vector<std::vector<FieldList<Dim<1> , Dim<1>::Vector>>>",
                "std::vector<std::vector<FieldList<Dim<2> , Dim<2>::Vector>>>",
                "std::vector<std::vector<FieldList<Dim<3> , Dim<3>::Vector>>>"]

#-------------------------------------------------------------------------------
# Namespaces
#-------------------------------------------------------------------------------
PYB11namespaces = ["Spheral"]

#-------------------------------------------------------------------------------
# Enums
#-------------------------------------------------------------------------------
AccessType = PYB11enum(("Undefined", 
                        "Create", 
                        "Read",
                        "Write",
                        "ReadWrite"), export_values=True,
                       doc="How are we opening/accessing a file")

#-------------------------------------------------------------------------------
# Instantiate our types
#-------------------------------------------------------------------------------
from FileIO import *
from SiloFileIO import *
from PyFileIO import *

#-------------------------------------------------------------------------------
# Module methods
#-------------------------------------------------------------------------------
@PYB11template("T")
def vector2string(val = "const std::vector<%(T)s>&",
                  precision = ("const int", "30")):
    "Serialize vector<%(T)s> -> std::string"
    return "std::string"

@PYB11template("T")
def string2vector(val = "const std::string&"):
    "Deserialize std::string -> std::vector<%(T)s>"
    return "std::vector<%(T)s>"

vec2string1  = PYB11TemplateFunction(vector2string, template_parameters="int", pyname="vector2string")
vec2string2  = PYB11TemplateFunction(vector2string, template_parameters="unsigned", pyname="vector2string")
vec2string3  = PYB11TemplateFunction(vector2string, template_parameters="uint64_t", pyname="vector2string")
vec2string4  = PYB11TemplateFunction(vector2string, template_parameters="double", pyname="vector2string")
vec2string5  = PYB11TemplateFunction(vector2string, template_parameters="std::string", pyname="vector2string")
vec2string6  = PYB11TemplateFunction(vector2string, template_parameters="Dim<1>::Vector", pyname="vector2string")
vec2string7  = PYB11TemplateFunction(vector2string, template_parameters="Dim<1>::Tensor", pyname="vector2string")
vec2string8  = PYB11TemplateFunction(vector2string, template_parameters="Dim<1>::SymTensor", pyname="vector2string")
vec2string9  = PYB11TemplateFunction(vector2string, template_parameters="Dim<1>::ThirdRankTensor", pyname="vector2string")
vec2string10 = PYB11TemplateFunction(vector2string, template_parameters="Dim<2>::Vector", pyname="vector2string")
vec2string11 = PYB11TemplateFunction(vector2string, template_parameters="Dim<2>::Tensor", pyname="vector2string")
vec2string12 = PYB11TemplateFunction(vector2string, template_parameters="Dim<2>::SymTensor", pyname="vector2string")
vec2string13 = PYB11TemplateFunction(vector2string, template_parameters="Dim<2>::ThirdRankTensor", pyname="vector2string")
vec2string14 = PYB11TemplateFunction(vector2string, template_parameters="Dim<3>::Vector", pyname="vector2string")
vec2string15 = PYB11TemplateFunction(vector2string, template_parameters="Dim<3>::Tensor", pyname="vector2string")
vec2string16 = PYB11TemplateFunction(vector2string, template_parameters="Dim<3>::SymTensor", pyname="vector2string")
vec2string17 = PYB11TemplateFunction(vector2string, template_parameters="Dim<3>::ThirdRankTensor", pyname="vector2string")

# These have to be uniquely named, since the arguments are the same.
string2vector_of_int = PYB11TemplateFunction(string2vector, template_parameters="int")
string2vector_of_unsigned = PYB11TemplateFunction(string2vector, template_parameters="unsigned")
string2vector_of_ULL = PYB11TemplateFunction(string2vector, template_parameters="uint64_t")
string2vector_of_double = PYB11TemplateFunction(string2vector, template_parameters="double")
string2vector_of_string = PYB11TemplateFunction(string2vector, template_parameters="std::string")
string2vector_of_Vector1d = PYB11TemplateFunction(string2vector, template_parameters="Dim<1>::Vector")
string2vector_of_Vector2d = PYB11TemplateFunction(string2vector, template_parameters="Dim<2>::Vector")
string2vector_of_Vector3d = PYB11TemplateFunction(string2vector, template_parameters="Dim<3>::Vector")
string2vector_of_Tensor1d = PYB11TemplateFunction(string2vector, template_parameters="Dim<1>::Tensor")
string2vector_of_Tensor2d = PYB11TemplateFunction(string2vector, template_parameters="Dim<2>::Tensor")
string2vector_of_Tensor3d = PYB11TemplateFunction(string2vector, template_parameters="Dim<3>::Tensor")
string2vector_of_SymTensor1d = PYB11TemplateFunction(string2vector, template_parameters="Dim<1>::SymTensor")
string2vector_of_SymTensor2d = PYB11TemplateFunction(string2vector, template_parameters="Dim<2>::SymTensor")
string2vector_of_SymTensor3d = PYB11TemplateFunction(string2vector, template_parameters="Dim<3>::SymTensor")
string2vector_of_ThirdRankTensor1d = PYB11TemplateFunction(string2vector, template_parameters="Dim<1>::ThirdRankTensor")
string2vector_of_ThirdRankTensor2d = PYB11TemplateFunction(string2vector, template_parameters="Dim<2>::ThirdRankTensor")
string2vector_of_ThirdRankTensor3d = PYB11TemplateFunction(string2vector, template_parameters="Dim<3>::ThirdRankTensor")
