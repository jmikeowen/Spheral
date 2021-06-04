//---------------------------------Spheral++----------------------------------//
// SidreDataCollection -- Store fields into sidre (axom's data storage)
//
//
// Created by Mikhail Zakharchanka, 2020
//----------------------------------------------------------------------------//
#ifndef SidreDataCollection_HH
#define SidreDataCollection_HH

#include "axom/sidre.hpp"
#include "Field/Field.hh"
#include "Geometry/Dimension.hh"

namespace Spheral
{

class SidreDataCollection
{
public:
    SidreDataCollection();
    ~SidreDataCollection();

    template <typename Dimension, typename DataType,
              typename std::enable_if<std::is_arithmetic<DataType>::value,
                                      DataType>::type* = nullptr>
    axom::sidre::View *alloc_view(const std::string &view_name,
                                  const Spheral::Field<Dimension, DataType> &field);

    template <typename Dimension, typename DataType,
              typename std::enable_if<!std::is_arithmetic<DataType>::value,
                                      DataType>::type* = nullptr>
    axom::sidre::View *alloc_view(const std::string &view_name,
                                  const Spheral::Field<Dimension, DataType> &field);

    
    template<typename Dimension>
    axom::sidre::View *alloc_view(const std::string &view_name,
                                  const Spheral::Field<Dimension, std::string> &field);
    template<typename Dimension, typename DataType>
    axom::sidre::View *alloc_view(const std::string &view_name,
                                  const Spheral::Field<Dimension, std::vector<DataType>> &field);
    template<typename Dimension, typename DataType>
    axom::sidre::View *alloc_view(const std::string &view_name,
                                  const Spheral::Field<Dimension, std::tuple<DataType, DataType, DataType>> &field);
    template<typename Dimension, typename DataType>
    axom::sidre::View *alloc_view(const std::string &view_name,
                                  const Spheral::Field<Dimension, std::tuple<DataType, DataType, DataType, DataType>> &field);
    template<typename Dimension, typename DataType>
    axom::sidre::View *alloc_view(const std::string &view_name,
                                  const Spheral::Field<Dimension, std::tuple<DataType, DataType, DataType, DataType, DataType>> &field);

    template<typename Dimension>
    axom::sidre::View *alloc_view(const std::string &view_name,
                                  const Spheral::Field<Dimension, Dim<1>::ThirdRankTensor> &field);

    void printData() {m_datastore_ptr->getRoot()->print();};
        
private:
    axom::sidre::DataStore *m_datastore_ptr;
};

}

#include "Utilities/SidreDataCollectionInline.hh"

#endif
