#include "axom/sidre.hpp"
#include "Field/Field.hh"

namespace Spheral
{

template <typename Dimension, typename DataType,
         typename std::enable_if<std::is_arithmetic<DataType>::value,
                                 DataType>::type* = nullptr>
inline
axom::sidre::View *SidreDataCollection::alloc_view(const std::string &view_name, 
                                                   const Spheral::Field<Dimension, DataType> &field)
{
   axom::sidre::DataTypeId dtype = field.getAxomType();
   axom::IndexType num_elements = field.numElements();
   const DataType *data = &(*field.begin());
   axom::sidre::View *v = m_datastore_ptr->getRoot()->createView(view_name, dtype,
                                                        num_elements, (void*)data);
   return v;
}

//------------------------------------------------------------------------------
template<typename Dimension>
inline
axom::sidre::View *SidreDataCollection::alloc_view(const std::string &view_name, 
                                                   const Spheral::Field<Dimension, std::string> &field)
{
   axom::sidre::DataTypeId dtype = field.getAxomType();
   int view_count = 0;
   for (u_int i = 0; i < field.size(); i++)
   {
      axom::IndexType num_elements = field[i].size();
      const char *data = &(*field[i].begin());
      m_datastore_ptr->getRoot()->createView(view_name + std::to_string(view_count), dtype, 
                                                         num_elements, (void*)data);
      view_count++;
   }
   return m_datastore_ptr->getRoot()->getView(view_name + "0");
}

//------------------------------------------------------------------------------
template<typename Dimension, typename DataType>
inline
axom::sidre::View *SidreDataCollection::alloc_view(const std::string &view_name, 
                                                   const Spheral::Field<Dimension, std::vector<DataType>> &field)
{
   axom::sidre::DataTypeId dtype = field.getAxomType();
   int view_count = 0;
   for (u_int i = 0; i < field.size(); i++)
   {
      axom::IndexType num_elements = field[i].size();
      const DataType *data = &(*field[i].begin());
      m_datastore_ptr->getRoot()->createView(view_name + std::to_string(view_count), dtype, 
                                                         num_elements, (void*)data);
      view_count++;
   }
   return m_datastore_ptr->getRoot()->getView(view_name + "0");
}

//------------------------------------------------------------------------------
template<typename Dimension, typename DataType>
inline
axom::sidre::View *SidreDataCollection::alloc_view(const std::string &view_name,
                              const Spheral::Field<Dimension, std::tuple<DataType, DataType, DataType>> &field)
{
   axom::sidre::DataTypeId dtype = field.getAxomType();
   axom::IndexType num_elements = 3;
   int view_count = 0;

   for (u_int i = 0; i < field.size(); i++)
   {
      DataType data [] = {std::get<0>(field[i]), std::get<1>(field[i]), std::get<2>(field[i])};
      axom::sidre::Buffer* buff = m_datastore_ptr->createBuffer()->allocate(dtype, num_elements)
                                                 ->copyBytesIntoBuffer(data, sizeof(DataType) * num_elements);
      m_datastore_ptr->getRoot()->createView(view_name + std::to_string(view_count), dtype, 
                                                         num_elements, buff);
      view_count++;
   }
   return m_datastore_ptr->getRoot()->getView(view_name + "0");
}

//------------------------------------------------------------------------------
template<typename Dimension, typename DataType>
inline
axom::sidre::View *SidreDataCollection::alloc_view(const std::string &view_name,
                              const Spheral::Field<Dimension, std::tuple<DataType, DataType, DataType, DataType>> &field)
{
   axom::sidre::DataTypeId dtype = field.getAxomType();
   axom::IndexType num_elements = 4;
   int view_count = 0;
   for (u_int i = 0; i < field.size(); i++)
   {
      DataType data [] = {std::get<0>(field[i]), std::get<1>(field[i]), std::get<2>(field[i]), std::get<3>(field[i])};
      axom::sidre::Buffer* buff = m_datastore_ptr->createBuffer()->allocate(dtype, num_elements)
                                                 ->copyBytesIntoBuffer(data, sizeof(DataType) * num_elements);
      m_datastore_ptr->getRoot()->createView(view_name + std::to_string(view_count), dtype, 
                                                         num_elements, buff);
      view_count++;
   }
   return m_datastore_ptr->getRoot()->getView(view_name + "0");
}

//------------------------------------------------------------------------------
template<typename Dimension, typename DataType>
inline
axom::sidre::View *SidreDataCollection::alloc_view(const std::string &view_name,
                              const Spheral::Field<Dimension, std::tuple<DataType, DataType, DataType, DataType, DataType>> &field)
{
   axom::sidre::DataTypeId dtype = field.getAxomType();
   axom::IndexType num_elements = 5;
   int view_count = 0;
   for (u_int i = 0; i < field.size(); i++)
   {
      DataType data [] = {std::get<0>(field[i]), std::get<1>(field[i]), std::get<2>(field[i]), std::get<3>(field[i]), std::get<4>(field[i])};
      axom::sidre::Buffer* buff = m_datastore_ptr->createBuffer()->allocate(dtype, num_elements)
                                                 ->copyBytesIntoBuffer(data, sizeof(DataType) * num_elements);
      m_datastore_ptr->getRoot()->createView(view_name + std::to_string(view_count), dtype, 
                                                         num_elements, buff);
      view_count++;
   }
   return m_datastore_ptr->getRoot()->getView(view_name + "0");
}

template <typename Dimension, typename DataType,
         typename std::enable_if<!std::is_arithmetic<DataType>::value,
                                 DataType>::type* = nullptr>
inline
axom::sidre::View *SidreDataCollection::alloc_view(const std::string &view_name, 
                                                   const Spheral::Field<Dimension, DataType> &field)
{
   axom::sidre::DataTypeId dtype = field.getAxomType();
   axom::IndexType num_elements = field.size() * DataTypeTraits<DataType>::numElements(field[0]);

   auto *data = &(*field.begin());
   m_datastore_ptr->getRoot()->createView(view_name, dtype, num_elements, (void*)data);

   printData();

   return m_datastore_ptr->getRoot()->getView(view_name);
}



template<typename Dimension>
inline
axom::sidre::View *SidreDataCollection::alloc_view(const std::string &view_name, 
                                                   const Spheral::Field<Dimension, Dim<1>::ThirdRankTensor> &field)
{
   axom::sidre::DataTypeId dtype = field.getAxomType();
   axom::IndexType num_elements = field.size() * DataTypeTraits<Dim<1>::ThirdRankTensor>::numElements(field[0]);

   for (u_int i = 0; i < field.size(); ++i)
      std::cout << *field[i].begin() << " ";
   std::cout << std::endl;

   double data [field.size()];
   for (u_int i = 0; i < field.size(); ++i)
      data[i] = *field[i].begin();
   
   axom::sidre::Buffer* buff = m_datastore_ptr->createBuffer()->allocate(dtype, num_elements)
                                              ->copyBytesIntoBuffer(data, sizeof(double) * num_elements);
   m_datastore_ptr->getRoot()->createView(view_name, dtype, num_elements, buff);

   printData();

   return m_datastore_ptr->getRoot()->getView(view_name);
}

}
