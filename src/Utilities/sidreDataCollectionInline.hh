namespace Spheral
{

template<typename Dimension, typename DataType>
/* sidre::View * */ void SidreDataCollection::alloc_view(const std::string &view_name, 
                                             const Spheral::Field<Dimension, DataType> &field)
{
   sidre::DataTypeId dtype = field->getAxomType();
   axom::IndexType num_elements = field->numElements();
   void *data = field->allValues();
   sidre::View *v = m_datastore_ptr->getRoot()->createView(view_name, dtype,
                                                           num_elements, data);
}

}