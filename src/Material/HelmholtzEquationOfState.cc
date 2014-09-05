//
//  HelmholtzEquationOfState.cc
//  
//
//  Created by Raskin, Cody Dantes on 8/28/14.
//
//

#include "HelmholtzEquationOfState.hh"
#include "PhysicalConstants.hh"
#include "Field/Field.hh"
#include "Utilities/SpheralFunctions.hh"
#include "Utilities/DBC.hh"

#define Fortran2(x) x##_

extern "C" {
	void Fortran2(init_helm_table)();
	
	void Fortran2(get_helm_table)(double *f,double *fd,double *ft,double *fdd,
								  double *ftt,double *fdt,double *fddt,
								  double *fdtt,double *fddtt,double *dpdf,
								  double *dpdfd,double *dpdft,double *dpdfdt,
								  double *ef,double *efd,double *eft,double *efdt,
								  double *xf,double *xfd,double *xft,double *xfdt);
	
	void Fortran2(wrapper_invert_helm_ed)(int *npart, double *density,
										  double *energy, double *abar,
										  double *zbar, double *temperature,
										  double *pressure, double *small_temp, double *vsound);
	
	void Fortran2(wrapper_helmeos)(int *npart, double *den_row,
								   double *etot_row, double *abar_row,
								   double *zbar_row, double *temperature,
								   double *pressure);
	
	void Fortran2(set_helm_table)(double *f, double *fd, double *ft, double *fdd,
								  double *ftt, double *fdt, double *fddt,
								  double *fdtt, double *fddtt, double *dpdf,
								  double *dpdfd, double *dpdft, double *dpdfdt,
								  double *ef, double *efd, double *eft,
								  double *efdt, double *xf, double *xfd,
								  double *xft, double *xfdt);
	
	void Fortran2(azbar)(double *xmass, double *aion, double *zion, int *ionmax,
						 double *ymass, double *abar, double *zbar);
}

namespace Spheral {
namespace Material {
    
    using FieldSpace::Field;
    using NodeSpace::NodeList;


    template<typename Dimension>
    HelmholtzEquationOfState<Dimension>::
    HelmholtzEquationOfState(const NodeList<Dimension>& myNodeList,
                             const PhysicalConstants& constants,
                             const double minimumPressure,
                             const double maximumPressure,
                             const double minimumTemperature,
                             const double maximumTemperature,
                             const MaterialPressureMinType minPressureType,
                             const Scalar abar0,
                             const Scalar zbar0):
    EquationOfState<Dimension>(constants, minimumPressure, maximumPressure, minPressureType),
    mabar0(abar0),
    mzbar0(zbar0),
    mPmin(minimumPressure),
    mPmax(maximumPressure),
    mTmin(minimumTemperature),
    mTmax(maximumTemperature),
    mConstants(constants)
    {
        needUpdate = 1; // flip this on and off later
        Fortran2(init_helm_table);
    }

    //------------------------------------------------------------------------------
    // Destructor.
    //------------------------------------------------------------------------------
    template<typename Dimension>
    HelmholtzEquationOfState<Dimension>::
    ~HelmholtzEquationOfState() {
    }


    /* check mzbar != nullptr?? */


    //------------------------------------------------------------------------------
    // Set the pressure.
    //------------------------------------------------------------------------------
    template<typename Dimension>
    void
    HelmholtzEquationOfState<Dimension>::
    setPressure(Field<Dimension, Scalar>& Pressure,
                const Field<Dimension, Scalar>& massDensity,
                const Field<Dimension, Scalar>& specificThermalEnergy) const {
        CHECK(valid());
        
        if(!fieldsStored) storeFields(Pressure);
        
        int npart = massDensity.numElements();
        myMassDensity = massDensity;
        mySpecificThermalEnergy = specificThermalEnergy;
        
        if(needUpdate){
            Fortran2(wrapper_invert_helm_ed)(&npart, &myMassDensity[0], &mySpecificThermalEnergy[0],
                                             &myAbar[0], &myZbar[0], &myTemperature[0],
                                             &myPressure[0], &mTmin, &mySoundSpeed[0]);
        }
        
        for (size_t i = 0; i != npart; ++i) {
            Pressure(i) = myPressure(i);
        }
    }

    //------------------------------------------------------------------------------
    // Set the temperature.
    //------------------------------------------------------------------------------
    template<typename Dimension>
    void
    HelmholtzEquationOfState<Dimension>::
    setTemperature(Field<Dimension, Scalar>& temperature,
                   const Field<Dimension, Scalar>& massDensity,
                   const Field<Dimension, Scalar>& specificThermalEnergy) const {
        CHECK(valid());

        if(!fieldsStored) storeFields(temperature);
        
        int npart = massDensity.numElements();
        myMassDensity = massDensity;
        mySpecificThermalEnergy = specificThermalEnergy;
        
        if(needUpdate){
            Fortran2(wrapper_invert_helm_ed)(&npart, &myMassDensity[0], &mySpecificThermalEnergy[0],
                                             &myAbar[0], &myZbar[0], &myTemperature[0],
                                             &myPressure[0], &mTmin, &mySoundSpeed[0]);
        }

        for (size_t i = 0; i != massDensity.numElements(); ++i) {
            temperature(i) = myTemperature(i);
        }
    }

    //------------------------------------------------------------------------------
    // Set the specific thermal energy.
    //------------------------------------------------------------------------------
    template<typename Dimension>
    void
    HelmholtzEquationOfState<Dimension>::
    setSpecificThermalEnergy(Field<Dimension, Scalar>& specificThermalEnergy,
                             const Field<Dimension, Scalar>& massDensity,
                             const Field<Dimension, Scalar>& temperature) const {
        CHECK(valid());
        
        if(!fieldsStored) storeFields(specificThermalEnergy);
        
        int npart = massDensity.numElements();
        myMassDensity = massDensity;
        myTemperature = temperature;
        
        if(needUpdate){
            Fortran2(wrapper_helmeos)(&npart, &myMassDensity[0], &mySpecificThermalEnergy[0],
                                             &myAbar[0], &myZbar[0], &myTemperature[0],
                                             &myPressure[0]);
        }
        
        for (size_t i = 0; i != npart; ++i) {
            specificThermalEnergy(i) = mySpecificThermalEnergy(i);
        }
    }

    //------------------------------------------------------------------------------
    // Set the specific heat.
    //------------------------------------------------------------------------------
    template<typename Dimension>
    void
    HelmholtzEquationOfState<Dimension>::
    setSpecificHeat(Field<Dimension, Scalar>& specificHeat,
                    const Field<Dimension, Scalar>& massDensity,
                    const Field<Dimension, Scalar>& temperature) const {
        CHECK(valid());
        
        if(!fieldsStored) storeFields(specificHeat);
        
        const double kB = mConstants.kB();
        const double mp = mConstants.protonMass();
        int npart = myGamma.numElements();
        double Cv;
        
        for (size_t i = 0; i != npart; ++i)
            Cv += kB/(myGamma(i)*myAbar(i)*mp);
        specificHeat = Cv/npart;
    }

    //------------------------------------------------------------------------------
    // Set the sound speed.
    //------------------------------------------------------------------------------
    template<typename Dimension>
    void
    HelmholtzEquationOfState<Dimension>::
    setSoundSpeed(Field<Dimension, Scalar>& soundSpeed,
                  const Field<Dimension, Scalar>& massDensity,
                  const Field<Dimension, Scalar>& specificThermalEnergy) const {
        CHECK(valid());

        if(!fieldsStored) storeFields(soundSpeed);
        
        int npart = massDensity.numElements();
        myMassDensity = massDensity;
        mySpecificThermalEnergy = specificThermalEnergy;
        
        if(needUpdate){
            Fortran2(wrapper_invert_helm_ed)(&npart, &myMassDensity[0], &mySpecificThermalEnergy[0],
                                             &myAbar[0], &myZbar[0], &myTemperature[0],
                                             &myPressure[0], &mTmin, &mySoundSpeed[0]);
        }
        
        for (size_t i = 0; i != npart; ++i) {
            soundSpeed(i) = mySoundSpeed(i);
            myGamma(i) = soundSpeed(i) * soundSpeed(i) * massDensity(i) / myPressure(i);
        }
    }

    //------------------------------------------------------------------------------
    // Set the bulk modulus (rho DP/Drho).  This is just the pressure for a
    // Helmholtz gas.
    //------------------------------------------------------------------------------
    template<typename Dimension>
    void
    HelmholtzEquationOfState<Dimension>::
    setBulkModulus(Field<Dimension, Scalar>& bulkModulus,
                   const Field<Dimension, Scalar>& massDensity,
                   const Field<Dimension, Scalar>& specificThermalEnergy) const {
        CHECK(valid());
        
        if(!fieldsStored) storeFields(bulkModulus);
        
        setPressure(bulkModulus, massDensity, specificThermalEnergy);
    }

    /* ACCESSORS */
    template<typename Dimension>
    const bool
    HelmholtzEquationOfState<Dimension>::
    getUpdateStatus() const {
        return needUpdate;
    }

    template<typename Dimension>
    void
    HelmholtzEquationOfState<Dimension>::
    setUpdateStatus(bool bSet){
        needUpdate = bSet;
    }


    //------------------------------------------------------------------------------
    // Access abar
    //------------------------------------------------------------------------------
    template<typename Dimension>
    const FieldSpace::Field<Dimension, typename Dimension::Scalar>&
    HelmholtzEquationOfState<Dimension>::
    abar() const {
        //return mabar;
        return myAbar;
    }

    //------------------------------------------------------------------------------
    // Access zbar
    //------------------------------------------------------------------------------
    template<typename Dimension>
    const FieldSpace::Field<Dimension, typename Dimension::Scalar>&
    HelmholtzEquationOfState<Dimension>::
    zbar() const {
        //return mzbar;
        return myZbar;
    }
        
    //------------------------------------------------------------------------------
    // Determine if the EOS is in a valid state.
    //------------------------------------------------------------------------------
    template<typename Dimension>
    bool
    HelmholtzEquationOfState<Dimension>::valid() const {
        return (1.0);
    }
    
    //------------------------------------------------------------------------------
    // Store Fields to local memory
    //------------------------------------------------------------------------------
    template<typename Dimension>
    void
    HelmholtzEquationOfState::storeFields(Field<Dimension, Scalar>& thisField);
    {
        NodeList myNodeList     = thisField.nodeList();
        myMassDensity           = new Field<Dimension, Scalar>("helmMassDensity",myNodeList);
        mySpecificThermalEnergy = new Field<Dimension, Scalar>("helmSpecificThermalEnergy",myNodeList);
        myTemperature           = new Field<Dimension, Scalar>("helmTemperature",myNodeList);
        myPressure              = new Field<Dimension, Scalar>("helmPressure",myNodeList);
        mySoundSpeed            = new Field<Dimension, Scalar>("helmSoundSpeed",myNodeList);
        myGamma                 = new Field<Dimension, Scalar>("helmGamma",myNodeList);
        myAbar                  = new Field<Dimension, Scalar>("helmAbar",myNodeList,mabar0);
        myZbar                  = new Field<Dimension, Scalar>("helmZbar",myNodeList,mzbar0);
        
        fieldsStored = 1;
    }

}
}

