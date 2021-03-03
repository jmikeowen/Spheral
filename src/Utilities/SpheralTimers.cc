//---------------------------------Spheral++----------------------------------//
// SpheralTimers -- Declare the Timers for use in profiling Spheral
//
// Created by J. Michael Owen, Sat Dec  1 15:03:19 PST 2001
//----------------------------------------------------------------------------//

#include "Timer.hh"
#include <list>

// Must initialize the static list defined in Timer.hh
#ifdef TIMER
std::list<Timer*> Timer::TimerList(0); 
#endif

//------------------------------------------------------------------------------
// Root Timers
//------------------------------------------------------------------------------
Timer TIME_Spheral               ("Root Timer                ");
// Timer TimeNestedGridNeighbor    ("Root NestedGridNeighbor   ");
Timer TIME_Physics               ("All physics derivatives   ", TIME_Spheral);
// Timer TimeSphNodeList           ("Root Sph NodeList         ");
// Timer TimeMashNodeList          ("Root MASH NodeList        ");
// Timer TimeDataBase              ("Root DataBase             ");
// Timer TimeDistributedBound      ("Root DistributedBoundary  ");
// Timer TimeNestedDistributedBound("Root NestedDistribBound   ");

//------------------------------------------------------------------------------
// Voronoi
//------------------------------------------------------------------------------
Timer TIME_computeVoronoiVolume("computeVoronoiVolume", TIME_Spheral);

//------------------------------------------------------------------------------
// Polyhedron timers
//------------------------------------------------------------------------------
Timer TIME_Polyhedron_construct1       ("Polyhedron::Polyhedron(points)", TIME_Spheral);
Timer TIME_Polyhedron_construct2       ("Polyhedron::Polyhedron(points, facets)", TIME_Spheral);
Timer TIME_Polyhedron_BB               ("Polyhedron::setBoundingBox", TIME_Spheral);
Timer   TIME_Polyhedron_BB_ancillary   ("Polyhedron::setBoundingBox - computeAncillaryGeometry", TIME_Polyhedron_BB);
Timer   TIME_Polyhedron_BB_centroid    ("Polyhedron::setBoundingBox - centroid", TIME_Polyhedron_BB);
Timer   TIME_Polyhedron_BB_R2          ("Polyhedron::setBoundingBox - Rinterior2", TIME_Polyhedron_BB);
Timer TIME_Polyhedron_convex           ("Polyhedron::convex", TIME_Spheral);

//------------------------------------------------------------------------------
// PolyClipper timers
//------------------------------------------------------------------------------
Timer TIME_PC2d_convertto           ("Spheral::Polygon -> PolyClipper::Polygon", TIME_Spheral);
Timer TIME_PC2d_convertfrom         ("PolyClipper::Polygon -> Spheral::Polygon", TIME_Spheral);
Timer TIME_PC3d_convertto("Spheral::Polyhedron -> PolyClipper::Polyhedron", TIME_Spheral);
Timer TIME_PC3d_convertfrom("PolyClipper::Polyhedron -> Spheral::Polyhedron", TIME_Spheral);

//------------------------------------------------------------------------------
// ConnectivityMap
//------------------------------------------------------------------------------
Timer TIME_ConnectivityMap_patch("ConnectivityMap::patchConnectivity", TIME_Spheral);
Timer TIME_ConnectivityMap_cutConnectivity("ConnectivityMap::cutConnectivity", TIME_Spheral);
Timer TIME_ConnectivityMap_valid("ConnectivityMap::valid", TIME_Spheral);
Timer TIME_ConnectivityMap_computeConnectivity("ConnectivityMap::computeConnectivity", TIME_Spheral);
Timer TIME_ConnectivityMap_computeOverlapConnectivity("ConnectivityMap::computeOverlapConnectivity", TIME_ConnectivityMap_computeConnectivity);
Timer TIME_ConnectivityMap_computeIntersectionConnectivity("ConnectivityMap::intersectionConnectivity", TIME_Spheral);
Timer TIME_ConnectivityMap_precomputeIntersectionConnectivity("ConnectivityMap::precomputeIntersectionConnectivity", TIME_ConnectivityMap_computeConnectivity);

//------------------------------------------------------------------------------
// CRKSPH
//------------------------------------------------------------------------------
Timer TIME_CRKSPH_editMultimaterialSurfaceTopology("CRKSPH editMultimaterialSurfaceTopology", TIME_Spheral);
Timer TIME_interpolateCRKSPH("RK interpolation standalone function", TIME_Spheral);
Timer TIME_interpolateRK("RK interpolation standalone function", TIME_Spheral);

// //------------------------------------------------------------------------------
// // Second order predictor corrector integrator
// //------------------------------------------------------------------------------
// Timer TimePC2               ("Root Pred-Correct 2nd or", TIME_Spheral);
// Timer TimePC2Boundaries1    ("Set boundaries 1        ", TimePC2);
// Timer TimePC2Dt             ("Set dt                  ", TimePC2);
// Timer TimePC2StepInitialize1("Pre-step initialize 1   ", TimePC2);
// Timer TimePC2ReadDB         ("Read DataBase FieldLists", TimePC2);
// Timer TimePC2CopyFields     ("Copy initial Derivs     ", TimePC2);
// Timer TimePC2PredStep       ("Predictor step          ", TimePC2);
// Timer TimePC2Weights1       ("Update Weights 1        ", TimePC2);
// Timer TimePC2StepInitialize2("Pre-step initialize 2   ", TimePC2);
// Timer TimePC2Boundaries2    ("Set boundaries 2        ", TimePC2);
// Timer TimePC2Zero1          ("Zero derivs  1          ", TimePC2);
// Timer TimePC2CopyDvDx1      ("Copy DvDx to temp       ", TimePC2);
// Timer TimePC2EvalDerivs1    ("Eval derivs mid step    ", TimePC2);
// Timer TimePC2CorStep        ("Corrector step          ", TimePC2);
// Timer TimePC2CopyDvDx2      ("Copy DvDx to DB         ", TimePC2);
// Timer TimePC2Weights2       ("Update Weights 2        ", TimePC2);
// Timer TimePC2Boundaries3    ("Set boundaries 3        ", TimePC2);
// Timer TimePC2StepInitialize3("Pre-step initialize 3   ", TimePC2);
// Timer TimePC2Zero2          ("Zero derivs  2          ", TimePC2);
// Timer TimePC2EvalDerivs2    ("Eval derivs end step    ", TimePC2);
// Timer TimePC2CopyDvDx3      ("Copy DvDx to DB again   ", TimePC2);
// Timer TimePC2SumDensity     ("Sum end mass density    ", TimePC2);
// Timer TimePC2Weights3       ("Update Weights 3        ", TimePC2);

// //------------------------------------------------------------------------------
// // RK2 Synchronous integrator
// //------------------------------------------------------------------------------
// Timer TimeSyncRK2               ("Root Sync RK2Integrator ", TIME_Spheral);
// Timer TimeSyncRK2Boundaries1    ("Set boundaries 1        ", TimeSyncRK2);
// Timer TimeSyncRK2Dt             ("Set dt                  ", TimeSyncRK2);
// Timer TimeSyncRK2StepInitialize1("Pre-step initialize 1   ", TimeSyncRK2);
// Timer TimeSyncRK2ReadDB         ("Read DataBase FieldLists", TimeSyncRK2);
// Timer TimeSyncRK2CopyFields     ("Copy initial Derivs     ", TimeSyncRK2);
// Timer TimeSyncRK2MidStep        ("Advance to mid-step     ", TimeSyncRK2);
// Timer TimeSyncRK2Weights1       ("Update Weights 1        ", TimeSyncRK2);
// Timer TimeSyncRK2StepInitialize2("Pre-step initialize 2   ", TimeSyncRK2);
// Timer TimeSyncRK2Boundaries2    ("Set boundaries 2        ", TimeSyncRK2);
// Timer TimeSyncRK2Zero1          ("Zero derivs  1          ", TimeSyncRK2);
// Timer TimeSyncRK2CopyDvDx1      ("Copy DvDx to temp       ", TimeSyncRK2);
// Timer TimeSyncRK2EvalDerivs1    ("Eval derivs mid step    ", TimeSyncRK2);
// Timer TimeSyncRK2EndStep        ("Advance to end of step  ", TimeSyncRK2);
// Timer TimeSyncRK2CopyDvDx2      ("Copy DvDx to DB         ", TimeSyncRK2);
// Timer TimeSyncRK2Weights2       ("Update Weights 2        ", TimeSyncRK2);
// Timer TimeSyncRK2Boundaries3    ("Set boundaries 3        ", TimeSyncRK2);
// Timer TimeSyncRK2StepInitialize3("Pre-step initialize 3   ", TimeSyncRK2);
// Timer TimeSyncRK2Zero2          ("Zero derivs  2          ", TimeSyncRK2);
// Timer TimeSyncRK2EvalDerivs2    ("Eval derivs end step    ", TimeSyncRK2);
// Timer TimeSyncRK2CopyDvDx3      ("Copy DvDx to DB again   ", TimeSyncRK2);
// Timer TimeSyncRK2SumDensity     ("Sum end mass density    ", TimeSyncRK2);
// Timer TimeSyncRK2Weights3       ("Update Weights 3        ", TimeSyncRK2);

//------------------------------------------------------------------------------
// Cheap RK2 Synchronous integrator
//------------------------------------------------------------------------------
Timer TIME_CheapRK2               ("Root Cheap RK2Integrator", TIME_Spheral);
Timer TIME_CheapRK2PreInit        ("preStepIntialize        ", TIME_CheapRK2);
Timer TIME_CheapRK2Dt             ("Set dt                  ", TIME_CheapRK2);
Timer TIME_CheapRK2CopyState      ("Copy initial state      ", TIME_CheapRK2);
Timer TIME_CheapRK2MidStep        ("Advance to mid-step     ", TIME_CheapRK2);
Timer TIME_CheapRK2EvalDerivs     ("Eval derivs mid step    ", TIME_CheapRK2);
Timer TIME_CheapRK2EndStep        ("Advance to end of step  ", TIME_CheapRK2);
Timer TIME_CheapRK2Finalize       ("postStepFinalize        ", TIME_CheapRK2);
Timer TIME_CheapRK2EnforceBound   ("Enforce boundaries      ", TIME_CheapRK2);

//------------------------------------------------------------------------------
// Verlet (2nd order) integrator
//------------------------------------------------------------------------------
Timer TIME_Verlet                 ("Root Verlet Integrator  ", TIME_Spheral);
Timer TIME_VerletPreInit          ("preStepInitialize       ", TIME_Verlet);
Timer TIME_VerletCopyPos0         ("Copy initial position   ", TIME_Verlet);
Timer TIME_VerletDt               ("Set dt                  ", TIME_Verlet);
Timer TIME_VerletCopyState0       ("Copy initial state      ", TIME_Verlet);
Timer TIME_VerletEvalDerivs1      ("Eval initial derivs     ", TIME_Verlet);
Timer TIME_VerletPredict1         ("Predict mid-point state ", TIME_Verlet);
Timer TIME_VerletDtCheck          ("Check timestep validity ", TIME_Verlet);
Timer TIME_VerletMidPointCopy     ("Copy mid-point state    ", TIME_Verlet);
Timer TIME_VerletPredict2         ("Predict final state     ", TIME_Verlet);
Timer TIME_VerletEvalDerivs2      ("Eval final derivs       ", TIME_Verlet);
Timer TIME_VerletUpdateState      ("Correct final state     ", TIME_Verlet);
Timer TIME_VerletFinalize         ("postStepFinalize        ", TIME_Verlet);

// //------------------------------------------------------------------------------
// // NestedGridNeighbor
// //------------------------------------------------------------------------------
// Timer TimeNestedMaster          ("Set the master list (node)", TimeNestedGridNeighbor);
// Timer TimeNestedRefine          ("Set the refine list (node)", TimeNestedGridNeighbor);
// Timer TimeNestedMasterPlane     ("Set the master list(plane)", TimeNestedGridNeighbor);
// Timer TimeNestedUpdateNodes     ("Update the node info      ", TimeNestedGridNeighbor);
// Timer TimeNestedUpdateNodes1    ("Update the given node info", TimeNestedGridNeighbor);
// Timer TimeNestedPrecull         ("Precull nodes             ", TimeNestedRefine);

// //------------------------------------------------------------------------------
// // Hydro
// //------------------------------------------------------------------------------
// Timer TimeHydro                 ("Root Hydro                ", TimePhysics);
// Timer TimeHydroFlagNodes        ("Initialize flag nodes     ", TimeHydro);
// Timer TimeHydroReadState        ("Read the initial state    ", TimeHydro);
// Timer TimeHydroSetMaster        ("Set master nodes          ", TimeHydro);
// Timer TimeHydroSetRefine        ("Set the refine nodes      ", TimeHydro);
// Timer TimeHydroEvalDerivs       ("Set derivs for node       ", TimeHydro);

//------------------------------------------------------------------------------
// SPH 
//------------------------------------------------------------------------------
Timer TIME_SPH                   ("SPH base timer                 ", TIME_Physics);
Timer TIME_SPHinitializeStartup  ("SPH initializeProblemStartup   ", TIME_SPH);
Timer TIME_SPHregister           ("SPH register                   ", TIME_SPH);
Timer TIME_SPHregisterDerivs     ("SPH registerDerivatives        ", TIME_SPH);
Timer TIME_SPHpreStepInitialize  ("SPH preStepInitialize (step)   ", TIME_SPH);
Timer TIME_SPHinitialize         ("SPH initialize (evalderivs)    ", TIME_SPH);
Timer TIME_SPHfinalizeDerivs     ("SPH finalizeDerivatives        ", TIME_SPH);
Timer TIME_SPHghostBounds        ("SPH ghost boundaries           ", TIME_SPH);
Timer TIME_SPHupdateVol          ("SPH updateVolume               ", TIME_SPH);
Timer TIME_SPHenforceBounds      ("SPH enforceBoundaries          ", TIME_SPH);
Timer TIME_SPHevalDerivs         ("SPH evaluateDerivates          ", TIME_SPH);
Timer TIME_SPHevalDerivs_initial ("SPH evaluateDerivates (initial)", TIME_SPHevalDerivs);
Timer TIME_SPHevalDerivs_pairs   ("SPH evaluateDerivates (pairs)  ", TIME_SPHevalDerivs);
Timer TIME_SPHevalDerivs_final   ("SPH evaluateDerivates (final)  ", TIME_SPHevalDerivs);

//------------------------------------------------------------------------------
// SolidSPH 
//------------------------------------------------------------------------------
Timer TIME_SolidSPH                   ("SolidSPH base timer                 ", TIME_Physics);
Timer TIME_SolidSPHinitializeStartup  ("SolidSPH initializeProblemStartup   ", TIME_SolidSPH);
Timer TIME_SolidSPHregister           ("SolidSPH register                   ", TIME_SolidSPH);
Timer TIME_SolidSPHregisterDerivs     ("SolidSPH registerDerivatives        ", TIME_SolidSPH);
Timer TIME_SolidSPHpreStepInitialize  ("SolidSPH preStepInitialize (step)   ", TIME_SolidSPH);
Timer TIME_SolidSPHinitialize         ("SolidSPH initialize (evalderivs)    ", TIME_SolidSPH);
Timer TIME_SolidSPHfinalizeDerivs     ("SolidSPH finalizeDerivatives        ", TIME_SolidSPH);
Timer TIME_SolidSPHghostBounds        ("SolidSPH ghost boundaries           ", TIME_SolidSPH);
Timer TIME_SolidSPHupdateVol          ("SolidSPH updateVolume               ", TIME_SolidSPH);
Timer TIME_SolidSPHenforceBounds      ("SolidSPH enforceBoundaries          ", TIME_SolidSPH);
Timer TIME_SolidSPHevalDerivs         ("SolidSPH evaluateDerivates          ", TIME_SolidSPH);
Timer TIME_SolidSPHevalDerivs_initial ("SolidSPH evaluateDerivates (initial)", TIME_SolidSPHevalDerivs);
Timer TIME_SolidSPHevalDerivs_pairs   ("SolidSPH evaluateDerivates (pairs)  ", TIME_SolidSPHevalDerivs);
Timer TIME_SolidSPHevalDerivs_final   ("SolidSPH evaluateDerivates (final)  ", TIME_SolidSPHevalDerivs);

//------------------------------------------------------------------------------
// Damage
//------------------------------------------------------------------------------
Timer TIME_Damage                       ("Damage base timer                 ", TIME_Physics);
Timer TIME_DamageModel_finalize         ("DamageModel::finalize             ", TIME_Damage);
Timer TIME_ThreePointCoupling           ("3-pt damage coupling constructor  ", TIME_Damage);
Timer TIME_ThreePointCoupling_initial   ("3-pt damage compute flags         ", TIME_Damage);
Timer TIME_ThreePointCoupling_pairs     ("3-pt damage loop over pairs       ", TIME_Damage);
Timer TIME_DamageGradientCoupling       ("damage gradient coupling ctor     ", TIME_Damage);
Timer TIME_DamageGradientCoupling_grad  ("damage gradient compute grad      ", TIME_Damage);
Timer TIME_DamageGradientCoupling_pairs ("damage gradient loop over pairs   ", TIME_Damage);

// //------------------------------------------------------------------------------
// // MASH NodeList
// //------------------------------------------------------------------------------
// Timer TimeMashDerivs            ("Base Mash Derivs          ", TimeMashNodeList);
// Timer TimeMashNodeIState        ("Get state for node i      ", TimeMashDerivs);
// Timer TimeMashNodeJState        ("Get state for node j      ", TimeMashDerivs);
// Timer TimeMashRij               ("Calc rij, etaij, ...      ", TimeMashDerivs);
// Timer TimeMashKernel            ("Evaluate Wij              ", TimeMashDerivs);
// Timer TimeMashQ                 ("Evaluate Qij              ", TimeMashDerivs);
// Timer TimeMashIncDerivs         ("Increment the derivs      ", TimeMashDerivs);
// Timer TimeMashHDeriv            ("Evaluate the H deriv      ", TimeMashDerivs);
// Timer TimeMashHSmooth           ("Smoothing contrib to H    ", TimeMashDerivs);

// //------------------------------------------------------------------------------
// // DataBase
// //------------------------------------------------------------------------------
// Timer TimeDataBaseRho           ("Sum fluid density         ", TimeDataBase);
// Timer TimeDataBaseGetState      ("Read state FieldLists     ", TimeDataBaseRho);
// Timer TimeDataBaseInitFlags     ("Init node done flags      ", TimeDataBaseRho);
// Timer TimeDataBaseSetMaster     ("Set master nodes          ", TimeDataBaseRho);
// Timer TimeDataBaseSetRefine     ("Set refine nodes          ", TimeDataBaseRho);
// Timer TimeDataBaseSumNode       ("Sum rho for single node   ", TimeDataBaseRho);
// Timer TimeDataBasePostConditions("Verify post conditions    ", TimeDataBaseRho);

// //------------------------------------------------------------------------------
// // DistributedBoundary
// //------------------------------------------------------------------------------
// Timer TimeDBExchange            ("Generic Field exchange    ", TimeDistributedBound);

// //------------------------------------------------------------------------------
// // NestedGridDistributedBoundary
// //------------------------------------------------------------------------------
// Timer TimeNDSetGhost            ("Set ghost nodes           ", TimeNestedDistributedBound);
// Timer TimeNDReset               ("Clear existing info       ", TimeNDSetGhost);
// Timer TimeNDFlatten             ("Flatten occupied gridcells", TimeNDSetGhost);
// Timer TimeNDReduceGridCells     ("Reduce occupied gridcells ", TimeNDSetGhost);
// Timer TimeNDBuildCommMaps       ("Build comm maps on master ", TimeNDSetGhost);
// Timer TimeNDDistributeCommMaps  ("Distribute comm maps      ", TimeNDSetGhost);
// Timer TimeNDBuildGhost          ("Build ghost nodes         ", TimeNDSetGhost);
// Timer TimeNDExchangeMinimal     ("Exchange r, H             ", TimeNDSetGhost);

// //------------------------------------------------------------------------------
// // Golden rule grad div vector field.
// //------------------------------------------------------------------------------
// Timer TimeGoldenSecond          ("Golden rule grad div      ");
// Timer TimeGoldenSecondReserve   ("Reserve return Fields     ", TimeGoldenSecond);
// Timer TimeGoldenSecondMaster    ("Select master nodes       ", TimeGoldenSecond);
// Timer TimeGoldenSecondRefine    ("Select refine nodes       ", TimeGoldenSecond);
// Timer TimeGoldenSecondNodeI     ("State for node I          ", TimeGoldenSecond);
// Timer TimeGoldenSecondNodeJ     ("State for node J          ", TimeGoldenSecond);
// Timer TimeGoldenSecondRefineLoop("Main loop over refine nei ", TimeGoldenSecond);
// Timer TimeGoldenSecondW         ("Calculate W, gradW, grad2W", TimeGoldenSecond);
// Timer TimeGoldenSecondAccumulate("Accumulate components     ", TimeGoldenSecond);
// Timer TimeGoldenSecondFinalize  ("Determine final answer    ", TimeGoldenSecond);
// Timer TimeGoldenSecondCheck     ("Completeness check        ", TimeGoldenSecond);
