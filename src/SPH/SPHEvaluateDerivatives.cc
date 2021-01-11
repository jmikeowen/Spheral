namespace Spheral {

//------------------------------------------------------------------------------
// Determine the principle derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SPHHydroBase<Dimension>::
evaluateDerivatives(const typename Dimension::Scalar /*time*/,
                    const typename Dimension::Scalar /*dt*/,
                    const DataBase<Dimension>& dataBase,
                    const State<Dimension>& state,
                    StateDerivatives<Dimension>& derivatives) const {
  TIME_SPHevalDerivs.start();
  TIME_SPHevalDerivs_initial.start();

#ifdef ENABLE_OPENMP
  using PAIR_EXEC_POL = RAJA::omp_parallel_for_exec;
  using PAIR_REDUCE_POL = RAJA::omp_reduce;
  using NODE_INNER_EXEC_POL = RAJA::omp_for_exec;
#else
  using PAIR_EXEC_POL = RAJA::seq_exec;
  using PAIR_REDUCE_POL = RAJA::seq_reduce;
  using NODE_INNER_EXEC_POL = RAJA::seq_exec;
#endif
  using NODE_OUTER_EXEC_POL = RAJA::seq_exec;

  //static double totalLoopTime = 0.0;
  // Get the ArtificialViscosity.
  auto& Q = this->artificialViscosity();

  // The kernels and such.
  const auto& W = this->kernel();
  const auto& WQ = this->PiKernel();

  // A few useful constants we'll use in the following loop.
  const double tiny = 1.0e-30;
  const Scalar W0 = W(0.0, 1.0);

  // The connectivity.
  const auto& connectivityMap = dataBase.connectivityMap();
  const auto& nodeLists = connectivityMap.nodeLists();
  const auto numNodeLists = nodeLists.size();

  // Get the state and derivative FieldLists.
  // State FieldLists.
  const auto mass = state.fields(HydroFieldNames::mass, 0.0);
  const auto position = state.fields(HydroFieldNames::position, Vector::zero);
  const auto velocity = state.fields(HydroFieldNames::velocity, Vector::zero);
  const auto massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  const auto H = state.fields(HydroFieldNames::H, SymTensor::zero);
  const auto pressure = state.fields(HydroFieldNames::pressure, 0.0);
  const auto soundSpeed = state.fields(HydroFieldNames::soundSpeed, 0.0);
  const auto omega = state.fields(HydroFieldNames::omegaGradh, 0.0);
  CHECK(mass.size() == numNodeLists);
  CHECK(position.size() == numNodeLists);
  CHECK(velocity.size() == numNodeLists);
  CHECK(massDensity.size() == numNodeLists);
  CHECK(H.size() == numNodeLists);
  CHECK(pressure.size() == numNodeLists);
  CHECK(soundSpeed.size() == numNodeLists);
  CHECK(omega.size() == numNodeLists);

  // Derivative FieldLists.
  auto  rhoSum = derivatives.fields(ReplaceFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
  auto  normalization = derivatives.fields(HydroFieldNames::normalization, 0.0);
  auto  DxDt = derivatives.fields(IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position, Vector::zero);
  auto  DrhoDt = derivatives.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
  auto  DvDt = derivatives.fields(HydroFieldNames::hydroAcceleration, Vector::zero);
  auto  DepsDt = derivatives.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::specificThermalEnergy, 0.0);
  auto  DvDx = derivatives.fields(HydroFieldNames::velocityGradient, Tensor::zero);
  auto  localDvDx = derivatives.fields(HydroFieldNames::internalVelocityGradient, Tensor::zero);
  auto  M = derivatives.fields(HydroFieldNames::M_SPHCorrection, Tensor::zero);
  auto  localM = derivatives.fields("local " + HydroFieldNames::M_SPHCorrection, Tensor::zero);
  auto  DHDt = derivatives.fields(IncrementFieldList<Dimension, SymTensor>::prefix() + HydroFieldNames::H, SymTensor::zero);
  auto  Hideal = derivatives.fields(ReplaceBoundedFieldList<Dimension, SymTensor>::prefix() + HydroFieldNames::H, SymTensor::zero);
  auto  maxViscousPressure = derivatives.fields(HydroFieldNames::maxViscousPressure, 0.0);
  auto  effViscousPressure = derivatives.fields(HydroFieldNames::effectiveViscousPressure, 0.0);
  auto  viscousWork = derivatives.fields(HydroFieldNames::viscousWork, 0.0);
  auto& pairAccelerations = derivatives.getAny(HydroFieldNames::pairAccelerations, vector<Vector>());
  auto  XSPHWeightSum = derivatives.fields(HydroFieldNames::XSPHWeightSum, 0.0);
  auto  XSPHDeltaV = derivatives.fields(HydroFieldNames::XSPHDeltaV, Vector::zero);
  auto  weightedNeighborSum = derivatives.fields(HydroFieldNames::weightedNeighborSum, 0.0);
  auto  massSecondMoment = derivatives.fields(HydroFieldNames::massSecondMoment, SymTensor::zero);
  CHECK(rhoSum.size() == numNodeLists);
  CHECK(normalization.size() == numNodeLists);
  CHECK(DxDt.size() == numNodeLists);
  CHECK(DrhoDt.size() == numNodeLists);
  CHECK(DvDt.size() == numNodeLists);
  CHECK(DepsDt.size() == numNodeLists);
  CHECK(DvDx.size() == numNodeLists);
  CHECK(localDvDx.size() == numNodeLists);
  CHECK(M.size() == numNodeLists);
  CHECK(localM.size() == numNodeLists);
  CHECK(DHDt.size() == numNodeLists);
  CHECK(Hideal.size() == numNodeLists);
  CHECK(maxViscousPressure.size() == numNodeLists);
  CHECK(effViscousPressure.size() == numNodeLists);
  CHECK(viscousWork.size() == numNodeLists);
  CHECK(XSPHWeightSum.size() == numNodeLists);
  CHECK(XSPHDeltaV.size() == numNodeLists);
  CHECK(weightedNeighborSum.size() == numNodeLists);
  CHECK(massSecondMoment.size() == numNodeLists);

  // The set of interacting node pairs.
  const auto& pairs = connectivityMap.nodePairList();
  const auto  npairs = pairs.size();

  // Size up the pair-wise accelerations before we start.
  if (mCompatibleEnergyEvolution) pairAccelerations.resize(npairs);

  // The scale for the tensile correction.
  const auto& nodeList = mass[0]->nodeList();
  const auto  nPerh = nodeList.nodesPerSmoothingScale();
  const auto  WnPerh = W(1.0/nPerh, 1.0);

  // Get RAJA::Reducer arrays. 
  auto DvDt_reducer = DvDt.getReduceSum(PAIR_REDUCE_POL());
  auto rhoSum_reducer = rhoSum.getReduceSum(PAIR_REDUCE_POL());
  auto normalization_reducer = normalization.getReduceSum(PAIR_REDUCE_POL());
  auto DepsDt_reducer = DepsDt.getReduceSum(PAIR_REDUCE_POL());
  auto DvDx_reducer = DvDx.getReduceSum(PAIR_REDUCE_POL());
  auto localDvDx_reducer = localDvDx.getReduceSum(PAIR_REDUCE_POL());
  auto M_reducer = M.getReduceSum(PAIR_REDUCE_POL());
  auto localM_reducer = localM.getReduceSum(PAIR_REDUCE_POL());
  auto maxViscousPressure_reducer = maxViscousPressure.getReduceMax(PAIR_REDUCE_POL());
  auto effViscousPressure_reducer = effViscousPressure.getReduceSum(PAIR_REDUCE_POL());
  //auto rhoSumCorrection_reducer = rhoSumCorrection.getReduceSum(PAIR_REDUCE_POL());
  auto viscousWork_reducer = viscousWork.getReduceSum(PAIR_REDUCE_POL());
  auto XSPHWeightSum_reducer = XSPHWeightSum.getReduceSum(PAIR_REDUCE_POL());
  auto XSPHDeltaV_reducer = XSPHDeltaV.getReduceSum(PAIR_REDUCE_POL());
  auto weightedNeighborSum_reducer = weightedNeighborSum.getReduceSum(PAIR_REDUCE_POL());
  auto massSecondMoment_reducer = massSecondMoment.getReduceSum(PAIR_REDUCE_POL());
  //auto DSDt_reducer = DSDt.getReduceSum(PAIR_REDUCE_POL());

  TIME_SPHevalDerivs_initial.stop();

  // Walk all the interacting pairs.
  TIME_SPHevalDerivs_pairs.start();
  RAJA::TypedRangeSegment<unsigned int> array_npairs(0, npairs);
  RAJA::forall<PAIR_EXEC_POL>(array_npairs, [&](unsigned int kk) {
      //
    // Thread private scratch variables
    int i, j, nodeListi, nodeListj;
    Scalar Wi, gWi, WQi, gWQi, Wj, gWj, WQj, gWQj;
    Tensor QPiij, QPiji;

    //typename SpheralThreads<Dimension>::FieldListStack threadStack;
    //auto rhoSum_thread = rhoSum.threadCopy(threadStack);
    //auto normalization_thread = normalization.threadCopy(threadStack);
    //auto DvDt_thread = DvDt.threadCopy(threadStack);
    //auto DepsDt_thread = DepsDt.threadCopy(threadStack);
    //auto DvDx_thread = DvDx.threadCopy(threadStack);
    //auto localDvDx_thread = localDvDx.threadCopy(threadStack);
    //auto M_thread = M.threadCopy(threadStack);
    //auto localM_thread = localM.threadCopy(threadStack);
    //auto maxViscousPressure_thread = maxViscousPressure.threadCopy(threadStack, ThreadReduction::MAX);
    //auto effViscousPressure_thread = effViscousPressure.threadCopy(threadStack);
    //auto viscousWork_thread = viscousWork.threadCopy(threadStack);
    //auto XSPHWeightSum_thread = XSPHWeightSum.threadCopy(threadStack);
    //auto XSPHDeltaV_thread = XSPHDeltaV.threadCopy(threadStack);
    //auto weightedNeighborSum_thread = weightedNeighborSum.threadCopy(threadStack);
    //auto massSecondMoment_thread = massSecondMoment.threadCopy(threadStack);

      i = pairs[kk].i_node;
      j = pairs[kk].j_node;
      nodeListi = pairs[kk].i_list;
      nodeListj = pairs[kk].j_list;

      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto& mi = mass(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto& rhoi = massDensity(nodeListi, i);
      const auto& Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto& ci = soundSpeed(nodeListi, i);
      const auto& omegai = omega(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      const auto  safeOmegai = safeInv(omegai, tiny);
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& rhoSumi = rhoSum_reducer[nodeListi][i];
      auto& normi = normalization_reducer[nodeListi][i];
      auto& DvDti = DvDt_reducer[nodeListi][i];
      auto& DepsDti = DepsDt_reducer[nodeListi][i];
      auto& DvDxi = DvDx_reducer[nodeListi][i];
      auto& localDvDxi = localDvDx_reducer[nodeListi][i];
      auto& Mi = M_reducer[nodeListi][i];
      auto& localMi = localM_reducer[nodeListi][i];
      auto& maxViscousPressurei = maxViscousPressure_reducer[nodeListi][i];
      auto& effViscousPressurei = effViscousPressure_reducer[nodeListi][i];
      //auto& rhoSumCorrectioni = rhoSumCorrection_reducer[nodeListi][i];
      auto& viscousWorki = viscousWork_reducer[nodeListi][i];
      auto& XSPHWeightSumi = XSPHWeightSum_reducer[nodeListi][i];
      auto& XSPHDeltaVi = XSPHDeltaV_reducer[nodeListi][i];
      auto& weightedNeighborSumi = weightedNeighborSum_reducer[nodeListi][i];
      auto& massSecondMomenti = massSecondMoment_reducer[nodeListi][i];
      //auto& rhoSumi = rhoSum_thread(nodeListi, i);
      //auto& normi = normalization_thread(nodeListi, i);
      //auto& DvDti = DvDt_thread(nodeListi, i);
      //auto& DepsDti = DepsDt_thread(nodeListi, i);
      //auto& DvDxi = DvDx_thread(nodeListi, i);
      //auto& localDvDxi = localDvDx_thread(nodeListi, i);
      //auto& Mi = M_thread(nodeListi, i);
      //auto& localMi = localM_thread(nodeListi, i);
      //auto& maxViscousPressurei = maxViscousPressure_thread(nodeListi, i);
      //auto& effViscousPressurei = effViscousPressure_thread(nodeListi, i);
      //auto& viscousWorki = viscousWork_thread(nodeListi, i);
      //auto& XSPHWeightSumi = XSPHWeightSum_thread(nodeListi, i);
      //auto& XSPHDeltaVi = XSPHDeltaV_thread(nodeListi, i);
      //auto& weightedNeighborSumi = weightedNeighborSum_thread(nodeListi, i);
      //auto& massSecondMomenti = massSecondMoment_thread(nodeListi, i);

      // Get the state for node j
      const auto& rj = position(nodeListj, j);
      const auto& mj = mass(nodeListj, j);
      const auto& vj = velocity(nodeListj, j);
      const auto& rhoj = massDensity(nodeListj, j);
      const auto& Pj = pressure(nodeListj, j);
      const auto& Hj = H(nodeListj, j);
      const auto& cj = soundSpeed(nodeListj, j);
      const auto& omegaj = omega(nodeListj, j);
      const auto  Hdetj = Hj.Determinant();
      const auto  safeOmegaj = safeInv(omegaj, tiny);
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& rhoSumj = rhoSum_reducer[nodeListj][j];
      auto& normj = normalization_reducer[nodeListi][i];
      auto& DvDtj = DvDt_reducer[nodeListj][j];
      auto& DepsDtj = DepsDt_reducer.at(nodeListj).at(j);
      auto& DvDxj = DvDx_reducer[nodeListj][j];
      auto& localDvDxj = localDvDx_reducer[nodeListj][j];
      auto& Mj = M_reducer[nodeListj][j];
      auto& localMj = localM_reducer[nodeListj][j];
      auto& maxViscousPressurej = maxViscousPressure_reducer[nodeListj][j];
      auto& effViscousPressurej = effViscousPressure_reducer[nodeListj][j];
      //auto& rhoSumCorrectionj = rhoSumCorrection_reducer[nodeListj][j];
      auto& viscousWorkj = viscousWork_reducer[nodeListj][j];
      auto& XSPHWeightSumj = XSPHWeightSum_reducer[nodeListj][j];
      auto& XSPHDeltaVj = XSPHDeltaV_reducer[nodeListj][j];
      auto& weightedNeighborSumj = weightedNeighborSum_reducer[nodeListj][j];
      auto& massSecondMomentj = massSecondMoment_reducer[nodeListj][j];
      //auto& rhoSumj = rhoSum_thread(nodeListj, j);
      //auto& normj = normalization_thread(nodeListj, j);
      //auto& DvDtj = DvDt_thread(nodeListj, j);
      //auto& DepsDtj = DepsDt_thread(nodeListj, j);
      //auto& DvDxj = DvDx_thread(nodeListj, j);
      //auto& localDvDxj = localDvDx_thread(nodeListj, j);
      //auto& Mj = M_thread(nodeListj, j);
      //auto& localMj = localM_thread(nodeListj, j);
      //auto& maxViscousPressurej = maxViscousPressure_thread(nodeListj, j);
      //auto& effViscousPressurej = effViscousPressure_thread(nodeListj, j);
      //auto& viscousWorkj = viscousWork_thread(nodeListj, j);
      //auto& XSPHWeightSumj = XSPHWeightSum_thread(nodeListj, j);
      //auto& XSPHDeltaVj = XSPHDeltaV_thread(nodeListj, j);
      //auto& weightedNeighborSumj = weightedNeighborSum_thread(nodeListj, j);
      //auto& massSecondMomentj = massSecondMoment_thread(nodeListj, j);

      // Flag if this is a contiguous material pair or not.
      const bool sameMatij = true; // (nodeListi == nodeListj and fragIDi == fragIDj);

      // Node displacement.
      const auto rij = ri - rj;
      const auto etai = Hi*rij;
      const auto etaj = Hj*rij;
      const auto etaMagi = etai.magnitude();
      const auto etaMagj = etaj.magnitude();
      CHECK(etaMagi >= 0.0);
      CHECK(etaMagj >= 0.0);

      // Symmetrized kernel weight and gradient.
      std::tie(Wi, gWi) = W.kernelAndGradValue(etaMagi, Hdeti);
      std::tie(WQi, gWQi) = WQ.kernelAndGradValue(etaMagi, Hdeti);
      const auto Hetai = Hi*etai.unitVector();
      const auto gradWi = gWi*Hetai;
      const auto gradWQi = gWQi*Hetai;

      std::tie(Wj, gWj) = W.kernelAndGradValue(etaMagj, Hdetj);
      std::tie(WQj, gWQj) = WQ.kernelAndGradValue(etaMagj, Hdetj);
      const auto Hetaj = Hj*etaj.unitVector();
      const auto gradWj = gWj*Hetaj;
      const auto gradWQj = gWQj*Hetaj;

      // Zero'th and second moment of the node distribution -- used for the
      // ideal H calculation.
      const auto fweightij = sameMatij ? 1.0 : mj*rhoi/(mi*rhoj);
      const auto rij2 = rij.magnitude2();
      const auto thpt = rij.selfdyad()*safeInvVar(rij2*rij2*rij2);
      weightedNeighborSumi +=     fweightij*std::abs(gWi);
      weightedNeighborSumj += 1.0/fweightij*std::abs(gWj);
      massSecondMomenti +=     fweightij*gradWi.magnitude2()*thpt;
      massSecondMomentj += 1.0/fweightij*gradWj.magnitude2()*thpt;

      // Contribution to the sum density.
      if (nodeListi == nodeListj) {
        rhoSumi += mj*Wi;
        rhoSumj += mi*Wj;
        normi += mi/rhoi*Wi;
        normj += mj/rhoj*Wj;
      }

      // Compute the pair-wise artificial viscosity.
      const auto vij = vi - vj;
      std::tie(QPiij, QPiji) = Q.Piij(nodeListi, i, nodeListj, j,
                                      ri, etai, vi, rhoi, ci, Hi,
                                      rj, etaj, vj, rhoj, cj, Hj);
      const auto Qacci = 0.5*(QPiij*gradWQi);
      const auto Qaccj = 0.5*(QPiji*gradWQj);
      // const auto workQi = 0.5*(QPiij*vij).dot(gradWQi);
      // const auto workQj = 0.5*(QPiji*vij).dot(gradWQj);
      const auto workQi = vij.dot(Qacci);
      const auto workQj = vij.dot(Qaccj);
      const auto Qi = rhoi*rhoi*(QPiij.diagonalElements().maxAbsElement());
      const auto Qj = rhoj*rhoj*(QPiji.diagonalElements().maxAbsElement());
      maxViscousPressurei.max(Qi);
      maxViscousPressurej.max(Qj);
      //maxViscousPressurei = max(maxViscousPressurei, Qi);
      //maxViscousPressurej = max(maxViscousPressurej, Qj);
      effViscousPressurei += mj*Qi*WQi/rhoj;
      effViscousPressurej += mi*Qj*WQj/rhoi;
      viscousWorki += mj*workQi;
      viscousWorkj += mi*workQj;

      // Determine an effective pressure including a term to fight the tensile instability.
//             const auto fij = epsTensile*pow(Wi/(Hdeti*WnPerh), nTensile);
      const auto fij = mEpsTensile*FastMath::pow4(Wi/(Hdeti*WnPerh));
      const auto Ri = fij*(Pi < 0.0 ? -Pi : 0.0);
      const auto Rj = fij*(Pj < 0.0 ? -Pj : 0.0);
      const auto Peffi = Pi + Ri;
      const auto Peffj = Pj + Rj;

      // Acceleration.
      CHECK(rhoi > 0.0);
      CHECK(rhoj > 0.0);
      const auto Prhoi = safeOmegai*Peffi/(rhoi*rhoi);
      const auto Prhoj = safeOmegaj*Peffj/(rhoj*rhoj);
      const auto deltaDvDt = Prhoi*gradWi + Prhoj*gradWj + Qacci + Qaccj;
      DvDti += -mj*deltaDvDt;
      DvDtj += mi*deltaDvDt;
      if (mCompatibleEnergyEvolution) pairAccelerations[kk] = -mj*deltaDvDt;  // Acceleration for i (j anti-symmetric)

      // Specific thermal energy evolution.
      // const Scalar workQij = 0.5*(mj*workQi + mi*workQj);
      DepsDti += mj*(Prhoi*vij.dot(gradWi) + workQi);
      DepsDtj += mi*(Prhoj*vij.dot(gradWj) + workQj);

      // Velocity gradient.
      const auto deltaDvDxi = mj*vij.dyad(gradWi);
      const auto deltaDvDxj = mi*vij.dyad(gradWj);
      DvDxi += -deltaDvDxi; 
      DvDxj += -deltaDvDxj;
      if (sameMatij) {
        localDvDxi += -deltaDvDxi; 
        localDvDxj += -deltaDvDxj;
      }

      // Estimate of delta v (for XSPH).
      if (mXSPH and (sameMatij)) {
        const auto wXSPHij = 0.5*(mi/rhoi*Wi + mj/rhoj*Wj);
        XSPHWeightSumi += wXSPHij;
        XSPHWeightSumj += wXSPHij;
        XSPHDeltaVi += -wXSPHij*vij;
        XSPHDeltaVj += wXSPHij*vij;
      }

      // Linear gradient correction term.
      Mi += -mj*rij.dyad(gradWi);
      Mj += -mi*rij.dyad(gradWj);
      if (sameMatij) {
        localMi += -mj*rij.dyad(gradWi);
        localMj += -mi*rij.dyad(gradWj);
      }

    // Reduce the thread values to the master.
    //threadReduceFieldLists<Dimension>(threadStack);

  });   // OpenMP parallel region
  TIME_SPHevalDerivs_pairs.stop();

  TIME_SPHevalDerivs_final.start();

  rhoSum.getReduction(rhoSum_reducer);
  DvDt.getReduction(DvDt_reducer);
  DepsDt.getReduction(DepsDt_reducer);
  DvDx.getReduction(DvDx_reducer);
  localDvDx.getReduction(localDvDx_reducer);
  M.getReduction(M_reducer);
  localM.getReduction(localM_reducer);
  maxViscousPressure.getReduction(maxViscousPressure_reducer);
  effViscousPressure.getReduction(effViscousPressure_reducer);
  //rhoSumCorrection.getReduction(rhoSumCorrection_reducer);
  viscousWork.getReduction(viscousWork_reducer);
  XSPHWeightSum.getReduction(XSPHWeightSum_reducer);
  XSPHDeltaV.getReduction(XSPHDeltaV_reducer);
  weightedNeighborSum.getReduction(weightedNeighborSum_reducer);
  massSecondMoment.getReduction(massSecondMoment_reducer);

  // Finish up the derivatives for each point.
  for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    const auto& nodeList = mass[nodeListi]->nodeList();
    const auto  hmin = nodeList.hmin();
    const auto  hmax = nodeList.hmax();
    const auto  hminratio = nodeList.hminratio();
    const auto  nPerh = nodeList.nodesPerSmoothingScale();

    const auto ni = nodeList.numInternalNodes();
#pragma omp parallel for
    for (auto i = 0u; i < ni; ++i) {

      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto& mi = mass(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto& rhoi = massDensity(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      const auto  numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& rhoSumi = rhoSum(nodeListi, i);
      auto& normi = normalization(nodeListi, i);
      auto& DxDti = DxDt(nodeListi, i);
      auto& DrhoDti = DrhoDt(nodeListi, i);
      auto& DvDti = DvDt(nodeListi, i);
      auto& DepsDti = DepsDt(nodeListi, i);
      auto& DvDxi = DvDx(nodeListi, i);
      auto& localDvDxi = localDvDx(nodeListi, i);
      auto& Mi = M(nodeListi, i);
      auto& localMi = localM(nodeListi, i);
      auto& DHDti = DHDt(nodeListi, i);
      auto& Hideali = Hideal(nodeListi, i);
      auto& XSPHWeightSumi = XSPHWeightSum(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment(nodeListi, i);

      // Add the self-contribution to density sum.
      rhoSumi += mi*W0*Hdeti;
      normi += mi/rhoi*W0*Hdeti;

      // Finish the gradient of the velocity.
      CHECK(rhoi > 0.0);
      if (this->mCorrectVelocityGradient and
          std::abs(Mi.Determinant()) > 1.0e-10 and
          numNeighborsi > Dimension::pownu(2)) {
        Mi = Mi.Inverse();
        DvDxi = DvDxi*Mi;
      } else {
        DvDxi /= rhoi;
      }
      if (this->mCorrectVelocityGradient and
          std::abs(localMi.Determinant()) > 1.0e-10 and
          numNeighborsi > Dimension::pownu(2)) {
        localMi = localMi.Inverse();
        localDvDxi = localDvDxi*localMi;
      } else {
        localDvDxi /= rhoi;
      }

      // Evaluate the continuity equation.
      DrhoDti = -rhoi*DvDxi.Trace();

      // If needed finish the total energy derivative.
      if (mEvolveTotalEnergy) DepsDti = mi*(vi.dot(DvDti) + DepsDti);

      // Complete the moments of the node distribution for use in the ideal H calculation.
      weightedNeighborSumi = Dimension::rootnu(max(0.0, weightedNeighborSumi/Hdeti));
      massSecondMomenti /= Hdeti*Hdeti;

      // Determine the position evolution, based on whether we're doing XSPH or not.
      if (mXSPH) {
        XSPHWeightSumi += Hdeti*mi/rhoi*W0;
        CHECK2(XSPHWeightSumi != 0.0, i << " " << XSPHWeightSumi);
        DxDti = vi + XSPHDeltaVi/max(tiny, XSPHWeightSumi);
      } else {
        DxDti = vi;
      }

      // The H tensor evolution.
      DHDti = mSmoothingScaleMethod.smoothingScaleDerivative(Hi,
                                                             ri,
                                                             DvDxi,
                                                             hmin,
                                                             hmax,
                                                             hminratio,
                                                             nPerh);
      Hideali = mSmoothingScaleMethod.newSmoothingScale(Hi,
                                                        ri,
                                                        weightedNeighborSumi,
                                                        massSecondMomenti,
                                                        W,
                                                        hmin,
                                                        hmax,
                                                        hminratio,
                                                        nPerh,
                                                        connectivityMap,
                                                        nodeListi,
                                                        i);
    }
  }
  TIME_SPHevalDerivs_final.stop();
  TIME_SPHevalDerivs.stop();
}

}
