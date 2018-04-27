#-------------------------------------------------------------------------------
# Dumper class for outputting Spheral++ format data for generating synthetic
# radiographs with Hades.
#-------------------------------------------------------------------------------
import Spheral
from SpheralModules import silo
from SpheralModules.silo import SiloAttributes as SA
import mpi
import struct, time, os, bisect
from operator import mul

#-------------------------------------------------------------------------------
# Write a silo file resampling to a fixed cartesian mesh for the density.
#-------------------------------------------------------------------------------
def hadesDump(integrator,
              nsample,
              xmin,
              xmax,
              W,
              baseFileName,
              baseDirectory = ".",
              procDirBaseName = "proc-%06i",
              mask = None,
              materials = None):

    # Currently suppport 2D and 3D.
    db = integrator.dataBase()
    if db.nDim == 2:
        import Spheral2d as sph
    elif db.nDim == 3:
        import Spheral3d as sph
    else:
        raise RuntimeError, "hadesDump ERROR: must be 2D or 3D"

    # Prepare to time how long this takes.
    t0 = time.clock()

    # Get the set of material names we're going to write.
    if materials is None:
        materials = list(db.fluidNodeLists())

    # HACK!  We are currently restricting to writing single material output!
    assert len(materials) == 1

    # Make sure the output directory exists.
    if mpi.rank == 0 and not os.path.exists(baseDirectory):
        try:
            os.makedirs(baseDirectory)
        except:
            raise RuntimeError, "Cannot create output directory %s" % baseDirectory
    mpi.barrier()

    # Sample the density.
    ntot = reduce(mul, nsample)
    for nodes in materials:
        print "hadesDump: sampling density for %s..." % nodes.name
        r = sph.VectorFieldList()
        H = sph.SymTensorFieldList()
        rho = sph.ScalarFieldList()
        r.appendField(nodes.positions())
        H.appendField(nodes.Hfield())
        rho.appendField(nodes.massDensity())

        mf = nodes.mass()
        rhof = nodes.massDensity()
        wf = sph.ScalarField("volume", nodes)
        for i in xrange(nodes.numNodes):
            wf[i] = mf[i]/max(1e-100, rhof[i])
        w = sph.ScalarFieldList()
        w.copyFields()
        w.appendField(wf)
        #w.appendField(sph.ScalarField("weight", nodes, 1.0))

        fieldListSet = sph.FieldListSet()
        fieldListSet.ScalarFieldLists.append(rho)
        localMask = sph.IntFieldList()
        if mask is None:
            localMask.copyFields()
            localMask.appendField(sph.IntField("mask", nodes, 1))
        else:
            localMask.appendField(mask.fieldForNodeList(nodes))

        scalar_samples = sph.vector_of_vector_of_double()
        vector_samples = sph.vector_of_vector_of_Vector()
        tensor_samples = sph.vector_of_vector_of_Tensor()
        symTensor_samples = sph.vector_of_vector_of_SymTensor()
        nsample_vec = sph.vector_of_int(db.nDim)
        for i in xrange(db.nDim):
            nsample_vec[i] = nsample[i]

        sph.sampleMultipleFields2Lattice(fieldListSet,
                                         r, w, H, localMask,
                                         W,
                                         xmin, xmax,
                                         nsample_vec,
                                         scalar_samples,
                                         vector_samples,
                                         tensor_samples,
                                         symTensor_samples)
        print "Generated %i scalar fields" % len(scalar_samples)

        # Rearrange the sampled data into rectangular blocks due to Silo's quad mesh limitations.
        rhosamp, xminblock, xmaxblock, nblock = shuffleIntoBlocks(db.nDim, scalar_samples[0], xmin, xmax, nsample)
        print "rho range: ", min(rhosamp), max(rhosamp)
        print "     xmin: ", xmin
        print "     xmax: ", xmax
        print "xminblock: ", xminblock
        print "xmaxblock: ", xmaxblock
        print "   nblock: ", nblock
        assert mpi.allreduce(len(rhosamp), mpi.SUM) == ntot

    # Write the master file.
    maxproc = writeMasterSiloFile(baseDirectory = baseDirectory,
                                  baseName = baseFileName,
                                  procDirBaseName = procDirBaseName,
                                  materials = materials,
                                  rhosamp = rhosamp,
                                  label = "Spheral++ cartesian sampled output",
                                  time = integrator.currentTime,
                                  cycle = integrator.currentCycle)

    # Write the process files.
    writeDomainSiloFile(ndim = db.nDim,
                        maxproc = maxproc,
                        baseDirectory = baseDirectory,
                        baseName = baseFileName,
                        procDirBaseName = procDirBaseName,
                        materials = materials,
                        rhosamp = rhosamp,
                        xminblock = xminblock,
                        xmaxblock = xmaxblock,
                        nblock = nblock,
                        label = "Spheral++ cartesian sampled output",
                        time = integrator.currentTime,
                        cycle = integrator.currentCycle,
                        pretendRZ = db.isRZ)

    mpi.barrier()
    print "hadesDump finished: required %0.2f seconds" % (time.clock() - t0)
    return

#-------------------------------------------------------------------------------
# Rearrange the distributed data into rectangular blocks for each domain due
# to Silo's restrictions for quad meshes.
# Returns:
#   valsblock : array of local values arranged in a block
#   xminblock : min coordinates of block
#   xmaxblock : max coordinates of block
#   nblock    : ndim dimensioned array, length of sampled block in each dimension
#
# This method simplifies by always cutting into slabs.  This may result in
# some domains having zero values though, if there are more processors than
# lattice slabs in the chosen direction.
#-------------------------------------------------------------------------------
def shuffleIntoBlocks(ndim, vals, xmin, xmax, nglobal):

    # In 2D we expect nglobal = (nx, ny, 1)
    #assert len(nglobal) == 3

    if ndim == 2:
        import Spheral2d as sph
    else:
        import Spheral3d as sph

    dx = [(xmax[j] - xmin[j])/nglobal[j] for j in xrange(ndim)]
    ntot = reduce(mul, nglobal)

    # Which dimension should we divide up into?
    jmax = min(ndim - 1, max(enumerate(nglobal), key = lambda x: x[1])[0])

    # Find the offset to the global lattice numbering on this domain.
    # This is based on knowing the native lattice sampling method stripes the original data
    # accoriding to (i + j*nx + k*nx*ny), and simply divides that 1D serialization sequentially
    # between processors.
    offset = 0
    for sendproc in xrange(mpi.procs):
        n = mpi.bcast(len(vals), root=sendproc)
        if sendproc < mpi.rank:
            offset += n
    if mpi.rank == mpi.procs - 1:
        assert offset + len(vals) == ntot

    # A function to turn an index into the integer lattice coordinates
    def latticeCoords(iglobal):
        return (iglobal % nglobal[0],
                (iglobal % (nglobal[0]*nglobal[1])) // nglobal[0],
                iglobal // (nglobal[0]*nglobal[1]))

    # A function to tell us which block to assign a global index to
    slabsperblock = max(1, nglobal[jmax] // mpi.procs)
    remainder = max(0, nglobal[jmax] - mpi.procs*slabsperblock)
    islabdomain = [min(nglobal[jmax], iproc*slabsperblock + min(iproc, remainder)) for iproc in xrange(mpi.procs + 1)]
    #print "Domain splitting: ", nglobal, jmax, islabdomain
    def targetBlock(index):
        icoords = latticeCoords(offset + index)
        return bisect.bisect(islabdomain, icoords[jmax]) - 1

    # Build a list of (global_index, value, target_proc) for each of the lattice values.
    id_val_procs = [(offset + i, val, targetBlock(offset + i)) for i, val in enumerate(vals)]
    
    # Send our values to other domains.
    sendreqs, sendvals = [], []
    for iproc in xrange(mpi.procs):
        if iproc != mpi.rank:
            sendvals.append([(i, val) for (i, val, proc) in id_val_procs if proc == iproc])
            sendreqs.append(mpi.isend(sendvals[-1], dest=iproc, tag=100))

    # Now we can build the dang result.
    xminblock, xmaxblock = sph.Vector(*xmin), sph.Vector(*xmax)
    xminblock[jmax] = xmin[jmax] + islabdomain[mpi.rank]    *dx[jmax]
    xmaxblock[jmax] = xmin[jmax] + islabdomain[mpi.rank + 1]*dx[jmax]
    nblock = list(nglobal)
    nblock[jmax] = islabdomain[mpi.rank + 1] - islabdomain[mpi.rank]
    newvals = []
    for iproc in xrange(mpi.procs):
        if iproc == mpi.rank:
            recvvals = [(i, val) for (i, val, proc) in id_val_procs if proc == mpi.rank]
        else:
            recvvals = mpi.recv(source=iproc, tag=100)
        newvals += recvvals
    newvals.sort()
    valsblock = sph.vector_of_double()
    for i, val in newvals:
        valsblock.append(val)
    assert len(valsblock) == reduce(mul, nblock)

    # Wait 'til all communication is done.
    for req in sendreqs:
        req.wait()

    # That should be it.
    return valsblock, xminblock, xmaxblock, nblock

#-------------------------------------------------------------------------------
# Write the master file.
#-------------------------------------------------------------------------------
def writeMasterSiloFile(baseDirectory, baseName, procDirBaseName, materials,
                        rhosamp, label, time, cycle):

    nullOpts = silo.DBoptlist()

    # Decide which domains have information.
    if len(rhosamp) > 0:
        myvote = mpi.rank + 1
    else:
        myvote = 0
    maxproc = mpi.allreduce(myvote, mpi.MAX)
    assert maxproc >= mpi.procs

    # Pattern for constructing per domain variables.
    domainNamePatterns = [os.path.join(procDirBaseName % i, "domain%i.silo:%%s" % i) for i in xrange(maxproc)]
    domainVarNames = Spheral.vector_of_string()
    name = "mass_density"
    for iproc, p in enumerate(domainNamePatterns):
        domainVarNames.append(p % name)
    assert len(domainVarNames) == maxproc

    # Create the master file.
    if mpi.rank == 0:
        fileName = os.path.join(baseDirectory, baseName + ".silo")
        f = silo.DBCreate(fileName, 
                          SA._DB_CLOBBER, SA._DB_LOCAL, label, SA._DB_HDF5)

        # Write the domain file names and types.
        domainNames = Spheral.vector_of_string()
        meshTypes = Spheral.vector_of_int(maxproc, SA._DB_QUADRECT)
        for p in domainNamePatterns:
            domainNames.append(p % "MESH")
        optlist = silo.DBoptlist(1024)
        assert optlist.addOption(SA._DBOPT_CYCLE, cycle) == 0
        assert optlist.addOption(SA._DBOPT_DTIME, time) == 0
        assert silo.DBPutMultimesh(f, "MMESH", domainNames, meshTypes, optlist) == 0

        # Write material names.
        material_names = Spheral.vector_of_string()
        matnames = Spheral.vector_of_string()
        matnos = Spheral.vector_of_int()
        for p in domainNamePatterns:
            material_names.append(p % "material")
        for i, name in enumerate([x.name for x in materials]):
            matnames.append(name)
            matnos.append(i + 1)
        assert len(material_names) == maxproc
        assert len(matnames) == len(materials)
        assert len(matnos) == len(materials)
        optlist = silo.DBoptlist(1024)
        assert optlist.addOption(SA._DBOPT_CYCLE, cycle) == 0
        assert optlist.addOption(SA._DBOPT_DTIME, time) == 0
        assert optlist.addOption(SA._DBOPT_MATNAMES, SA._DBOPT_NMATNOS, matnames) == 0
        assert optlist.addOption(SA._DBOPT_MATNOS, SA._DBOPT_NMATNOS, matnos) == 0
        assert silo.DBPutMultimat(f, "MMATERIAL", material_names, optlist) == 0
        
        # Write the variables descriptors.
        # We currently hardwire for the single density variable.
        name = "mass_density"
        types = Spheral.vector_of_int(maxproc, SA._DB_QUADVAR)
        assert len(domainVarNames) == maxproc
        optlistMV = silo.DBoptlist()
        assert optlistMV.addOption(SA._DBOPT_CYCLE, cycle) == 0
        assert optlistMV.addOption(SA._DBOPT_DTIME, time) == 0
        assert optlistMV.addOption(SA._DBOPT_TENSOR_RANK, SA._DB_VARTYPE_SCALAR) == 0
        assert silo.DBPutMultivar(f, name, domainVarNames, types, optlistMV) == 0

        # Close the file.
        assert silo.DBClose(f) == 0
        del f

    return maxproc

#-------------------------------------------------------------------------------
# Write the domain file.
#-------------------------------------------------------------------------------
def writeDomainSiloFile(ndim, maxproc, baseDirectory, baseName, procDirBaseName,
                        materials, rhosamp,
                        xminblock, xmaxblock, nblock,
                        label, time, cycle,
                        pretendRZ):

    # Make sure the directories are there.
    if mpi.rank == 0:
        for iproc in xrange(maxproc):
            pth = os.path.join(baseDirectory,
                               procDirBaseName % iproc)
            if not os.path.exists(pth):
                os.makedirs(pth)
    mpi.barrier()

    # Is there anything to do?
    if mpi.rank < maxproc:
        numZones = 1
        for x in nblock:
            numZones *= x
        assert numZones > 0
        assert len(rhosamp) == numZones

        # Create the file.
        fileName = os.path.join(baseDirectory,
                                procDirBaseName % mpi.rank,
                                "domain%i.silo" % mpi.rank)
        f = silo.DBCreate(fileName, 
                          SA._DB_CLOBBER, SA._DB_LOCAL, label, SA._DB_HDF5)
        nullOpts = silo.DBoptlist()

        # Write the domain mesh.
        coords = Spheral.vector_of_vector_of_double(ndim)
        for jdim in xrange(ndim):
            coords[jdim] = Spheral.vector_of_double(nblock[jdim] + 1)
            dx = (xmaxblock[jdim] - xminblock[jdim])/nblock[jdim]
            for i in xrange(nblock[jdim] + 1):
                coords[jdim][i] = xminblock[jdim] + i*dx
        optlist = silo.DBoptlist()
        assert optlist.addOption(SA._DBOPT_CYCLE, cycle) == 0
        assert optlist.addOption(SA._DBOPT_DTIME, time) == 0
        if pretendRZ:
            assert optlist.addOption(SA._DBOPT_COORDSYS, SA._DB_CYLINDRICAL) == 0
        else:
            assert optlist.addOption(SA._DBOPT_COORDSYS, SA._DB_CARTESIAN) == 0
        assert silo.DBPutQuadmesh(f, "MESH", coords, optlist) == 0
        
        # Write materials.
        if materials:
            matnos = Spheral.vector_of_int()
            for i in xrange(len(materials)):
                matnos.append(i)
            assert len(matnos) == len(materials)
            matlist = Spheral.vector_of_int(numZones, 0)
            matnames = Spheral.vector_of_string()
            for imat, nodeList in enumerate(materials):
                for i in xrange(numZones):
                    if rhosamp[i] > 0.0:
                        matlist[i] = imat + 1
                matnames.append(nodeList.name)
            assert len(matlist) == numZones
            assert len(matnames) == len(materials)
            matOpts = silo.DBoptlist(1024)
            assert matOpts.addOption(SA._DBOPT_CYCLE, cycle) == 0
            assert matOpts.addOption(SA._DBOPT_DTIME, time) == 0
            assert matOpts.addOption(SA._DBOPT_MATNAMES, SA._DBOPT_NMATNOS, matnames) == 0
            assert silo.DBPutMaterial(f, "MATERIAL", "MESH", matnos, matlist,
                                      Spheral.vector_of_int(), Spheral.vector_of_int(), Spheral.vector_of_int(), Spheral.vector_of_double(),
                                      matOpts) == 0
        
        # Write the field variables.
        varOpts = silo.DBoptlist(1024)
        assert varOpts.addOption(SA._DBOPT_CYCLE, cycle) == 0
        assert varOpts.addOption(SA._DBOPT_DTIME, time) == 0
        nblock_vec = Spheral.vector_of_int(ndim)
        for jdim in xrange(ndim):
            nblock_vec[jdim] = nblock[jdim]
        assert silo.DBPutQuadvar1(f, "mass_density", "MESH", rhosamp, 
                                  Spheral.vector_of_double(), SA._DB_ZONECENT, nblock_vec, varOpts) == 0

        # That's it.
        assert silo.DBClose(f) == 0
        del f

    return

#-------------------------------------------------------------------------------
# Our original proposition for a Hades file format.
#-------------------------------------------------------------------------------
## hadesHeader = """
## # Spheral++ -> Hades file
## # Format:
## #   N (Num materials)
## #   xmin ymin zmin
## #   xmax ymax zmax
## #   nx ny nz
## #   (num isotopes 1) ZA frac ZA frac ...
## #   ...
## #   (num isotopes N) ZA frac ZA frac ...
## #   mass density 1
## #   ...
## #   mass density N
## # Mass densities are written as index1 rho1 index2 rho2 ...
## """
def hadesDump0(integrator,
               nsample,
               xmin,
               xmax,
               W,
               isotopes,
               baseFileName,
               baseDirectory = ".",
               dumpGhosts = False,
               materials = "all"):

    # We currently only support 3-D.
    assert isinstance(integrator, Spheral.Integrator3d)
    assert len(nsample) == 3
    assert isinstance(xmin, Spheral.Vector3d)
    assert isinstance(xmax, Spheral.Vector3d)
    assert isinstance(W, Spheral.TableKernel3d)
    for x in isotopes:
        for xx in x:
            assert len(xx) == 2

    # Prepare to time how long this takes.
    t0 = time.clock()

    # Extract the data base.
    db = integrator.dataBase()

    # If requested, set ghost node info.
    if dumpGhosts and not integrator is None:
        state = Spheral.State3d(db, integrator.physicsPackages())
        derivs = Spheral.StateDerivatives3d(db, integrator.physicsPackages())
        integrator.setGhostNodes()
        integrator.applyGhostBoundaries(state, derivs)

    # Get the set of material names we're going to write.
    if materials == "all":
        materials = [n for n in db.fluidNodeLists()]
    assert len(materials) == len(isotopes)

    # Make sure the output directory exists.
    import mpi
    import os
    if mpi.rank == 0 and not os.path.exists(baseDirectory):
        try:
            os.makedirs(baseDirectory)
        except:
            raise "Cannot create output directory %s" % baseDirectory
    mpi.barrier()

    # Open a file for the output.
    currentTime = integrator.currentTime
    currentCycle = integrator.currentCycle
    filename = baseDirectory + "/" + baseFileName + "-time=%g-cycle=%i.hades" % (currentTime, currentCycle)

    if mpi.rank == 0:
        f = open(filename, "wb")

        # Write the header info.
        #f.write(hadesHeader)
        f.write(struct.pack("I", len(materials)))
        f.write(struct.pack("ddd", *tuple(xmin.elements())))
        f.write(struct.pack("ddd", *tuple(xmax.elements())))
        f.write(struct.pack("III", *nsample))
        for materialIsotopes in isotopes:
            f.write(struct.pack("I", len(materialIsotopes)))
            for iso in materialIsotopes:
                f.write(struct.pack("Id", *iso))

    # For each material, sample the mass density and write it out.
    ntot = nsample[0]*nsample[1]*nsample[2]
    for nodes in materials:
        r = Spheral.VectorFieldList3d()
        w = Spheral.ScalarFieldList3d()
        H = Spheral.SymTensorFieldList3d()
        rho = Spheral.ScalarFieldList3d()
        r.appendField(nodes.positions())
        w.appendField(nodes.weight())
        H.appendField(nodes.Hfield())
        rho.appendField(nodes.massDensity())
        fieldListSet = Spheral.FieldListSet3d()
        fieldListSet.ScalarFieldLists.append(rho)
        rhosamp = Spheral.sampleMultipleFields2LatticeMash(fieldListSet,
                                                           r, w, H,
                                                           W,
                                                           xmin, xmax,
                                                           nsample)[0][0][1]
        assert mpi.allreduce(len(rhosamp), mpi.SUM) == ntot
        icum = 0
        for sendProc in xrange(mpi.procs):
            valsproc = [(i, x) for (i, x) in zip(range(ntot), rhosamp) if x > 0.0]
            vals = mpi.bcast(valsproc, sendProc)
            if mpi.rank == 0:
                f.write(struct.pack("I", len(vals)))
                for i, x in vals:
                    f.write(struct.pack("id", i + icum, x))
            icum += len(vals)

    if mpi.rank == 0:
        # Close the file and we're done.
        f.close()

    mpi.barrier()
    print "hadesDump finished: required %0.2f seconds" % (time.clock() - t0)

    return

#-------------------------------------------------------------------------------
# Write a pre-existing Hades format. (Another defunct option)
#
# This format writes four files:
# 
# <basename>.spr (ASCII)
#   3     # (I), number of dimensions
#   nx    # (I), numer in x direction
#   xmin  # (F), minimum x value
#   lx    # (F), cell length in x
#   ny    # (I), numer in y direction
#   ymin  # (F), minimum y value
#   ly    # (F), cell length in y
#   nz    # (I), numer in z direction
#   zmin  # (F), minimum z value
#   lz    # (F), cell length in z
#   3     # (I), HADES flag indicating type
#
# <basename>.sdt (binary)
# <float data, nx x ny x nz>  # Full density data as float*4 array
#
# <basename)>_mat.sdt  (binary)
# <int data, nx x ny x nz>    # Integer flags indicating material in each cell
#
# isos.mat       (ASCII)
#  isofracs(1) = ZA frac ZA frac ....   # isotopics for material 1
#  isofracs(2) = ZA frac ZA frac ....   # isotopics for material 2
#  ...
#  isofracs(n) = ZA frac ZA frac ....   # isotopics for material n
#-------------------------------------------------------------------------------
def hadesDump1(integrator,
              nsample,
              xmin,
              xmax,
              W,
              isotopes,
              baseFileName,
              baseDirectory = ".",
              mask = None,
              dumpGhosts = True,
              materials = "all"):

    # We currently only support 3-D.
    assert isinstance(integrator, Spheral.Integrator3d)
    assert len(nsample) == 3
    assert isinstance(xmin, Spheral.Vector3d)
    assert isinstance(xmax, Spheral.Vector3d)
    assert isinstance(W, Spheral.TableKernel3d)
    for x in isotopes:
        for xx in x:
            assert len(xx) == 2

    # Prepare to time how long this takes.
    t0 = time.clock()

    # Extract the data base.
    db = integrator.dataBase()

    # If requested, set ghost node info.
    if dumpGhosts and not integrator is None:
        state = Spheral.State3d(db, integrator.physicsPackages())
        derivs = Spheral.StateDerivatives3d(db, integrator.physicsPackages())
        integrator.setGhostNodes()
        integrator.applyGhostBoundaries(state, derivs)

    # Get the set of material names we're going to write.
    if materials == "all":
        materials = [n for n in db.fluidNodeLists()]
    assert len(materials) == len(isotopes)

    # HACK!  We are currently restricting to writing single material output!
    assert len(materials) == 1

    # Make sure the output directory exists.
    import mpi
    import os
    if mpi.rank == 0 and not os.path.exists(baseDirectory):
        try:
            os.makedirs(baseDirectory)
        except:
            raise "Cannot create output directory %s" % baseDirectory
    mpi.barrier()

    # Write the density header file.
    print "hadesDump: writing density header..."
    if mpi.rank == 0:
        filename = os.path.join(baseDirectory, baseFileName + ".spr")
        f = open(filename, "w")
        f.write("3\n")
        for i in xrange(3):
            f.write("%i\n" % nsample[i])
            f.write("%f\n" % xmin(i))
            f.write("%f\n" % ((xmax(i) - xmin(i))/nsample[i]))
        f.write("3\n")
        f.close()
    mpi.barrier()

    # Sample the density.
    ntot = nsample[0]*nsample[1]*nsample[2]
    for nodes in materials:
        print "hadesDump: sampling density for %s..." % nodes.name
        r = Spheral.VectorFieldList3d()
        H = Spheral.SymTensorFieldList3d()
        rho = Spheral.ScalarFieldList3d()
        r.appendField(nodes.positions())
        w.appendField(nodes.weight())
        H.appendField(nodes.Hfield())
        rho.appendField(nodes.massDensity())

        w = Spheral.ScalarFieldList3d()
        w.copyFields()
        w.appendField(Spheral.ScalarField3d("weight", nodes, 1.0))

        fieldListSet = Spheral.FieldListSet3d()
        fieldListSet.ScalarFieldLists.append(rho)
        localMask = Spheral.IntFieldList3d()
        if mask is None:
            localMask.copyFields()
            localMask.appendField(Spheral.IntField3d("mask", nodes, 1))
        else:
            localMask.appendField(mask.fieldForNodeList(nodes))

        scalar_samples = Spheral.vector_of_vector_of_double()
        vector_samples = Spheral.vector_of_vector_of_Vector3d()
        tensor_samples = Spheral.vector_of_vector_of_Tensor3d()
        symTensor_samples = Spheral.vector_of_vector_of_SymTensor3d()
        nsample_vec = Spheral.vector_of_int(3)
        for i in xrange(3):
            nsample_vec[i] = nsample[i]

        Spheral.sampleMultipleFields2Lattice3d(fieldListSet,
                                               r, w, H, localMask,
                                               W,
                                               xmin, xmax,
                                               nsample_vec,
                                               scalar_samples,
                                               vector_samples,
                                               tensor_samples,
                                               symTensor_samples)
        print "Generated %i scalar fields" % len(scalar_samples)
        rhosamp = scalar_fields[0]
        nlocal = len(rhosamp)
        assert mpi.allreduce(nlocal, mpi.SUM) == ntot

        print "hadesDump: writing density for %s..." % nodes.name
        filename = os.path.join(baseDirectory, baseFileName + ".sdt")
        for sendProc in xrange(mpi.procs):
            if mpi.rank == sendProc:
                f = open(filename, "ab")
                f.write(struct.pack(nlocal*"f", *tuple(rhosamp)))
                f.close()
            mpi.barrier()

    # Write the material arrays.
    print "hadesDump: writing material flags..."
    filename = os.path.join(baseDirectory, baseFileName + "_mat.sdt")
    for sendProc in xrange(mpi.procs):
        if mpi.rank == sendProc:
            f = open(filename, "ab")
            f.write(struct.pack(nlocal*"i", *(nlocal*(1,))))
            f.close()
        mpi.barrier()

    # Write the isotopes.
    print "hadesDump: writing isotopics..."
    if mpi.rank == 0:
        filename = os.path.join(baseDirectory, "isos.mat")
        f = open(filename, "w")
        i = 0
        for isofracs in isotopes:
            f.write("isofrac(%i) =" % i)
            for (iso, frac) in isofracs:
                f.write(" %i %f" % (iso, frac))
            f.write("\n")
            i += 1
        f.close()
    mpi.barrier()

    mpi.barrier()
    print "hadesDump finished: required %0.2f seconds" % (time.clock() - t0)
    return

