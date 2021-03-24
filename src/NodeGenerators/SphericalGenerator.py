#-------------------------------------------------------------------------------
# Factory function to convert any of the 1D generators to spherical coordinates.
#-------------------------------------------------------------------------------
def SphericalGenerator(generator):

    # Correct the mass.
    n = len(generator.m)
    for i in xrange(n):
        ri = generator.localPosition(i).x
        assert ri > 0.0
        generator.m[i] *= ri*ri

    return generator
