import sys
import time
import numpy as np
from pathlib import Path
from scipy import sparse


"""
This file contains numpy implementations of
various uv mappers (functions that map a 3D triangulated
surface to a 2D uv coordinate system).

Here are the papers that I consulted, along with the nicknames I've
given to the algorithms they describe:

"Xyz-based LSCM"
Lévy B., Petitjean S., Ray N., Maillot J.: Least squares conformal maps
for automatic texture atlas generation. ACM Transactions on Graphics
(Proc. SIGGRAPH) 21, 3 (2002), 362–371.

"ABF"
Sheffer A., de Sturler E.: Parameterization of faceted surfaces
for meshing using angle based flattening. Engineering with
Computers 17, 3 (2001), 326–337.

"ABF++", "Angle-based LSCM"
Sheffer A., Lévy B., Mogilnitsky M., Bogomyakov A.: ABF++: fast and
robust angle based flattening.  ACM Trans. Graph. 24, 2 (2005), 311–330.

NOTE: If you look for the ABF++ paper online, make sure you get
the correct version.  If, just below the title, you see the phrase
"Temporary version- In print", you have the WRONG version, which is
missing the description of angle-based LSCM.
The following link points to the correct version, as of June 2024:
https://people.engr.tamu.edu/schaefer/teaching/689_Fall2006/p311-sheffer.pdf
Also, as of June 2024, the version of the paper on sci-hub.se contains
only the first page!

"Linearized ABF"
Zayer R., Levy B., Seidel H.P. Linear Angle Based Parameterization.
Belyaev A., Garland M. Fifth Eurographics Symposium on Geometry
Processing SGP 2007, Jul 2007, Barcelona, Spain. Eurographics
Association, pp.135-141, 2007.

It is interesting to note that three of the four papers have Bruno Lévy
as a co-author.

Comments on the individual algorithms:

Xyz-based LSCM, as the name suggests, uses the xyz locations
of the vertices as input to the algorithm that finds the uv
parameterization.  The algorithm sets up a linear system, which
is slightly under-determined.  In order to make it fully deterministic,
the uv coordinates of two points must be specified (the "pinned points").
On a typical non-flat surface, the end result (the uv parameterization)
is dependent on the location of the pinned points, since the algorithm
drifts when far from these points.  For best results, the authors
recommend selecting two pinned points that are as far apart from each
other as possible.

When given only two pinned points, xyz-based LSCM, as implemented
here, produces counter-clockwise triangles in uv space (this assumes
that the uv coordinate system is right-handed). However, if
there are more than two pinned points, and these points imply
the triangles should be clockwise, then xyz-based LSCM will
create clockwise triangles.

The current file contains a numpy implementation of xyz-based LSCM.


Angle-based LSCM uses the angles of the individual triangles
as input to the uv parameterization algorithm.  The angles can
of course be computed from the xyz locations of the vertices,
but they can also come from other sources, such as angle-based
flattening (See below).  This algorithm is described in the ABF++ paper.
Like the xyz-based LSCM algorithm, the angle-based algorithm requires
that uv values be specified at two pinned points, and like xyz-based
LSCM, the uv values will drift when far from the pinned points.  However,
if the surface is flat (that is, if the input angles are consistent
with a perfectly flat surface), the location of the pinned points
does not matter; the uv values do not drift in that case.

As mentioned before, xyz-based LSCM can be coaxed into producing
either clockwise or counter-clockwise triangles, depending on
the constraints. Angle-based LSCM is not so flexible; it will
produce triangles of only a single orientation.  If there are
more than two constraints, and the constraints imply triangles
of the opposite orientation, angle-based LSCM will produce
bad results.  So it is important that angle-based LSCM
be implemented with the orientation that matches the expected
orientation of the constraints.

The paper and the EduceLab implementations of angle-based LSCM
produce clockwise triangles in uv space.

The current file contains a numpy implementation of angle-based LSCM.
This implementation was guided, in part, by the C++ header-file-only
implementation of angle-based LSCM at https://github.com/educelab/OpenABF
However, the here implementation is modified (as highlighted in the 
code comments) in order to produce counter-clockwise instead of
clockwise triangles in uv space.


ABF and ABF++ are two angle-based flattening methods.  That is,
they take as input the angles around the individual triangles,
and as output they produce modified angles that are consistent
with a perfectly flat surface.  These algorithms are non-linear.
The difference between ABF and ABF++ is that ABF++ uses a more
efficient formulation of the problem.  I have not programmed either
of these algorithms.  A C++ header-file-only implementation of ABF++
is provided at https://github.com/educelab/OpenABF.

ABF++ and LinABF take angles as input, and produce angles
as output, so they are not concerned with the question of
whether the triangles are clockwise or counter-clockwise in
uv space.

Since ABF and ABF++ output angles rather than uv values, the
uv values must subsequently be computed using an algorithm such
as angle-based LSCM.  During this computation, the triangle orientation
(clockwise vs counter-clockwise) is determined.


Linearized ABF (often referred to as LinABF) is based on a linearized
form of the equations developed in the ABF paper.  Like ABF and ABF++,
it takes as input the angles around the triangles, and produces
modified angles that are supposed to be consistent with a perfectly
flat surface.  Because the original equations are non-linear,
linearized ABF does not provide a perfectly accurate (that is,
perfectly flat) set of angles.  This is noticeable when the angles
produced by LinABF are fed into angle-based LSCM to compute uv
locations.  Because the input angles don't represent a perfectly
flat surface, the output of angle-based LSCM turns out, in this
case, to be dependent on the location of the pinned points.

My solution to this inaccuracy is to rerun linearized ABF as many
times as necessary to reduce the angle inconsistencies below a
certain tolerance level.  After that, the angles can be passed
to angle-based LSCM.  Because I have not implemented the ABF++
algorithm, I don't know whether it would be faster to run ABF++
once, or rerun LinABF as many times as necessary.

The current file contains a numpy implementation of linearized ABF.
This implementation was guided, in part, by the C++ implementation
of linearized ABF at https://gist.github.com/ialhashim/c34aa1159350bce13835


"""

# copied from utils.py to remove dependency
class Timer():

    def __init__(self, active=True):
        self.t0 = time.time()
        self.active = active

    def time(self, msg=""):
        t = time.time()
        if self.active:
            print("%.3f %s"%(t-self.t0, msg))
        self.t0 = t


class UVMapper:
    def __init__(self, points, trgls):
        self.points = points
        self.trgls = trgls

        # shape (nc, 3), and dtype float64,
        # with each row: pt index, u, v
        # note that pt index is stored as a float
        self.constraints = None

        # shape (nt, 3) and dtype float64
        # with each row containing the 3 angles (in radians)
        # at the 3 triangle vertices
        self.angles = None

        # shape (nt, 3) and dtype int64
        # the trgl indices of the 3 neighboring trgls
        # (first is the trgl opposite vertex 0)
        # if no neighbor, set to -1
        # created by self.createNeighbors()
        self.neighbors = None

        # shape (nb, 2) and dtype int64,
        # nb is number of boundary edges,
        # each row has trgl index, edge index
        # (0,1,2, based on opposing vertex)
        # boundary edges are NOT given in any particular order
        self.boundaries = None

        # should be same size as points; values corresponding
        # to constrained points will be ignored
        self.initial_points = None
        self.ip_weight = 0.
        # Same size as initial_points.  ip_weight will be
        # ignored if ip_weights is set
        self.ip_weights = None

    # Copied from base_fragment.py in order
    # to remove dependency
    @staticmethod
    def findNeighbors(trgls):
        index = np.indices((len(trgls),1))[0]
        ones = np.ones((len(trgls),1), dtype=np.int32)
        zeros = np.zeros((len(trgls),1), dtype=np.int32)
        twos = 2*ones
        
        e01 = np.concatenate((trgls[:, (0,1)], index, twos, ones), axis=1)
        e12 = np.concatenate((trgls[:, (1,2)], index, zeros, ones), axis=1)
        e20 = np.concatenate((trgls[:, (2,0)], index, ones, ones), axis=1)
        
        edges = np.concatenate((e01,e12,e20), axis=0)
        rev = (edges[:,0] > edges[:,1])
        edges[rev,0:2] = edges[rev,1::-1]
        edges[rev,4] = -1
        edges = edges[edges[:,4].argsort()]
        edges = edges[edges[:,1].argsort(kind='mergesort')]
        edges = edges[edges[:,0].argsort(kind='mergesort')]
        
        ediff = np.diff(edges, axis=0)
        duprows = np.where(((ediff[:,0]==0) & (ediff[:,1]==0)))[0]
        duprows2 = np.sort(np.append(duprows, duprows+1))
        bdup = np.zeros((len(edges)), dtype=np.bool_)
        bdup[duprows2] = True
        
        neighbors = np.full((len(trgls), 3), -1, dtype=np.int32)
        
        eplus = edges[duprows+1,:4]
        eminus = edges[duprows,:4]
        # print(eplus)
        # print(eminus)
        neighbors[eplus[:,2],eplus[:,3]] = eminus[:,2]
        neighbors[eminus[:,2],eminus[:,3]] = eplus[:,2]
        return neighbors
    

    def createNeighbors(self):
        if self.trgls is None or self.points is None:
            print("createNeighbors: no triangles specified!")
            return
        # print("Creating neighbors")
        self.neighbors = self.findNeighbors(self.trgls)

    def createBoundaries(self):
        # print("Creating boundaries")
        if self.neighbors is None:
            self.createNeighbors()
        self.boundaries = np.argwhere(self.neighbors[:,:] < 0)

    def onBoundaryArray(self):
        if self.boundaries is None:
            self.createBoundaries()
        boundaries = self.boundaries
        points = self.points
        trgls = self.trgls
        npt = points.shape[0]
        # by default, all points are on boundary
        is_on_boundary = np.full(npt, True, dtype=np.bool_)
        # but if point belongs to at least one trgl,
        # then it is not on a boundary
        is_on_boundary[trgls.flatten()] = False
        # unless it really is listed as being on a boundary:
        is_on_boundary[trgls[boundaries[:,0], (boundaries[:,1]+1)%3]] = True
        is_on_boundary[trgls[boundaries[:,0], (boundaries[:,1]+2)%3]] = True
        return is_on_boundary

    def usedInTrglArray(self):
        points = self.points
        trgls = self.trgls
        npt = points.shape[0]
        used_in_trgl = np.zeros(npt, dtype=np.bool_)
        used_in_trgl[trgls.flatten()] = True
        return used_in_trgl

    def getTwoAdjacentBoundaryPoints(self, bdy_index=None):
        if self.boundaries is None:
            self.createBoundaries()
        if self.boundaries is None or len(self.boundaries) == 0:
            print("No boundaries!")
            return None
        if bdy_index is None:
            bds = self.boundaries
            pt0 = self.points[self.trgls[bds[:,0], (bds[:,1]+1)%3]]
            pt1 = self.points[self.trgls[bds[:,0], (bds[:,1]+2)%3]]
            d = pt1-pt0
            l2 = (d*d).sum(axis=1)
            bdy_index = np.argmax(l2)
        bd = self.boundaries[bdy_index]
        pt0 = self.trgls[bd[0], (bd[1]+1)%3]
        pt1 = self.trgls[bd[0], (bd[1]+2)%3]
        return pt0, pt1

    def createAngles(self):
        # print("creating angles")
        points = self.points
        trgls = self.trgls
        # print(points.shape, trgls.shape)
        if points is None or len(points) < 3:
            # print("Not enough points")
            return None
        if trgls is None or len(trgls) == 0:
            # print("No triangles")
            return None

        # shape is nt, 3, 3
        trglxyz = points[trgls]

        # d02, d10, d21
        # that is, vector from pt 0 to pt 2, vector from
        # pt 1 to pt 0, and vector from pt 2 to pt 1
        trglvecs = np.roll(trglxyz, 1, axis=1) - trglxyz

        # length of each vector
        trgllens = np.sqrt((trglvecs*trglvecs).sum(axis=2))
        trgllens[trgllens[:,:]==0] = 1.

        # normalized vectors
        trglnvecs = trglvecs/trgllens[:,:,np.newaxis] 

        # dot products of normalized vectors;
        # the number stored at position i in the array
        # corresponds to the dot product of the two vectors
        # that share vertex i.
        # minus sign because one of the two vectors is reversed
        trglndps = (-trglnvecs*np.roll(trglnvecs, -1, axis=1)).sum(axis=2)
        # angle in radians is arccos of dot product of normalized vectors
        trglangles = np.arccos(trglndps)
        self.angles = trglangles
        # print(self.angles.shape)

    # adjust angles if point is not on a
    # boundary, and sum is < (2 pi - min_deficit).
    # if so, reduce the angles proportionally.
    def adjustedAngles(self, min_deficit):
        if self.angles is None:
            # print("No angles to adjust")
            return None
        angles = self.angles.copy()
        if angles is None:
            # print("adjustAngles: angles not set!")
            return
        points = self.points
        trgls = self.trgls
        sums = np.zeros(len(points), dtype=np.float64)
        # print(trgls.shape, angles.shape)
        is_on_boundary = self.onBoundaryArray()

        # https://stackoverflow.com/questions/60481343/numpy-sum-over-repeated-entries-in-index-array
        np.add.at(sums, trgls.flatten(), angles.flatten())
        factors = np.full(len(points), 1., dtype=np.float64)
        has_deficit = np.logical_and(np.logical_and(~is_on_boundary, sums < (2*np.pi-min_deficit)), sums > 0)
        factors[has_deficit] = (2*np.pi)/sums[has_deficit]
        # print("adjusted", (factors > 1).sum())
        angles *= factors[trgls]
        # Try to prevent case where angle = 0
        tol = .001
        angles[angles < tol] = tol
        # print("adjusted", (angles[:,:] != self.angles[:,:]).sum())
        return angles

    def computeUvs(self):
        return self.computeUvsFromXyzs()

    def linABF(self):
        timer = Timer()
        timer.active = False
        if self.angles is None:
            self.createAngles()
            timer.time("angles created")
        adjusted_angles = self.adjustedAngles(1.)
        points = self.points
        trgls = self.trgls
        if adjusted_angles is None:
            # print("Not enough angles")
            return None
        if trgls is None or len(trgls) == 0:
            # print("No triangles")
            return None

        interior_points_bool = np.logical_and(~self.onBoundaryArray(), self.usedInTrglArray())
        interior_points = np.where(interior_points_bool)[0]

        nt = trgls.shape[0]
        npt = points.shape[0]
        nipt = interior_points.shape[0]
        # print("nt, npt, nipt", nt, npt, nipt)
        pt2ipt = np.full((npt), -1, dtype=np.int64)
        pt2ipt[interior_points] = np.arange(nipt)

        # Matrix A has nt+2*nipt rows and 3*nt columns
        # Vector b has 3*nt elements
        # Triplets consist of row index, column index, value
        # (row and column are stored as floats, but are used as ints)

        # Useful grids
        tgrid = np.mgrid[:nt, :3]
        tptid = 3*tgrid[0]+tgrid[1]
        # print("tptid", tptid)

        # Vertex consistency (internal points only):
        # (but start with all points, not just internal)
        A1triplet = np.stack((trgls.flatten(), tptid.flatten(), np.broadcast_to(1., 3*nt)), axis=1)
        # Convert pt index to interior-point index
        A1triplet[:,0] = pt2ipt[A1triplet[:,0].astype(np.int64)]
        # eliminate non-interior points
        A1triplet = A1triplet[A1triplet[:,0] >= 0]
        pt_angle_sum = np.zeros(npt, dtype=np.float64)
        np.add.at(pt_angle_sum, trgls.flatten(), adjusted_angles.flatten())
        pt_angle_sum = pt_angle_sum[interior_points_bool]
        pt_angle_sum = 2*np.pi - pt_angle_sum
        b1 = pt_angle_sum

        # Triangle consistency:
        A2triplet = np.stack((tgrid[0].flatten(), tptid.flatten(), np.broadcast_to(1., 3*nt)), axis=1)
        A2triplet[:,0] += nipt

        b2 = np.pi - adjusted_angles.sum(axis=1)

        # Wheel consistency (internal points only):
        # cotangent of angles
        ct = 1./np.tan(adjusted_angles)

        A3left = np.stack((np.roll(trgls, 1, axis=1).flatten(), tptid.flatten(), ct.flatten()), axis=1)
        # notice the minus sign before ct
        A3right = np.stack((np.roll(trgls, -1, axis=1).flatten(), tptid.flatten(), -ct.flatten()), axis=1)
        A3triplet = np.concatenate((A3left, A3right), axis=0)
        A3triplet[:,0] = pt2ipt[A3triplet[:,0].astype(np.int64)]
        A3triplet = A3triplet[A3triplet[:,0] >= 0]

        A3triplet[:,0] += nt + nipt

        # log of sine of angles
        ls = np.log(np.sin(adjusted_angles))
        # difference between the two angles opposite a point
        b3 = np.zeros(npt, dtype=np.float64)
        # notice the minus sign before ls
        np.add.at(b3, np.roll(trgls, 1, axis=1).flatten(), -ls.flatten())
        np.add.at(b3, np.roll(trgls, -1, axis=1).flatten(), ls.flatten())
        b3 = b3[interior_points_bool]

        Atriplet = np.concatenate((A1triplet, A2triplet, A3triplet), axis=0)
        b = np.concatenate((b1, b2, b3))

        Atriplet[:,2] *= adjusted_angles.flatten()[Atriplet[:,1].astype(np.int64)]

        AV = Atriplet[:,2]
        AI = Atriplet[:,0]
        AJ = Atriplet[:,1]
        A_sparse = sparse.coo_array((AV, (AI, AJ)), shape=(nt+2*nipt, 3*nt))

        Acsc = A_sparse.tocsc()
        At = Acsc.transpose()
        timer.time("  created Acsc, At")
        AAt = Acsc@At
        lu = sparse.linalg.splu(AAt)
        timer.time("  created splu")
        x = At@lu.solve(b)
        timer.time("  solved splu")
        # print("x min max", x.min(), x.max())

        flattened_angles = adjusted_angles*(x.reshape(nt, 3) + 1.)
        return flattened_angles

    def angleQuality(self, angles):
        points = self.points
        trgls = self.trgls
        interior_points_bool = np.logical_and(~self.onBoundaryArray(), self.usedInTrglArray())
        interior_points = np.where(interior_points_bool)[0]
        nt = trgls.shape[0]
        npt = points.shape[0]
        nipt = interior_points.shape[0]
        if nipt == 0:
            print("angleQuality: no interior points")
            return
        pt2ipt = np.full((npt), -1, dtype=np.int64)
        pt2ipt[interior_points] = np.arange(nipt)

        pt_angle_sum = np.zeros(npt, dtype=np.float64)
        # print("pt_angle_sum", pt_angle_sum.shape)
        np.add.at(pt_angle_sum, trgls.flatten(), angles.flatten())
        pt_angle_sum = pt_angle_sum[interior_points_bool]
        pt_angle_sum = 2*np.pi - pt_angle_sum

        trgl_angle_sum = np.pi - angles.sum(axis=1)

        # print("zero angles", (angles==0.).sum())
        # print("negative angles", (angles<0.).sum())
        naind = (angles<0.).nonzero()
        # print("negative angles", naind, angles[naind])
        sn = np.sin(angles)
        isnegsn = (sn <= 0)
        sn[isnegsn] = 1.
        ls = np.log(sn)
        ls[isnegsn] = np.nan
        # difference between the two angles opposite a point
        wheel_error = np.zeros(npt, dtype=np.float64)
        # notice the minus sign before ls
        np.add.at(wheel_error, np.roll(trgls, 1, axis=1).flatten(), -ls.flatten())
        np.add.at(wheel_error, np.roll(trgls, -1, axis=1).flatten(), ls.flatten())
        interior_points_bool[np.isnan(wheel_error)] = False
        wheel_error = wheel_error[interior_points_bool]
        print(pt_angle_sum)
        print(np.abs(pt_angle_sum))

        print("max errors", 
              np.max(np.abs(pt_angle_sum)), 
              np.max(np.abs(trgl_angle_sum)),
              np.max(np.abs(wheel_error))
              )
        print("avg errors", 
              np.abs(pt_angle_sum).sum()/nipt, 
              np.abs(trgl_angle_sum).sum()/nt,
              np.abs(wheel_error).sum()/nipt
              )

    def maxWheelError(self, angles):
        points = self.points
        trgls = self.trgls
        interior_points_bool = np.logical_and(~self.onBoundaryArray(), self.usedInTrglArray())
        interior_points = np.where(interior_points_bool)[0]
        nt = trgls.shape[0]
        npt = points.shape[0]
        nipt = interior_points.shape[0]
        if nipt == 0:
            # print("maxWheelError: no interior points")
            return None
        sn = np.sin(angles)
        isnegsn = (sn <= 0)
        sn[isnegsn] = 1.
        ls = np.log(sn)
        ls[isnegsn] = np.nan
        # difference between the two angles opposite a point
        wheel_error = np.zeros(npt, dtype=np.float64)
        # notice the minus sign before ls
        np.add.at(wheel_error, np.roll(trgls, 1, axis=1).flatten(), -ls.flatten())
        np.add.at(wheel_error, np.roll(trgls, -1, axis=1).flatten(), ls.flatten())
        interior_points_bool[np.isnan(wheel_error)] = False
        wheel_error = wheel_error[interior_points_bool]
        return np.max(np.abs(wheel_error))

    def computeUvsFromABF(self):
        trgls = self.trgls
        if trgls is None or len(trgls) == 0:
            print("No triangles")
            return None
        self.createAngles()
        # self.angleQuality(self.angles)
        for i in range(10):
            abf_angles = self.linABF()
            if abf_angles is None:
                print("linABF failed!")
                return None
            # self.angleQuality(abf_angles)
            self.angles = abf_angles
            mwe = self.maxWheelError(abf_angles)
            print("error",mwe)
            if mwe is None:
                print("linABF: too few points")
                return None
            # if mwe < 1.e-5:
            if mwe < 1.e-7:
                print("wheel error is small enough at iteration", i+1)
                break
        return self.computeUvsFromAngles()

    def computeUvsFromAngles(self):
        timer = Timer()
        timer.active = False
        trgls = self.trgls
        if trgls is None or len(trgls) == 0:
            # print("No triangles")
            return None
        if self.angles is None:
            self.createAngles()
            timer.time("angles created")
            # self.adjustAngles()
            adjusted = self.adjustedAngles(1.)
            # timer.time("angles adjusted")
            if adjusted is None:
                print("Adjust angles failed")
                return None
        angles = self.angles
        points = self.points
        constraints = self.constraints
        if angles is None:
            print("Not enough angles")
            return None
        '''
        if constraints is None or len(constraints) < 2:
            print("Not enough constraints")
            return None
        '''

        if constraints is None:
            constraints = np.zeros((0,3))

        nt = trgls.shape[0]
        npt = points.shape[0]
        ncp = constraints.shape[0]
        nfp = npt-ncp
        # print(nt, npt, ncp)

        # find the index of the point with the
        # largest angle
        rindex = np.argmax(np.sin(angles), axis=1)
        # The point with the largest angle should be moved to index 2
        axis1 = np.full(nt, 1, dtype=np.int64)
        # print(angles.shape, rindex.shape, axis1.shape)

        # roll trgls and angles along axis=1, according
        # to (2 - rindex), which give the per-row value of the shift.
        # This shift results in angle [:,2] being the largest.
        # Can't use np.roll, because it doesn't allow a
        # per-row shift to be set.
        # To do a per-row shift:
        # https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
        rows, column_indices = np.ogrid[:angles.shape[0], :angles.shape[1]]
        r = 2-rindex
        r[r < 0] += angles.shape[1]
        column_indices = column_indices - r[:, np.newaxis]
        rangles = angles[rows, column_indices]
        rtrgls = trgls[rows, column_indices]

        angles = None
        trgls = None

        # TODO: for testing
        # rangles = rangles[:, (1,0,2)]
        # rtrgls = rtrgls[:, (1,0,2)]

        # numerator
        num = np.sin(rangles[:,1])
        # denominator
        den = np.sin(rangles[:,2])
        # abs(den) is >= abs(num), due to the shift above
        # if den == 0, then num must == 0 too.
        ooc = (den < num).sum()
        # print("num/den out of order", ooc)

        # print("zero-value den", (den==0).sum())
        den[den == 0] = 1.
        ratio = num/den
        sn = ratio*np.sin(rangles[:,0])
        cs = ratio*np.cos(rangles[:,0])
        ones = np.full(ratio.shape, 1., dtype=np.float64)
        zeros = np.zeros(ratio.shape, dtype=np.float64)

        # m has shape nt, 3, 2, 2
        # in other words, m has a 2x2 matrix for
        # each vertex of each trgl
        m = np.zeros((nt, 3, 2, 2), dtype=np.float64)

        # Based on ABF++ paper and EduceLab implementation
        # (assumes clockwise triangles in uv space):
        # m[:, 0] = np.array([[cs-1, sn],  [-sn, cs-1]]).transpose(2,0,1)
        # m[:, 1] = np.array([[-cs,  -sn], [sn,  -cs]]).transpose(2,0,1)
        # Reversed; assumes counter-clockwise triangles in
        # uv space (when uv coordinates are right-handed)
        # The only difference is the sign of sn is changed everywhere
        m[:, 0] = np.array([[cs-1, -sn],  [sn, cs-1]]).transpose(2,0,1)
        m[:, 1] = np.array([[-cs,  sn], [-sn,  -cs]]).transpose(2,0,1)

        m[:, 2] = np.array([[ones, zeros], [zeros, ones]]).transpose(2,0,1)

        minds = np.indices(m.shape)

        m_sparse = np.stack((
            # row: trgl_id*2 + 2x2 row number
            minds[0].flatten()*2+minds[2].flatten(),
            # column: (pt-in-trgl id)*2 + 2x2 column number
            rtrgls[minds[0].flatten(), minds[1].flatten()]*2+minds[3].flatten(),
            # m value
            m.flatten()), 
            axis=1)

        # print("m_sparse")
        # print(m_sparse)

        # constraint indices
        # NOTE that cindex is not sorted
        cindex = constraints[:,0].astype(np.int64)
        # print("cindex")
        # print(cindex)
        ptindex = np.arange(npt)

        # boolean array, length na: whether point is
        # free (True) or constrained (False)
        isfree = np.full(npt, True, dtype=np.bool_)
        # isfree = np.logical_not(np.isin(ptindex, cindex))
        isfree[cindex] = False
        isfree2 = np.full(2*npt, True, dtype=np.bool_)
        isfree2[2*cindex] = False
        isfree2[2*cindex+1] = False
        # print("isfree")
        # print(isfree)

        # free points in sparse array (m_sparse[:,1] contains
        # point index)
        # mf_sparse = m_sparse[np.logical_not(np.isin(m_sparse[:,1]//2, constraints[:,0]))]
        # mf_sparse = m_sparse[isfree2[m_sparse[:,1]]]
        mf_sparse = m_sparse[isfree[m_sparse[:,1].astype(np.int64)//2]]

        # print("mf_sparse")
        # print(mf_sparse)

        # pt index to mf pt index
        # each element either contains the corresponding mf pt index,
        # or -1 if original point is constrained.
        # Keep in mind that the mf point index equals
        # 2 times the original point index, plus 0 or 1
        pt2mf = np.full((2*npt), -1, dtype=np.int64)
        # indexes of free points
        free_ind = np.nonzero(isfree)[0]
        # print("npt, free_ind, nfp", npt, isfree.sum(), free_ind.shape, nfp)
        pt2mf[2*free_ind] = 2*np.arange(nfp)
        pt2mf[2*free_ind+1] = 2*np.arange(nfp)+1

        # renumber the pt indices in mf_sparse to use
        # the free-point indexing

        mf_sparse[:,1] = pt2mf[mf_sparse[:,1].astype(np.int64)]

        # A_sparse = sparse.coo_array((AV, (AI, AJ)), shape=(2*nt, 2*nfp))

        use_weights = False
        if self.initial_points is not None and (self.ip_weight > 0 or self.ip_weights is not None):
            use_weights = True
            if self.ip_weights is None:
                weights = np.full(points.shape[0], self.ip_weight, np.float64)
            else:
                weights = self.ip_weights

            # TODO: continue here!
            wf_sparse = np.stack((
                np.ogrid[:2*nfp], 
                np.ogrid[:2*nfp], 
                # weights[isfree][np.ogrid[:npt*2]//2]),
                weights[isfree][np.ogrid[:nfp*2]//2]),
                                 axis=1)
            wf_sparse_shift = wf_sparse.copy()
            wf_sparse_shift[:, 0] += 2*nt
            all_sparse = np.concatenate((mf_sparse, wf_sparse_shift), axis=0)
            AV = all_sparse[:,2].astype(np.float64)
            AI = all_sparse[:,0]
            AJ = all_sparse[:,1]
            A_sparse = sparse.csr_array((AV, (AI, AJ)), shape=(2*nt+2*nfp, 2*nfp))
        else:
            AV = mf_sparse[:,2].astype(np.float64)
            AI = mf_sparse[:,0]
            AJ = mf_sparse[:,1]
            A_sparse = sparse.csr_array((AV, (AI, AJ)), shape=(2*nt, 2*nfp))

        # print("A sparse shape", 2*nt, 2*nfp)
        # print(A_sparse)
        # print(AI.min(), AI.max(), AJ.min(), AJ.max())
        # print(AI)
        # print(AJ)

        # pinned points in sparse array
        mp_sparse = m_sparse[np.isin(m_sparse[:,1]//2, constraints[:,0])]

        # pt index to mp pt index
        # each element either contains the corresponding mp pt index,
        # or -1 if original point is free
        pt2mp = np.full((2*npt), -1, dtype=np.int64)
        # NOTE: cindex is not sorted
        pt2mp[2*cindex] = 2*np.arange(ncp)
        pt2mp[2*cindex+1] = 2*np.arange(ncp)+1

        # renumber the pt indices in mp_sparse to use
        # the pinned-point indexing
        mp_sparse[:,1] = pt2mp[mp_sparse[:,1].astype(np.int64)]

        # to compute b, need the full (non-sparse) version
        # of mp.  Create this by first creating an
        # array with all zeros, and then filling in
        # the points from mp_sparse
        mp_full = np.zeros((2*nt, 2*ncp), dtype=np.float64)
        mp_full[mp_sparse[:,0].astype(np.int64), mp_sparse[:,1].astype(np.int64)] = mp_sparse[:, 2]

        # print("mp_full")
        # print(mp_full)
        # print(constraints[:,(1,2)])

        b = -mp_full @ constraints[:,(1,2)].flatten()
        if use_weights:
            AV = wf_sparse[:,2].astype(np.float64)
            AI = wf_sparse[:,0]
            AJ = wf_sparse[:,1]
            wf_csr = sparse.csr_array((AV, (AI, AJ)), shape=(2*nfp, 2*nfp))
            bw = wf_csr @ self.initial_points[isfree].flatten()
            # print("bw", bw)
        b = np.concatenate((b, bw))

        timer.time("  created arrays")

        # Acsr = A_sparse.tocsr()
        Acsr = A_sparse
        At = Acsr.transpose()
        # print(At@Acsr)
        # print(b)
        # print(At@b)
        try:
            lu = sparse.linalg.splu(At@Acsr)
            x = lu.solve(At@b)
        except Exception as e:
            print("SPLU exception:", e)
            return None
        uv = np.zeros((npt, 2), dtype=np.float64)
        uv[isfree, :] = x.reshape(nfp, 2)
        uv[cindex, :] = constraints[:,(1,2)]
        return uv

    def computeUvsFromXyzs(self):
        timer = Timer()
        timer.active = False
        points = self.points
        trgls = self.trgls
        constraints = self.constraints
        if points is None or len(points) < 3:
            print("Not enough points")
            return None
        if trgls is None or len(trgls) == 0:
            print("No triangles")
            return None
        '''
        if constraints is None or len(constraints) < 2:
            print("Not enough constraints")
            return None
        '''
        if constraints is None:
            constraints = np.zeros((0,3))

        nt = trgls.shape[0]
        npt = points.shape[0]
        ncp = constraints.shape[0]
        nfp = npt-ncp
        # print("nt, npt, ncp", nt, npt, ncp)


        trglxyz = points[trgls]
        trgld = np.zeros((trglxyz.shape[0],3,3))
        trgld[:,0] = trglxyz[:,2] - trglxyz[:,1]
        trgld[:,1] = trglxyz[:,0] - trglxyz[:,2]
        trgld[:,2] = trglxyz[:,1] - trglxyz[:,0]
        d21 = trgld[:,0]
        d02 = trgld[:,1]
        d10 = trgld[:,2]
        normvec = np.cross(d10, -d02)
        area = .5*np.sqrt(((normvec*normvec).sum(axis=1)))
        cmin = np.argmin(np.abs(normvec), axis=1)

        # choose the axis (x, y, z) that is most
        # orthogonal to normvec
        orth = np.zeros((nt, 3), dtype=np.float64)

        rows = np.ogrid[:nt]
        orth[rows, cmin] = 1.

        # compute a local x axis and a local y axis that are
        # mutually orthogonal and that both lie in the
        # plane of the triangle
        trgl_yaxis = np.cross(normvec, orth)
        leny = np.sqrt((trgl_yaxis*trgl_yaxis).sum(axis=1))
        leny[leny==0] = 1.
        trgl_yaxis /= leny[:,np.newaxis]
        trgl_xaxis = np.cross(trgl_yaxis, normvec)
        lenx = np.sqrt((trgl_xaxis*trgl_xaxis).sum(axis=1))
        lenx[lenx==0] = 1.
        trgl_xaxis /= lenx[:,np.newaxis]
        trglw = np.zeros((trglxyz.shape[0], 3, 2), dtype=np.float64)

        # difference vector
        trglw[:,:,0] = (trgld*trgl_xaxis[:,np.newaxis,:]).sum(axis=2)
        trglw[:,:,1] = (trgld*trgl_yaxis[:,np.newaxis,:]).sum(axis=2)
        dT = area.copy()
        dT[dT==0.] = 1.
        trglm = trglw / np.sqrt(dT)[:,np.newaxis,np.newaxis]

        # create a sparse matrix.  For each non-zero point
        # in the matrix need the trgl index, the pt index, and
        # the value at that point
        mx = trglm[:,:,0]
        my = trglm[:,:,1]
        mxval = mx.flatten()
        myval = my.flatten()

        # triangle indices
        mtind = np.indices((nt, 3))[0].flatten()

        # point indices
        mptind = trgls.flatten()
        m_sparse = np.stack((mtind, mptind, mxval, myval), axis=1)

        # constraint indices
        cindex = constraints[:,0].astype(np.int64)
        ptindex = np.arange(npt)

        # boolean array, length npt: whether point is
        # free (True) or constrained (False)
        # isfree = np.logical_not(np.isin(ptindex, cindex))
        # print(isfree)
        # print(m_sparse)
        isfree = np.full(points.shape[0], True, dtype=np.bool_)
        isfree[cindex] = False

        # free points in sparse array (m_sparse[:,1] contains
        # point index)
        mf_sparse = m_sparse[np.logical_not(np.isin(m_sparse[:,1], constraints[:,0]))]

        # pt index to mf pt index
        # each element either contains the corresponding mf pt index,
        # or -1 if original point is constrained
        pt2mf = np.full((npt), -1, dtype=np.int64)
        pt2mf[isfree] = np.arange(nfp)

        # renumber the pt indices in mf_sparse to use
        # the free-point indexing
        mf_sparse[:,1] = pt2mf[mf_sparse[:,1].astype(np.int64)]

        use_weights = False
        if self.initial_points is not None and (self.ip_weight > 0 or self.ip_weights is not None):
            use_weights = True
            if self.ip_weights is None:
                weights = np.full(points.shape[0], self.ip_weight, np.float64)
            else:
                weights = self.ip_weights
            # wf = self.ip_weights[isfree]

            # wf_sparse = np.stack((pt2mf[isfree], mptind[isfree], wf[:,0], wf[:,1]), axis=1)
            # wf_sparse = np.stack((pt2mf[isfree], mptind[isfree], wf[:,0], wf[:,1]), axis=1)
            # diagonal matrix with weights of free points
            wf_sparse = np.stack((np.ogrid[:nfp], np.ogrid[:nfp], weights[isfree]), axis=1)
            # wf = sparse.csr_array((weights[isfree], (np.ogrid[:nfp], np.ogrid[:nfp])), shape=(nfp, nfp))
            wf = sparse.csr_array((wf_sparse[:,2], (wf_sparse[:,0],wf_sparse[:,1])), shape=(nfp, nfp))
            wf_ul = wf_sparse.copy()
            wf_ul[:,0] += 2*nt
            wf_lr = wf_sparse.copy()
            wf_lr[:,0] += 2*nt+nfp
            wf_lr[:,1] += nfp
            # w1fr = wf_sparse[:,(0,1,2)]
            # w2fr = wf_sparse[:,(0,1,3)]
            # w1f = sparse.csr_array((wf_sparse[:,2], (wf_sparse[:,0], wf_sparse[:,1])), shape=(nfp, nfp))
            bw1 = wf @ self.initial_points[isfree,0]
            bw2 = wf @ self.initial_points[isfree,1]

        # pinned points in sparse array
        mp_sparse = m_sparse[np.isin(m_sparse[:,1], constraints[:,0])]

        # pt index to mp pt index
        # each element either contains the corresponding mp pt index,
        # or -1 if original point is free
        pt2mp = np.full((npt), -1, dtype=np.int64)
        pt2mp[~isfree] = np.arange(ncp)

        # renumber the pt indices in mp_sparse to use
        # the pinned-point indexing
        mp_sparse[:,1] = pt2mp[mp_sparse[:,1].astype(np.int64)]

        '''
        # to compute b, need the full (non-sparse) version
        # of mp.  Create this by first creating an
        # array with all zeros, and then filling in
        # the points from mp_sparse
        mp_full = np.zeros((nt, ncp, 2), dtype=np.float64)
        mp_full[mp_sparse[:,0].astype(np.int64), mp_sparse[:,1].astype(np.int64)] = mp_sparse[:, (2,3)]
        m1p = mp_full[:,:,0]
        m2p = mp_full[:,:,1]
        '''
        ''''''
        m1pr = mp_sparse[:,(0,1,2)]
        m1p = sparse.csr_array((m1pr[:,2], (m1pr[:,0], m1pr[:,1])), shape=(nt, ncp))
        m2pr = mp_sparse[:,(0,1,3)]
        m2p = sparse.csr_array((m2pr[:,2], (m2pr[:,0], m2pr[:,1])), shape=(nt, ncp))
        ''''''
        u1p = constraints[:,1]
        u2p = constraints[:,2]
        bu1 = -m1p@u1p + m2p@u2p
        bu2 = -m2p@u1p - m1p@u2p
        if use_weights:
            b = np.concatenate((bu1, bu2, bw1, bw2))
        else:
            b = np.concatenate((bu1, bu2))
        '''
        b = np.concatenate((
            -m1p@u1p + m2p@u2p,
            -m2p@u1p - m1p@u2p))
        '''
        # print("b", b)

        m1f = mf_sparse[:,(0,1,2)]
        m2f = mf_sparse[:,(0,1,3)]
        ul = m1f.copy()
        ur = m2f.copy()
        ur[:,1] += nfp
        ur[:,2] *= -1.
        ll = m2f.copy()
        ll[:,0] += nt
        lr = m1f.copy()
        lr[:,0] += nt
        lr[:,1] += nfp

        if use_weights:
            A_csr_in = np.concatenate((ul, ur, ll, lr, wf_ul, wf_lr), axis=0)
        else:
            A_csr_in = np.concatenate((ul, ur, ll, lr), axis=0)
        AV = A_csr_in[:,2]
        AI = A_csr_in[:,0]
        AJ = A_csr_in[:,1]
        if use_weights:
            Acsr = sparse.csr_array((AV, (AI, AJ)), shape=(2*nt+2*nfp, 2*nfp))
        else:
            Acsr = sparse.csr_array((AV, (AI, AJ)), shape=(2*nt, 2*nfp))
        timer.time("  created sparse array")

        # Acsr = A_sparse.tocsr()
        At = Acsr.transpose()
        # print(b)
        # print(A_coo)
        # print(At@Acsr)
        try:
            lu = sparse.linalg.splu(At@Acsr)
            # print("splu solved")
            x = lu.solve(At@b)
        except Exception as e:
            print("SPLU exception:", e)
            # print(b)
            # print(A_coo)
            # print(At@Acsr)
            return None
        uv = np.zeros((npt, 2), dtype=np.float64)
        # mf2pt = np.arange(npt)[pt2mf >= 0]
        uv[isfree, :] = x.reshape(2, nfp).transpose()
        uv[~isfree, :] = constraints[:,(1,2)]
        # print("uv", uv)
        return uv

if __name__ == '__main__':
    from trgl_fragment import TrglFragment
    points = None
    trgls = None
    tpoints = None
    timer = Timer()
    out_file = "test_out.obj"
    if len(sys.argv) > 1:
        obj_file = sys.argv[1]
        trgl_frags = TrglFragment.load(obj_file)
        if trgl_frags is not None and len(trgl_frags) > 0:
            trgl_frag = trgl_frags[0]
            points = trgl_frag.gpoints
            tpoints = trgl_frag.gtpoints
            if len(tpoints) != len(points):
                if len(tpoints) != 0:
                    print("discrepancy between tpoints and points lengths")
                tpoints = None
            trgls = trgl_frag.trgls
        if len(sys.argv) > 2:
            out_file = sys.argv[2]

    if points is None or trgls is None:
        print("couldn't read")
        exit()
    # print(points.shape, points.dtype, trgls.shape, trgls.dtype)
    timer.time("read")
    lscm = UVMapper(points, trgls)
    pt0, pt1 = lscm.getTwoAdjacentBoundaryPoints(0)
    # print("pt0, pt1", pt0, pt1)
    # pt2, pt3 = lscm.getTwoAdjacentBoundaryPoints(-1)
    # print("pt2, pt3", pt2, pt3)
    # ptmn, ptmx = lscm.getTwoAdjacentBoundaryPoints()
    # print("ptmn, ptmx", ptmn, ptmx)
    lscm.constraints = np.array([[pt0, 0., 0.], [pt1, 1., 0.]], dtype=np.float64)
    '''
    lscm.constraints = np.array(
            [
            [1., 600., 1500.],
            [2., 500., 1600.],
            [3., 400., 1500.],
            [4., 500., 1400.],
            ], dtype=np.float64)
    '''
    if tpoints is not None:
        lscm.initial_points = tpoints
        weight = .000001
        # weight = 0.
        # lscm.ip_weight = weight
        weights = np.full(tpoints.shape[0], weight, dtype=np.float64)
        # weights[0] = 0.
        lscm.ip_weights = weights
    # uvs = lscm.computeUvs()
    # uvs = lscm.computeUvsFromAngles()
    # uvs = lscm.computeUvsFromXyzs()
    uvs = lscm.computeUvsFromABF()
    timer.time("computed uvs")
    # lscm.angleQuality(lscm.angles)
    uvmin = uvs.min(axis=0)
    uvmax = uvs.max(axis=0)
    # print("uv min max", uvmin, uvmax)
    np.set_printoptions(precision=3)
    print(uvs)
    duv = uvmax-uvmin
    if (duv>0).all():
        uvs -= uvmin
        uvs /= duv
    trgl_frag.gtpoints = uvs
    trgl_frag.save(Path(out_file))
    timer.time("saved")

