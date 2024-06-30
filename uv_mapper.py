import sys
sys.path.append('..')
from pathlib import Path

import numpy as np
from scipy import sparse

from trgl_fragment import TrglFragment
from base_fragment import BaseFragment
from utils import Utils

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

    def createNeighbors(self):
        if self.trgls is None or self.points is None:
            print("createNeighbors: no triangles specified!")
            return
        print("Creating neighbors")
        self.neighbors = BaseFragment.findNeighbors(self.trgls)

    def createBoundaries(self):
        print("Creating boundaries")
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
        is_on_boundary = np.zeros(npt, dtype=np.bool_)
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
        print("creating angles")
        points = self.points
        trgls = self.trgls
        if points is None or len(points) < 3:
            print("Not enough points")
            return None
        if trgls is None or len(trgls) == 0:
            print("No triangles")
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

    # adjust angles if point is not on a 
    # boundary, and sum is < (2 pi - min_deficit).
    # if so, reduce the angles proportionally.  
    def adjustedAngles(self, min_deficit):
        angles = self.angles.copy()
        if angles is None:
            print("adjustAngles: angles not set!")
            return
        points = self.points
        trgls = self.trgls
        sums = np.zeros(len(points), dtype=np.float64)
        print(trgls.shape, angles.shape)
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
        timer = Utils.Timer()
        if self.angles is None:
            self.createAngles()
            timer.time("angles created")
        adjusted_angles = self.adjustedAngles(1.)
        points = self.points
        trgls = self.trgls
        if adjusted_angles is None:
            print("Not enough angles")
            return None
        if trgls is None or len(trgls) == 0:
            print("No triangles")
            return None

        interior_points_bool = np.logical_and(~self.onBoundaryArray(), self.usedInTrglArray())
        interior_points = np.where(interior_points_bool)[0]

        nt = trgls.shape[0]
        npt = points.shape[0]
        nipt = interior_points.shape[0]
        print("nt, npt, nipt", nt, npt, nipt)
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
        print("x min max", x.min(), x.max())

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
        pt2ipt = np.full((npt), -1, dtype=np.int64)
        pt2ipt[interior_points] = np.arange(nipt)

        pt_angle_sum = np.zeros(npt, dtype=np.float64)
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
        self.createAngles()
        self.angleQuality(self.angles)
        for i in range(10):
            abf_angles = self.linABF()
            if abf_angles is None:
                print("linABF failed!")
                return
            self.angleQuality(abf_angles)
            self.angles = abf_angles
            mwe = self.maxWheelError(abf_angles)
            if mwe < 1.e-5:
                print("wheel error is small enough at iteration", i+1)
                break
        return self.computeUvsFromAngles()

    def computeUvsFromAngles(self):
        timer = Utils.Timer()
        if self.angles is None:
            self.createAngles()
            timer.time("angles created")
            # self.adjustAngles()
            adjusted = self.adjustedAngles(1.)
            # timer.time("angles adjusted")
        angles = self.angles
        points = self.points
        trgls = self.trgls
        constraints = self.constraints
        if angles is None:
            print("Not enough angles")
            return None
        if trgls is None or len(trgls) == 0:
            print("No triangles")
            return None
        if constraints is None or len(constraints) < 2:
            print("Not enough constraints")
            return None

        nt = trgls.shape[0]
        npt = points.shape[0]
        ncp = constraints.shape[0]
        nfp = npt-ncp

        # find the index of the point with the
        # largest angle
        rindex = np.argmax(np.sin(angles), axis=1)
        # The point with the largest angle should be moved to index 2
        axis1 = np.full(nt, 1, dtype=np.int64)
        print(angles.shape, rindex.shape, axis1.shape)

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

        m[:, 0] = np.array([[cs-1, sn],  [-sn, cs-1]]).transpose(2,0,1)
        m[:, 1] = np.array([[-cs,  -sn], [sn,  -cs]]).transpose(2,0,1)
        m[:, 2] = np.array([[ones, zeros], [zeros, ones]]).transpose(2,0,1)

        minds = np.indices(m.shape)

        m_sparse = np.stack((
            minds[0].flatten()*2+minds[2].flatten(),
            rtrgls[minds[0].flatten(), minds[1].flatten()]*2+minds[3].flatten(),
            m.flatten()), axis=1)

        # constraint indices
        # NOTE that cindex is not sorted
        cindex = constraints[:,0].astype(np.int64)
        ptindex = np.arange(npt)

        # boolean array, length na: whether point is
        # free (True) or constrained (False)
        isfree = np.logical_not(np.isin(ptindex, cindex))

        # free points in sparse array (m_sparse[:,1] contains
        # point index)
        mf_sparse = m_sparse[np.logical_not(np.isin(m_sparse[:,1]//2, constraints[:,0]))]

        # pt index to mf pt index
        # each element either contains the corresponding mf pt index,
        # or -1 if original point is constrained.
        # Keep in mind that the mf point index equals
        # 2 times the original point index, plus 0 or 1
        pt2mf = np.full((2*npt), -1, dtype=np.int64)
        # indexes of free points
        free_ind = np.nonzero(isfree)[0]
        pt2mf[2*free_ind] = 2*np.arange(nfp)
        pt2mf[2*free_ind+1] = 2*np.arange(nfp)+1

        # renumber the pt indices in mf_sparse to use
        # the free-point indexing

        mf_sparse[:,1] = pt2mf[mf_sparse[:,1].astype(np.int64)]

        AV = mf_sparse[:,2].astype(np.float64)
        AI = mf_sparse[:,0]
        AJ = mf_sparse[:,1]
        A_sparse = sparse.coo_array((AV, (AI, AJ)), shape=(2*nt, 2*nfp))

        # print("A shape", 2*nt, 2*nfp)
        # print(AI.min(), AI.max(), AJ.min(), AJ.max())

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

        b = -mp_full @ constraints[:,(1,2)].flatten()

        timer.time("  created arrays")

        Acsr = A_sparse.tocsr()
        At = Acsr.transpose()
        lu = sparse.linalg.splu(At@Acsr)
        x = lu.solve(At@b)
        uv = np.zeros((npt, 2), dtype=np.float64)
        uv[isfree, :] = x.reshape(nfp, 2)
        uv[cindex, :] = constraints[:,(1,2)]
        return uv

    def computeUvsFromXyzs(self):
        timer = Utils.Timer()
        points = self.points
        trgls = self.trgls
        constraints = self.constraints
        if points is None or len(points) < 3:
            print("Not enough points")
            return None
        if trgls is None:
            print("No triangles")
            return None
        if constraints is None or len(constraints) < 2:
            print("Not enough constraints")
            return None

        nt = trgls.shape[0]
        npt = points.shape[0]
        ncp = constraints.shape[0]
        nfp = npt-ncp
        print("nt, npt, ncp", nt, npt, ncp)

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
        isfree = np.logical_not(np.isin(ptindex, cindex))

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

        # to compute b, need the full (non-sparse) version
        # of mp.  Create this by first creating an
        # array with all zeros, and then filling in 
        # the points from mp_sparse
        mp_full = np.zeros((nt, ncp, 2), dtype=np.float64)
        mp_full[mp_sparse[:,0].astype(np.int64), mp_sparse[:,1].astype(np.int64)] = mp_sparse[:, (2,3)]
        u1p = constraints[:,1]
        u2p = constraints[:,2]
        m1p = mp_full[:,:,0]
        m2p = mp_full[:,:,1]
        b = np.concatenate((
            -m1p@u1p + m2p@u2p,
            -m2p@u1p - m1p@u2p))

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

        A_coo = np.concatenate((ul, ur, ll, lr), axis=0)
        AV = A_coo[:,2]
        AI = A_coo[:,0]
        AJ = A_coo[:,1]
        A_sparse = sparse.coo_array((AV, (AI, AJ)), shape=(2*nt, 2*nfp))
        timer.time("  created sparse array")

        Acsr = A_sparse.tocsr()
        At = Acsr.transpose()
        lu = sparse.linalg.splu(At@Acsr)
        x = lu.solve(At@b)
        uv = np.zeros((npt, 2), dtype=np.float64)
        # mf2pt = np.arange(npt)[pt2mf >= 0]
        uv[isfree, :] = x.reshape(2, nfp).transpose()
        uv[~isfree, :] = constraints[:,(1,2)]
        # print("uv", uv)
        return uv

if __name__ == '__main__':
    points = None
    trgls = None
    timer = Utils.Timer()
    out_file = "test_out.obj"
    if len(sys.argv) > 1:
        obj_file = sys.argv[1]
        trgl_frags = TrglFragment.load(obj_file)
        if trgl_frags is not None and len(trgl_frags) > 0:
            trgl_frag = trgl_frags[0]
            points = trgl_frag.gpoints
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
    # uvs = lscm.computeUvs()
    uvs = lscm.computeUvsFromAngles()
    # uvs = lscm.computeUvsFromXyzs()
    # uvs = lscm.computeUvsFromABF()
    timer.time("computed uvs")
    uvmin = uvs.min(axis=0)
    uvmax = uvs.max(axis=0)
    # print("uv min max", uvmin, uvmax)
    duv = uvmax-uvmin
    if (duv>0).all():
        uvs -= uvmin
        uvs /= duv
    trgl_frag.gtpoints = uvs
    trgl_frag.save(Path(out_file))
    timer.time("saved")

