import sys
sys.path.append('..')
from pathlib import Path

import numpy as np
from scipy import sparse

from trgl_fragment import TrglFragment
from base_fragment import BaseFragment
from utils import Utils

class LSCM:
    def __init__(self, points, trgls):
        self.points = points
        self.trgls = trgls

        # shape (nc, 3), and dtype float32,
        # with each row: pt index, u, v
        # note that pt index is stored as a float
        self.constraints = None

        # shape (nt, 3) and dtype float32
        # with each row containing the 3 angles (in radians)
        # at the 3 triangle vertices
        self.angles = None

        # shape (nt, 3) and dtype int32
        # the trgl indices of the 3 neighboring trgls
        # (first is the trgl opposite vertex 0)
        # if no neighbor, set to -1
        # created by self.createNeighbors()
        self.neighbors = None

        # shape (nb, 2) and dtype int32,
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
        # print("bds", self.boundaries)

    def getTwoAdjacentBoundaryPoints(self):
        if lscm.boundaries is None:
            lscm.createBoundaries()
        if lscm.boundaries is None or len(lscm.boundaries) == 0:
            print("No boundaries!")
            return None
        bd = lscm.boundaries[0]
        pt0 = lscm.trgls[bd[0], (bd[1]+1)%3]
        pt1 = lscm.trgls[bd[0], (bd[1]+2)%3]
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
        # print(dtrgl[:2])
        # print(d02[:2])
        # print(d10[:2])
        # print(d21[:2])
        # print("dtrgl", dtrgl)

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
        # print(trglangles)
        self.angles = trglangles
        # print("angles", self.angles)

    # if sum(angles around a point) is > 2 pi, reduce
    # the angles proportionally.  Don't make any changes
    # if the sum is < 2 pi (this can happen if the point
    # is on a boundary).
    def adjustAngles(self):
        angles = self.angles.copy()
        if angles is None:
            print("adjustAngles: angles not set!")
            return
        points = self.points
        trgls = self.trgls
        sums = np.zeros(len(points), dtype=np.float32)
        # indexed_angles = np.stack((trgls.flatten(), angles.flatten()), axis=1)
        # sums[indexed_angles[:,0]] += indexed_angles[:,1]
        print(trgls.shape, angles.shape)
        # sums[trgls.flatten()] += angles.flatten()
        # https://stackoverflow.com/questions/60481343/numpy-sum-over-repeated-entries-in-index-array
        np.add.at(sums, trgls.flatten(), angles.flatten())
        sums /= 2*np.pi
        factors = np.full(len(points), 1., dtype=np.float32)
        factors[sums > 1.] /= sums[sums > 1.]
        print("adjusted", (factors < 1).sum())
        angles *= factors[trgls]
        print("adjusted", (angles[:,:] != self.angles[:,:]).sum())
        self.angles = angles


        # print("sums")
        # print(sums)


    def computeUvs(self):
        return self.computeUvsFromXyzs()

    def computeUvsFromAngles(self):
        timer = Utils.Timer()
        if self.angles is None:
            self.createAngles()
            timer.time("angles created")
            # self.adjustAngles()
            # timer.time("angles adjusted")
        angles = self.angles
        points = self.points
        trgls = self.trgls
        constraints = self.constraints
        # if angles is None or len(angles) < 3:
        if angles is None:
            print("Not enough angles")
            return None
        # if points is None or len(points) < 3:
        #     print("Not enough points")
        #     return None
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
        rindex = np.argmax(np.abs(angles), axis=1)
        # The point with the largest angle should be moved to index 2
        axis1 = np.full(nt, 1, dtype=np.int32)
        print(angles.shape, rindex.shape, axis1.shape)

        # rtrgls = np.roll(trgls, 2-rindex, axis1)
        # rangles = np.roll(angles, (2-rindex).tolist(), axis1.tolist())
        # print(angles)
        # print(2-rindex)

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
        # print(rangles)

        # print("constraints", constraints)
        # print("rangles", rangles)
        # print("rtrgls", rtrgls)
        angles = None
        trgls = None

        # numerator
        num = np.sin(rangles[:,1])
        # denominator
        den = np.sin(rangles[:,2])
        # abs(den) is >= abs(num), due to the shift above
        # if den == 0, then num must == 0 too.
        den[den == 0] = 1.
        ratio = num/den
        # print("ratio", ratio)
        sn = ratio*np.sin(rangles[:,0])
        cs = ratio*np.cos(rangles[:,0])
        ones = np.full(ratio.shape, 1., dtype=np.float32)
        zeros = np.zeros(ratio.shape, dtype=np.float32)
        # twos = np.full(ratio.shape, 2., dtype=np.float32)
        # threes = np.full(ratio.shape, 3., dtype=np.float32)
        # fours = np.full(ratio.shape, 4., dtype=np.float32)

        # m has shape nt, 3, 2, 2
        # in other words, m has a 2x2 matrix for
        # each vertex of each trgl
        m = np.zeros((nt, 3, 2, 2), dtype=np.float32)

        '''
        m[:, 0, 0, 0] = cs-1
        m[:, 0, 0, 1] = sn
        m[:, 0, 1, 0] = -sn
        m[:, 0, 1, 1] = cs-1

        m[:, 1, 0, 0] = -cs
        m[:, 1, 0, 1] = -sn
        m[:, 1, 1, 0] = sn
        m[:, 1, 1, 1] = -cs

        m[:, 2, 0, 0] = ones
        m[:, 2, 0, 1] = zeros
        m[:, 2, 1, 0] = zeros
        m[:, 2, 1, 1] = ones
        '''

        m[:, 0] = np.array([[cs-1, sn],  [-sn, cs-1]]).transpose(2,0,1)
        # m[:, 0] = np.array([[ones, twos],  [threes, fours]]).transpose(2,0,1)
        # print("m after 0")
        # print(m)
        # exit()
        m[:, 1] = np.array([[-cs,  -sn], [sn,  -cs]]).transpose(2,0,1)
        m[:, 2] = np.array([[ones, zeros], [zeros, ones]]).transpose(2,0,1)
        # print("m")
        # print(m)

        minds = np.indices(m.shape)
        # print("minds")
        # print(minds)

        '''
        mflat = np.stack(
                (minds[0].flatten(),
                 minds[1].flatten(),
                 minds[2].flatten(),
                 minds[3].flatten(),
                 m.flatten()), axis=1)
        '''

        # print((minds[0].flatten()*2+minds[2].flatten()).shape)
        # print((rtrgls[:, minds[1].flatten()]*2+minds[3].flatten()).shape)
        # print(m.flatten().shape)
        # print(minds[1].flatten())
        # print(rtrgls)
        # print(rtrgls[:, minds[1].flatten()]*2)
        m_sparse = np.stack((
            minds[0].flatten()*2+minds[2].flatten(),
            rtrgls[minds[0].flatten(), minds[1].flatten()]*2+minds[3].flatten(),
            m.flatten()), axis=1)

        # print("m_sparse")
        # print(m_sparse)

        '''
        # create a sparse matrix.  For each non-zero point
        # in the matrix need the trgl index, the pt index, and
        # the value at that point
        # mx = trglm[:,:,0]
        # my = trglm[:,:,1]
        # mxval = mx.flatten()
        # myval = my.flatten()

        # triangle indices
        # mtind = np.indices((nt, 3))[0].flatten()
        mtind = np.arange(nt).astype(np.int32)

        # point indices
        # mptind = trgls.flatten()
        # m_sparse = np.stack((mtind, mptind, mxval, myval), axis=1)
        # print(mtind.shape, mptind.shape, sn.shape, (cs-1).shape)
        # m_sparse = np.stack((mtind, mptind, 
        #                      cs-1, sn, -sn, cs-1,
        #                      -cs, -sn, sn, -cs,
        #                      1, 0, 0, 1), axis=1)

        # ms000 = np.stack(4*mtind, 2*trgls[:,0], m[:, :, 0, 0])
        # ms001 = np.stack(4*mtind, 2*trgls[:,0], m[:, :, 0, 0])
        # print(mtind.shape, trgls[:,0].shape, m[:, 0].reshape(-1,4).shape)
        ms0 = np.concatenate((mtind[:,np.newaxis], rtrgls[:,0][:,np.newaxis], m[:, 0].reshape(-1,4)), axis=1)
        # print(ms0)
        ms1 = np.concatenate((mtind[:,np.newaxis], rtrgls[:,1][:,np.newaxis], m[:, 1].reshape(-1,4)), axis=1)
        ms2 = np.concatenate((mtind[:,np.newaxis], rtrgls[:,2][:,np.newaxis], m[:, 2].reshape(-1,4)), axis=1)
        msall = np.concatenate((ms0, ms1, ms2), axis=0)

        # TODO: need to split free from pinned points 
        # in msall

        msall[:,0] *= 2
        msall[:,1] *= 2
        ms00 = msall[:,(0,1,2+0)]
        ms00[:,0] += 0
        ms00[:,1] += 0
        ms01 = msall[:,(0,1,2+1)]
        ms01[:,0] += 0
        ms01[:,1] += 1
        ms10 = msall[:,(0,1,2+2)]
        ms10[:,0] += 1
        ms10[:,1] += 0
        ms11 = msall[:,(0,1,2+3)]
        ms11[:,0] += 1
        ms11[:,1] += 1

        m_sparse = np.concatenate((ms00, ms01, ms10, ms11), axis=0)
        '''

        # constraint indices
        # NOTE that cindex is not sorted
        cindex = constraints[:,0].astype(np.int32)
        ptindex = np.arange(npt)

        # boolean array, length na: whether point is
        # free (True) or constrained (False)
        isfree = np.logical_not(np.isin(ptindex, cindex))
        # print("isfree", isfree)

        # free points in sparse array (m_sparse[:,1] contains
        # point index)
        mf_sparse = m_sparse[np.logical_not(np.isin(m_sparse[:,1]//2, constraints[:,0]))]
        # print("mf_sparse")
        # print(mf_sparse)
        # print("m, mf", m_sparse.shape, mf_sparse.shape)
        # print(m_sparse)

        # pt index to mf pt index
        # each element either contains the corresponding mf pt index,
        # or -1 if original point is constrained.
        # Keep in mind that the mf point index equals
        # 2 times the original point index, plus 0 or 1
        pt2mf = np.full((2*npt), -1, dtype=np.int32)
        # indexes of free points
        free_ind = np.nonzero(isfree)[0]
        # print("where", np.nonzero(isfree)[0])
        pt2mf[2*free_ind] = 2*np.arange(nfp)
        pt2mf[2*free_ind+1] = 2*np.arange(nfp)+1
        # pt2mf[isfree] = np.arange(nfp)
        # print("pt2mf", pt2mf)

        # renumber the pt indices in mf_sparse to use
        # the free-point indexing

        mf_sparse[:,1] = pt2mf[mf_sparse[:,1].astype(np.int32)]
        # print(mf_sparse)
        # print("mf_sparse renumbered")
        # print(mf_sparse)

        AV = mf_sparse[:,2]
        AI = mf_sparse[:,0]
        AJ = mf_sparse[:,1]
        A_sparse = sparse.coo_array((AV, (AI, AJ)), shape=(2*nt, 2*nfp))

        # A_coo = np.concatenate((ms0, ms1, ms2), axis=0)
        # print(A_coo)
        '''
        AV = mf_sparse[:,2]
        AI = mf_sparse[:,0]
        AJ = mf_sparse[:,1]
        # A_sparse = sparse.coo_array((AV, (AI, AJ)), shape=(3*nt, 4*nfp))
        A_sparse = sparse.coo_array((AV, (AI, AJ)), shape=(2*nt, 2*nfp))
        '''

        # coo
        # A_coo = np.concatenate((ms0, ms1, ms2), axis=0)
        # complete matrix (free plus pinned) has nt
        # rows, and 2*na columns.


        # pinned points in sparse array
        mp_sparse = m_sparse[np.isin(m_sparse[:,1]//2, constraints[:,0])]
        # print("mp_sparse", mp_sparse)

        # pt index to mp pt index
        # each element either contains the corresponding mp pt index,
        # or -1 if original point is free
        pt2mp = np.full((2*npt), -1, dtype=np.int32)
        # pt2mp[~isfree] = np.arange(ncp)
        # pinned_ind = np.nonzero(~isfree)[0]
        # pt2mp[2*pinned_ind] = 2*np.arange(ncp)
        # pt2mp[2*pinned_ind+1] = 2*np.arange(ncp)+1
        # NOTE: cindex is not sorted
        pt2mp[2*cindex] = 2*np.arange(ncp)
        pt2mp[2*cindex+1] = 2*np.arange(ncp)+1

        # renumber the pt indices in mp_sparse to use
        # the pinned-point indexing
        # print("mp_sparse")
        # print(mp_sparse)
        mp_sparse[:,1] = pt2mp[mp_sparse[:,1].astype(np.int32)]
        # print(mp_sparse)

        # to compute b, need the full (non-sparse) version
        # of mp.  Create this by first creating an
        # array with all zeros, and then filling in 
        # the points from mp_sparse
        mp_full = np.zeros((2*nt, 2*ncp), dtype=np.float32)
        mp_full[mp_sparse[:,0].astype(np.int32), mp_sparse[:,1].astype(np.int32)] = mp_sparse[:, 2]
        # print(mp_full)

        b = -mp_full @ constraints[:,(1,2)].flatten()

        timer.time("  created arrays")

        # print(mf_sparse)
        # print(b)

        # res = sparse.linalg.lsqr(A_sparse, b)
        res = sparse.linalg.lsmr(A_sparse, b)
        timer.time("  solved lsqr")
        x = res[0]
        diagnostics = res[1:]
        # print("x", x)
        print("diagnostics", diagnostics)
        uv = np.zeros((npt, 2), dtype=np.float32)
        # mf2pt = np.arange(npt)[pt2mf >= 0]
        # uv[isfree, :] = x.reshape(2, nfp).transpose()
        uv[isfree, :] = x.reshape(nfp, 2)
        # uv[~isfree, :] = constraints[:,(1,2)]
        uv[cindex, :] = constraints[:,(1,2)]
        # print("uv", uv)
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

        # print("input", points.shape, points.dtype, trgls.shape, trgls.dtype)
        trglxyz = points[trgls]
        # print("trglxyz", trglxyz.shape)
        # print(trglxyz)
        trgld = np.zeros((trglxyz.shape[0],3,3))
        trgld[:,0] = trglxyz[:,2] - trglxyz[:,1]
        trgld[:,1] = trglxyz[:,0] - trglxyz[:,2]
        trgld[:,2] = trglxyz[:,1] - trglxyz[:,0]
        # print("trgld", trgld.shape)
        # print(trgld)
        d21 = trgld[:,0]
        d02 = trgld[:,1]
        d10 = trgld[:,2]
        normvec = np.cross(d10, -d02)
        # print("normvec", normvec.shape)
        area = .5*np.sqrt(((normvec*normvec).sum(axis=1)))
        # print("area", area.shape)
        cmin = np.argmin(np.abs(normvec), axis=1)
        # print("cmin", cmin.shape, cmin.dtype)

        # choose the axis (x, y, z) that is most
        # orthogonal to normvec
        orth = np.zeros((nt, 3), dtype=np.float32)
        # ones = np.full(nt, 1., dtype=np.float32)
        # print("orth 0", orth.shape)
        # For some reason, this way is very slow!
        # orth[:,cmin] = 1.

        rows = np.ogrid[:nt]
        orth[rows, cmin] = 1.
        # orth[:,cmin] = ones
        # print("orth", orth.shape)

        # compute a local x axis and a local y axis that are
        # mutually orthogonal and that both lie in the
        # plane of the triangle
        trgl_yaxis = np.cross(normvec, orth)
        leny = np.sqrt((trgl_yaxis*trgl_yaxis).sum(axis=1))
        leny[leny==0] = 1.
        trgl_yaxis /= leny[:,np.newaxis]
        # print("trgl_yaxis", trgl_yaxis.shape)
        trgl_xaxis = np.cross(trgl_yaxis, normvec)
        lenx = np.sqrt((trgl_xaxis*trgl_xaxis).sum(axis=1))
        lenx[lenx==0] = 1.
        trgl_xaxis /= lenx[:,np.newaxis]
        # print("trgl_xaxis", trgl_xaxis.shape)
        trglw = np.zeros((trglxyz.shape[0], 3, 2), dtype=np.float32)

        # difference vector
        trglw[:,:,0] = (trgld*trgl_xaxis[:,np.newaxis,:]).sum(axis=2)
        trglw[:,:,1] = (trgld*trgl_yaxis[:,np.newaxis,:]).sum(axis=2)
        dT = area.copy()
        dT[dT==0.] = 1.
        trglm = trglw / np.sqrt(dT)[:,np.newaxis,np.newaxis]
        # print("trglm", trglm.shape)
        # trglwx = trglw[:,:,0]
        # trglwy = trglw[:,:,1]

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
        # print("m_sparse", m_sparse)

        # constraint indices
        cindex = constraints[:,0].astype(np.int32)
        ptindex = np.arange(npt)

        # boolean array, length npt: whether point is
        # free (True) or constrained (False)
        isfree = np.logical_not(np.isin(ptindex, cindex))

        # free points in sparse array (m_sparse[:,1] contains
        # point index)
        mf_sparse = m_sparse[np.logical_not(np.isin(m_sparse[:,1], constraints[:,0]))]
        # print("mf_sparse", mf_sparse)

        # pt index to mf pt index
        # each element either contains the corresponding mf pt index,
        # or -1 if original point is constrained
        pt2mf = np.full((npt), -1, dtype=np.int32)
        pt2mf[isfree] = np.arange(nfp)

        # renumber the pt indices in mf_sparse to use
        # the free-point indexing
        mf_sparse[:,1] = pt2mf[mf_sparse[:,1].astype(np.int32)]
        # print("mf_sparse", mf_sparse)

        # pinned points in sparse array
        mp_sparse = m_sparse[np.isin(m_sparse[:,1], constraints[:,0])]
        # print("mp_sparse", mp_sparse)

        # pt index to mp pt index
        # each element either contains the corresponding mp pt index,
        # or -1 if original point is free
        pt2mp = np.full((npt), -1, dtype=np.int32)
        pt2mp[~isfree] = np.arange(ncp)

        # renumber the pt indices in mp_sparse to use
        # the pinned-point indexing
        mp_sparse[:,1] = pt2mp[mp_sparse[:,1].astype(np.int32)]
        # print("mp_sparse", mp_sparse)

        # to compute b, need the full (non-sparse) version
        # of mp.  Create this by first creating an
        # array with all zeros, and then filling in 
        # the points from mp_sparse
        mp_full = np.zeros((nt, ncp, 2), dtype=np.float32)
        mp_full[mp_sparse[:,0].astype(np.int32), mp_sparse[:,1].astype(np.int32)] = mp_sparse[:, (2,3)]
        # print("mp_full", mp_full)
        u1p = constraints[:,1]
        u2p = constraints[:,2]
        m1p = mp_full[:,:,0]
        m2p = mp_full[:,:,1]
        b = np.concatenate((
            -m1p@u1p + m2p@u2p,
            -m2p@u1p - m1p@u2p))
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
        A_coo = np.concatenate((ul, ur, ll, lr), axis=0)
        # print("A", A_coo)
        AV = A_coo[:,2]
        AI = A_coo[:,0]
        AJ = A_coo[:,1]
        A_sparse = sparse.coo_array((AV, (AI, AJ)), shape=(2*nt, 2*nfp))
        timer.time("  created sparse array")

        # res = sparse.linalg.lsqr(A_sparse, b)
        res = sparse.linalg.lsmr(A_sparse, b)
        timer.time("  solved lsqr")
        x = res[0]
        diagnostics = res[1:]
        # print("x", x)
        print("diagnostics", diagnostics)
        uv = np.zeros((npt, 2), dtype=np.float32)
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
    lscm = LSCM(points, trgls)
    pt0, pt1 = lscm.getTwoAdjacentBoundaryPoints()
    lscm.constraints = np.array([[pt0, 0., 0.], [pt1, 1., 0.]], dtype=np.float32)
    # lscm.constraints = np.array([[0, 100., 2200.], [3, 200., 2300.]], dtype=np.float32)
    # lscm.constraints = np.array([[0, 0., 0.], [len(points)-1, 1., 1.]], dtype=np.float32)
    # lscm.constraints = np.array([[0, 0., 0.], [len(points)-1, 1., 0.]], dtype=np.float32)
    # lscm.constraints = np.array([[0, 0., 0.], [1, 1., 0.]], dtype=np.float32)
    # uvs = lscm.computeUvs()
    uvs = lscm.computeUvsFromAngles()
    # uvs = lscm.computeUvsFromXyzs()
    timer.time("computed uvs")
    uvmin = uvs.min(axis=0)
    uvmax = uvs.max(axis=0)
    print("uv min max", uvmin, uvmax)
    duv = uvmax-uvmin
    if (duv>0).all():
        uvs -= uvmin
        uvs /= duv
    trgl_frag.gtpoints = uvs
    trgl_frag.save(Path(out_file))
    timer.time("saved")

