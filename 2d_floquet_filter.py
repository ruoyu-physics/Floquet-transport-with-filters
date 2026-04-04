import numpy as np
import matplotlib.pyplot as plt
from numpy import identity, conj, exp, pi, sqrt, vstack, hstack, zeros, heaviside, trace
from numpy.linalg import inv
from scipy.special import jv

# System Parameters
a = 1            # Neareast neighbor distance
t1 = 1            # Nearest hopping amplitude
t2 = 0          # Second nearest hopping amplitude
M = 0       # Onsite energy (Haldane)
phi = 0         # Second nearest hopping phase (Haldane)

A0 = 0.5      # drive amplitude(vector potential)
omega = 3       # drive frequency
maxorder = 5    # maximum order of Bessel functions kept

gammaL = -1        # Im[self energy]
gammaR = -1

length = 20     # system length
width = 30      # system width
trunc = 5        # truncation range (trunc=3: m=-1, 0, 1)

# Filter parameters
filt = "square"   # Filter type. <square> or <impurity>
lengthf = 20     # Length of the filter

tf = 0.25         # Coupling between sites inside the filter
tcl = sqrt(tf*t1)        # Coupling between the filter and the system
tcr = sqrt(tf*t1)

ns = 2*(width+1) # number of sites in one slice
nf = 2*(width+1)
d = nf

if(filt=="impurity"):
    tf = 5
    ti = 5           # Coupling between lattice and impurity.

    Mf = 15           # Filter onsite energy
    Mi = 1.25           # Impurity onsite energy
    tcl = 2.2
    tcr = 2.2
    gammaL = -5
    gammaR = -5
    nf = 3*(width+1) # number of sites in one silce of impurtiry band filter


# Setup parameters
backgateV = 0
vbf = 0     #backgate on the filter
vf = vbf*identity(trunc*ns)
muL = 0
muR = 0

# conveninent constants
z1 = A0*a
z2 = A0*sqrt(3)*a

angle1 = np.array([-pi/6, pi/2, 7*pi/6])
angle2 = np.array([0, 2*pi/3, 4*pi/3])

ett = -0
eta = ett*identity(trunc*ns, dtype=complex)

# Bessel function Coefficients
# vec gives the hopping vector and sign gives whether it's parrellel or opposite.
def C1(order, vec, sign):
    if(vec==0):
        return 0
    m = -order
    return 1j**m*jv(sign*m, z1)*exp(-1j*m*angle1[vec-1])

def C2(order, vec, sign):
    if(vec==0):
        return 0
    m = -order
    return 1j**m*jv(sign*m, z2)*exp(-1j*m*angle2[vec-1])

def fermi(energy, mu):
    return 1-heaviside(energy-mu, 1)

def Hn(m):
    h = np.zeros((ns,ns), dtype=complex)
    U = np.zeros((ns,ns), dtype=complex)
    for j in range(ns):
        for i in range(j+1):
            if(i==j):
                s = -np.sign(i%2-0.5) # alternating 1, -1 (-s gives alterning -1, 1)
                if(m==0):
                    h[i, j] = s*M

                U[i, j] = t2*exp(-s*1j*phi)*C2(m, 1, 1)

            elif(abs(i-j)==1):

                index = i%4
                vech = [3, 2, 1, 2]
                signh = [1, -1, 1, -1]
                h[i, j] = t1* C1(m, vech[index], signh[index])
                h[j, i] = t1* C1(m, vech[index], -signh[index])

                vecu = [1, 0, 0, 0]
                U[i, j] = t1* C1(m, vecu[index], 1)

                vecu = [0, 0, 3, 0]
                U[j, i] = t1* C1(m, vecu[index], -1)

            elif(abs(i-j)==2):
                index = i%4

                vech = [3, 2, 2, 3]
                signh = [1, -1, -1, 1]
                phaseh = [-1, -1, 1, 1]
                h[i, j] = t2*exp(phaseh[index]*1j*phi)* C2(m, vech[index], signh[index])
                h[j, i] = t2*exp(-phaseh[index]*1j*phi)* C2(m, vech[index], -signh[index])

                vecu = [2, 0, 0, 2]
                phaseu = [1, 0, 0, -1]
                U[i, j] = t2*exp(phaseu[index]*1j*phi)* C2(m, vecu[index], -1)

                vecu = [0, 3, 3, 0]
                phaseu = [0, -1, 1, 0]
                U[j, i] = t2*exp(phaseu[index]*1j*phi)* C2(m, vecu[index], -1)

    return h, U

# Hamiltonian in frequency space representation
def Hf():
    tsize = trunc*ns
    hf = np.zeros((tsize, tsize), dtype=complex)
    uf = np.zeros((tsize, tsize), dtype=complex)
    
    for i in range(trunc):
        for j in range(i+1):
            for m in range(maxorder+1):
                if(i-j==m):
                    h, u = Hn(m)
                    hf[ns*i:ns*(i+1), ns*j:ns*(j+1)] = h
                    uf[ns*i:ns*(i+1), ns*j:ns*(j+1)] = u
                    h, u = Hn(-m)
                    hf[ns*j:ns*(j+1), ns*i:ns*(i+1)] = h
                    uf[ns*j:ns*(j+1), ns*i:ns*(i+1)] = u

    return hf, uf

def Momega(n):
    tsize = trunc*n
    momega = np.zeros((tsize, tsize), dtype=complex)
    aa = np.arange(-(trunc//2), trunc//2+1)*omega
    for i in range(trunc):
        for j in range(i+1):
            if(i==j):
                momega[n*i:n*(i+1), n*j:n*(j+1)] = np.identity(n)*aa[i]

    return momega

def Hfil():
    if(filt=="square"):
        hf, Uf, Ul, Ur = Hsquare()
    elif(filt=="impurity"):
        hf, Uf, Ul, Ur = Himpurity()

    return hf, Uf, Ul, Ur

# square lattice filter
def Hsquare():
    hfil = np.zeros((ns,ns), dtype=complex)
    Ufil = np.zeros((ns,ns), dtype=complex)
    Usfl = np.zeros((ns,ns), dtype=complex)
    Usfr = np.zeros((ns,ns), dtype=complex)
    for j in range(ns):
        for i in range(j+1):
            if(i==j):
                hfil[i, j] = 0
                Ufil[i, j] = tf
                Usfl[i, j] = tcl
                Usfr[i, j] = tcr

            elif(abs(i-j)==1):
                hfil[i, j] = tf
                hfil[j, i] = tf

    hf = np.zeros((trunc*ns,trunc*ns), dtype=complex)
    Uf = np.zeros((trunc*ns,trunc*ns), dtype=complex)
    Ul = np.zeros((trunc*ns,trunc*ns), dtype=complex)
    Ur = np.zeros((trunc*ns,trunc*ns), dtype=complex)
    for i in range(trunc):
        hf[ns*i:ns*(i+1), ns*i:ns*(i+1)] = hfil
        Uf[ns*i:ns*(i+1), ns*i:ns*(i+1)] = Ufil
        Ul[ns*i:ns*(i+1), ns*i:ns*(i+1)] = Usfl
        Ur[ns*i:ns*(i+1), ns*i:ns*(i+1)] = Usfr

    return hf, Uf, Ul, Ur

# impurtity band filter
def Himpurity():
    hfil = np.zeros((nf,nf), dtype=complex)
    Ufil = np.zeros((nf,nf), dtype=complex)
    Usfl = np.zeros((nf,ns), dtype=complex)
    Usfr = np.zeros((ns,nf), dtype=complex)

    for j in range(ns):
        for i in range(j+1):

            if(i==j):
                s = -np.sign(i%2-0.5)

                hfil[i, j] = s*Mf
                Usfl[i, j] = tcl
                Usfr[i, j] = tcr
            elif(abs(i-j)==1):
                index = i%4
                hfil[i, j] = tf
                hfil[j, i] = tf

                vec = [1, 0, 0, 0]
                Ufil[i, j] = tf*vec[index]

                vec = [0, 0, 1, 0]
                Ufil[j, i] = tf*vec[index]

    for i in range(width+1):
        hfil[i+ns, i+ns] = Mi
        hfil[i+ns, 2*i] = ti
        hfil[2*i, i+ns] = ti
        # Usfl[i+ns, i] = tcl
        # Usfr[i, i+ns] = tcr

    hf = np.zeros((trunc*nf,trunc*nf), dtype=complex)
    Uf = np.zeros((trunc*nf,trunc*nf), dtype=complex)
    Ul = np.zeros((trunc*nf,trunc*ns), dtype=complex)
    Ur = np.zeros((trunc*ns,trunc*nf), dtype=complex)
    for i in range(trunc):

        hf[nf*i:nf*(i+1), nf*i:nf*(i+1)] = hfil
        Uf[nf*i:nf*(i+1), nf*i:nf*(i+1)] = Ufil
        Ul[nf*i:nf*(i+1), ns*i:ns*(i+1)] = Usfl
        Ur[ns*i:ns*(i+1), nf*i:nf*(i+1)] = Usfr

    return hf, Uf, Ul, Ur

# def SelfE(lead="L"):
#     tsize = nf*trunc

#     if(filt=="square"):
#         if(lead=="L"):
#             return 1j*gammaL*identity(tsize, dtype=complex)
#         elif(lead=="R"):
#             return 1j*gammaR*identity(tsize, dtype=complex)
#     elif(filt=="impurity"):
#         sig = np.zeros((tsize, tsize), dtype=complex)
#         for i in range(trunc):
#             sig[i*ns:(i+1)*ns, i*ns:(i+1)*ns] = 1j*gammaL*identity(ns, dtype=complex)
#         return sig
        
def SelfE(lead="L"):
    tsize = nf*trunc

    G = np.zeros((nf, nf), dtype = complex)
    G[0:d, 0:d] = identity(d, dtype=complex)*gammaL*1j
    
    Gamma = np.zeros((tsize, tsize), dtype=complex)
    for i in range(trunc):
        Gamma[nf*i:nf*(i+1), nf*i:nf*(i+1)] = G
    return Gamma

def GL(energy, backgate):
    tsize = ns*trunc
    hs, Us = Hf()
    hf, Uf, Usfl, Usfr = Hfil()

    Usd = conj(Us.T)
    Ufd = conj(Uf.T)
    Usfld = conj(Usfl.T)
    Usfrd = conj(Usfr.T)

    w = energy*identity(tsize, dtype=complex)
    wf = energy*identity(nf*trunc, dtype=complex)
    vb = backgate*identity(tsize, dtype=complex)

    GL = []
    Gn1 = inv(wf+Momega(nf)-(hf+vf)-SelfE("L"))
    GL.append(Gn1)
    for i in range(lengthf-1):
        Gn2 = inv(wf+Momega(nf)-(hf+vf)-Ufd@Gn1@Uf)
        GL.append(Gn2)
        Gn1 = Gn2

    Gn2 = inv(w+Momega(ns)-(hs+vb)-Usfld@Gn1@Usfl)
    GL.append(Gn2)
    Gn1 = Gn2

    for i in range(length-1):
        Gn2 = inv(w+Momega(ns)-(hs+vb)-Usd@Gn1@Us)
        GL.append(Gn2)
        Gn1 = Gn2

    Gn2 = inv(wf+Momega(nf)-(hf+vf)-Usfrd@Gn1@Usfr)
    GL.append(Gn2)
    Gn1 = Gn2

    for i in range(lengthf-1):
        Gn2 = inv(wf+Momega(nf)-(hf+vf)-Ufd@Gn1@Uf)
        GL.append(Gn2)
        Gn1 = Gn2

    return GL

def G1NL(energy, backgate):

    tsize = ns*trunc
    hs, Us = Hf()
    hf, Uf, Usfl, Usfr = Hfil()

    Usd = conj(Us.T)
    Ufd = conj(Uf.T)
    Usfld = conj(Usfl.T)
    Usfrd = conj(Usfr.T)

    w = energy*identity(tsize, dtype=complex)
    wf = energy*identity(nf*trunc, dtype=complex)
    vb = backgate*identity(tsize, dtype=complex)

    gl = GL(energy, backgate)
    N = len(gl)

    GNN = inv(wf+Momega(nf)-(hf+vf)-Ufd@gl[N-2]@Uf-SelfE("R"))
    G1N = np.identity(nf*trunc)
    for i in range(lengthf-1):
        G1N = G1N@gl[i]@Uf

    G1N = G1N@gl[lengthf-1]@Usfl

    for i in range(length-1):
        G1N = G1N@gl[i+lengthf]@Us

    G1N = G1N@gl[lengthf+length-1]@Usfr

    for i in range(lengthf-1):
        G1N = G1N@gl[i+lengthf+length]@Uf

    G1N = G1N@GNN

    return G1N

def GR(energy, backgate):
    tsize = ns*trunc
    hs, Us = Hf()
    hf, Uf, Usfl, Usfr = Hfil()
    Usd = conj(Us.T)
    Ufd = conj(Uf.T)
    Usfld = conj(Usfl.T)
    Usfrd = conj(Usfr.T)

    w = energy*identity(tsize, dtype=complex)
    wf = energy*identity(nf*trunc, dtype=complex)
    vb = backgate*identity(tsize, dtype=complex)

    GL = []
    Gn1 = inv(wf+Momega(nf)-(hf+vf)-SelfE("R"))
    GL.append(Gn1)
    for i in range(lengthf-1):
        Gn2 = inv(wf+Momega(nf)-(hf+vf)-Uf@Gn1@Ufd)
        GL.append(Gn2)
        Gn1 = Gn2

    Gn2 = inv(w+Momega(ns)-(hs+vb)-Usfr@Gn1@Usfrd - 1j*eta)
    GL.append(Gn2)
    Gn1 = Gn2

    for i in range(length-1):
        Gn2 = inv(w+Momega(ns)-(hs+vb)-Us@Gn1@Usd)
        GL.append(Gn2)
        Gn1 = Gn2

    Gn2 = inv(wf+Momega(nf)-(hf+vf)-Usfl@Gn1@Usfld)
    GL.append(Gn2)
    Gn1 = Gn2

    for i in range(lengthf-1):
        Gn2 = inv(wf+Momega(nf)-(hf+vf)-Uf@Gn1@Ufd)
        GL.append(Gn2)
        Gn1 = Gn2

    return GL

def G1NR(energy, backgate):

    tsize = ns*trunc
    hs, Us = Hf()
    hf, Uf, Usfl, Usfr = Hfil()
    Usd = conj(Us.T)
    Ufd = conj(Uf.T)
    Usfld = conj(Usfl.T)
    Usfrd = conj(Usfr.T)

    gr = GR(energy, backgate)
    N = len(gr)
    w = energy*identity(tsize, dtype=complex)
    wf = energy*identity(nf*trunc, dtype=complex)
    vb = backgate*identity(tsize, dtype=complex)


    GNN = inv(wf+Momega(nf)-(hf+vf)-Uf@gr[N-2]@Ufd-SelfE("L"))
    G1N = np.identity(nf*trunc)
    for i in range(lengthf-1):
        G1N = G1N@gr[i]@Ufd

    G1N = G1N@gr[lengthf-1]@Usfrd

    for i in range(length-1):
        G1N = G1N@gr[i+lengthf]@Usd

    G1N = G1N@gr[lengthf+length-1]@Usfld

    for i in range(lengthf-1):
        G1N = G1N@gr[i+lengthf+length]@Ufd

    G1N = G1N@GNN

    return G1N

def GetaL(energy, backgate):
    tsize = ns*trunc
    hs, Us = Hf()
    hf, Uf, Usfl, Usfr = Hfil()

    Usd = conj(Us.T)
    Ufd = conj(Uf.T)
    Usfld = conj(Usfl.T)
    Usfrd = conj(Usfr.T)

    w = energy*identity(tsize, dtype=complex)
    vb = backgate*identity(tsize, dtype=complex)

    gl = GL(energy, backgate)
    gr = GR(energy, backgate)

    Gtt = inv(w+Momega()-(hs+vb)-Usd@gl[lengthf+length-2]@Us-Usfr@gr[length-1]@Usfrd-1j*eta)
    Gst = identity(tsize)
    for i in range(length-1):
        Gst = Gst@gl[lengthf+i]@Us
    Gst = Gst@Gtt
    return Gst

def GetaR(energy, backgate):
    tsize = ns*trunc
    hs, Us = Hf()
    hf, Uf, Usfl, Usfr = Hfil()

    Usd = conj(Us.T)
    Ufd = conj(Uf.T)
    Usfld = conj(Usfl.T)
    Usfrd = conj(Usfr.T)

    w = energy*identity(tsize, dtype=complex)
    vb = backgate*identity(tsize, dtype=complex)

    gl = GL(energy, backgate)
    gr = GR(energy, backgate)

    Gtt = inv(w+Momega()-(hs+vb)-Us@gr[lengthf+length-2]@Usd-Usfl@gl[length-1]@Usfld-1j*eta)
    Gst = identity(tsize)
    for i in range(length-1):
        Gst = Gst@gr[lengthf+i]@Usd
    Gst = Gst@Gtt
    return Gst

def GNN(energy, backgate):
    gl = GL(energy, backgate)
    gr = GR(energy, backgate)
    tsize = ns*trunc
    hs, U = Hf()
    Ud = conj(U.T)
    hf, Uf, Usfl, Usfr = Hfil()
    Ufd = conj(Uf.T)
    Usfld = conj(Usfl.T)
    Usfrd = conj(Usfr.T)
    w = np.identity(trunc*ns)*energy
    N = len(gr)
    vb = backgate*identity(tsize, dtype=complex)

    gnn = []

    # Left filter
    g1 = inv(w+Momega(nf)-(hf+vf)-SelfE("L")-Uf@gr[-2]@Ufd)
    gnn.append(g1)
    for i in range(lengthf-2):
        g1 = inv(w+Momega(nf)-(hf+vf)-Ufd@gl[i]@Uf-Uf@gr[-(3+i)]@Ufd)
        gnn.append(g1)
    g1 = inv(w+Momega(nf)-(hf+vf)-Ufd@gl[lengthf-2]@Uf-Usfl@gr[-(lengthf+1)]@Usfld)
    gnn.append(g1)

    # System
    g1 = inv(w+Momega(ns)-(hs+vb)-Usfld@gl[lengthf-1]@Usfl-U@gr[-(lengthf+2)]@Ud)
    gnn.append(g1)
    for i in range(length-2):
        g1 = inv(w+Momega(ns)-(hs+vb)-Ud@gl[i+lengthf]@U-U@gr[-(i+lengthf+3)]@Ud)
        gnn.append(g1)
    g1 = inv(w+Momega(ns)-(hs+vb)-Ud@gl[-(lengthf+2)]@U-Usfr@gr[lengthf-1]@Usfrd)
    gnn.append(g1)

    # Right filter
    g1 = inv(w+Momega(nf)-(hf+vf)-Usfrd@gl[-(lengthf+1)]@Usfr-Uf@gr[lengthf-2]@Ufd)
    gnn.append(g1)
    for i in range(lengthf-2):
        g1 = inv(w+Momega(nf)-(hf+vf)-Ufd@gl[-(lengthf-i)]@Uf-Uf@gr[lengthf-3-i]@Ufd)
        gnn.append(g1)
    g1 = inv(w+Momega(nf)-(hf+vf)-Ufd@gl[-2]@Uf-SelfE("R"))
    gnn.append(g1)

    return gnn

def distribution(energy, backgate, density=False):
    tsize = ns*trunc
    gl = GL(energy, backgate)
    gr = GR(energy, backgate)
    hs, U = Hf()
    Ud = conj(U.T)
    hf, Uf, Usfl, Usfr = Hfil()
    Ufd = conj(Uf.T)
    Usfld = conj(Usfl.T)
    Usfrd = conj(Usfr.T)
    midindex = trunc//2
    N = len(gl)
    w = energy*identity(tsize, dtype=complex)
    wf = energy*identity(nf*trunc, dtype=complex)
    vb = backgate*identity(tsize, dtype=complex)

    gless01 = 0
    glesstotal = 0

    gr01 = 0
    grtotal = 0
    a1 = inv(wf+Momega(nf)-(hf+vf)-Uf@gr[N-2]@Ufd-SelfE("L"))
    a = a1[midindex*ns:(midindex+1)*ns, :]

    b1 = inv(wf+Momega(nf)-(hf+vf)-Ufd@gl[N-2]@Uf-SelfE("R"))
    b = b1[midindex*ns:(midindex+1)*ns, :]

    r = 0

    densi = []

    for i in range(N):

        if(i<(lengthf+length-r) and i>=lengthf+r):
            for j in range(trunc):
                m = -(j-midindex)
                am = a[:, j*ns:(j+1)*ns]
                bm = b[:, j*ns:(j+1)*ns]
                # print(m)

                incre1 = gammaL*am@identity(ns)@conj(am.T)*fermi(energy-m*omega, muL) + gammaR*bm@identity(ns)@conj(bm.T)*fermi(energy-m*omega, muR)

                gless01 = gless01 + trace(incre1[0:2, 0:2])
                glesstotal = glesstotal + trace(incre1)

                incre2 = gammaL*am@identity(ns)@conj(am.T) + gammaR*bm@identity(ns)@conj(bm.T)


                gr01 = gr01 + trace(incre2[0:2, 0:2])
                grtotal = grtotal + trace(incre2)


        if(i<lengthf-1):
            U1 = Ufd
            U2 = Uf
        elif(i==lengthf-1):
            U1 = Usfld
            U2 = Usfr
        elif(i<(lengthf+length-1)):
            U1 = Ud
            U2 = U
        elif(i==lengthf+length-1):
            U1 = Usfrd
            U2 = Usfl
        else:
            U1 = Ufd
            U2 = Uf

        a2 = gr[N-i-2]@U1@a1
        a1 = a2
        a = a2[midindex*ns:(midindex+1)*ns,:]

        b2 = gl[N-i-2]@U2@b1
        b1 = b2
        b = b2[midindex*ns:(midindex+1)*ns,:]

    # Add i*eta to the first slice in system
    # a1 = inv(w+Momega()-(hs+vf)-Usfld@gl[lengthf-1]@Usfl-U@gr[-(lengthf+2)]@Ud - 1j*eta)
    # a = a1[midindex*ns:(midindex+1)*ns, :]
    #
    # b1 = inv(w+Momega()-(hs+vf)-Ud@gl[-(lengthf+2)]@U-Usfr@gr[lengthf-1]@Usfrd - 1j*eta)
    # b = b1[midindex*ns:(midindex+1)*ns, :]
    # for i in range(length):
    #     for j in range(trunc):
    #         if(i>=r and i<length-r):
    #             m = j-midindex
    #             am = a[:, j*ns:(j+1)*ns]
    #             bm = b[:, j*ns:(j+1)*ns]
    #
    #             incre = ett*am@identity(ns)@conj(am.T)*fermi(energy+m*omega, muL) + ett*bm@identity(ns)@conj(bm.T)*fermi(energy+m*omega, muR)
    #
    #             gless0 = gless0 + incre[0, 0] + incre[1, 1]
    #             gless1 = gless1 + incre[-1, -1] + incre[-2, -2]
    #             glesstotal = glesstotal + trace(incre)
    #
    #     a2 = gr[N-lengthf-i-2]@Ud@a1
    #     a1 = a2
    #     a = a2[midindex*ns:(midindex+1)*ns,:]
    #
    #     b2 = gl[N-lengthf-i-2]@U@b1
    #     b1 = b2
    #     b = b2[midindex*ns:(midindex+1)*ns,:]

    f1 = abs(gless01.real/gr01.real)
    f2 = abs(glesstotal.real/grtotal.real)


    # return abs(gr1), abs(gr2), abs(gr3)
    if(density==False):
        return f1
    else:
        return densi

def Transmission(energy, backgate):
    midindex = trunc//2

    gl = G1NL(energy, backgate)
    gr = G1NR(energy, backgate)

    # gel = GetaL(energy, backgate)
    # ger = GetaR(energy, backgate)

    TL = 0
    TR = 0
    SL = np.zeros((nf, nf), dtype=complex)
    SR = np.zeros((nf, nf), dtype=complex)

    SL[0:d, 0:d] = 1j*gammaL*np.identity(d, dtype=complex)
    SR[0:d, 0:d] = 1j*gammaR*np.identity(d, dtype=complex)
    # Eta = 1j*ett*np.identity(ns, dtype=complex)
    for i in range(trunc):
        Gml = gl[nf*i:nf*(i+1), nf*midindex:nf*(midindex+1)]
        Gmr = gr[nf*i:nf*(i+1), nf*midindex:nf*(midindex+1)]

        # Geml = gel[ns*i:ns*(i+1), ns*midindex:ns*(midindex+1)]
        # Gemr = ger[ns*i:ns*(i+1), ns*midindex:ns*(midindex+1)]

        Gmld = conj(Gml.T)
        Gmrd = conj(Gmr.T)

        # Gemld = conj(Geml.T)
        # Gemrd = conj(Geml.T)

        TL = TL + 2*np.trace(Gmld@SL@Gml@SR)
        TR = TR + 2*np.trace(Gmrd@SR@Gmr@SL)

    return abs(TL+TR)

# control function to plot distribution function
def plot_distri():
    num = 51

    print("Calculating Distribution")
    Estart = -1
    Eend = 1
    V = np.linspace(Estart, Eend, num)
    distri1 = np.zeros(num)
    distri2 = np.zeros(num)

    for i in range(num):
        dist = distribution(energy=V[i], backgate=backgateV, density=False)
        distri1[i] = dist

        print(int(i/num*100),"%")
    print(100, "%")

    print(distri1)
    directory = "/Users/ruoyuzhang/Desktop/Research/energy_filtered_lead/data/main_results/floq_filt_dist_s_1"
    np.save(directory, np.vstack((V-backgateV,distri1)))
    save_param("dist", Estart, Eend, num, directory+".txt")

    plt.figure()
    plt.plot(V, distri1, '.-')

    # title = "muL="+str(muL)+ ", W="+str(width)+", L=" +str(length)+ ", Lf="+ str(lengthf) + ", eta="+ str(ett)+ ", tf=" + str(tf)
    # plt.title(title)

    plt.ylabel("occupation")
    plt.xlabel("energy")
    plt.show()


# control function to plot dI/dV (differential conductance)
def plot_dIdV():
    Estart = -0.2
    Eend = 0.2
    num = 11

    V = np.linspace(Estart, Eend, num)
    # print(V+mu)

    print("Calculating dI/dV")
    dIdV = np.zeros(num)
    dIdV2 = np.zeros(num)
    dIdV3 = np.zeros(num)


    for i in range(num):
        T = Transmission(0, -V[i])
        dIdV[i] = T

        print(int(i/num*100),"%")

    print(100, "%")


    directory = "/Users/ruoyuzhang/Desktop/Research/energy_filtered_lead/data/main_results/floq_dIdV_filt_2reso_o1"
    np.save(directory, np.vstack((V,dIdV)))
    save_param("cond", Estart, Eend, num, directory+".txt")


    plt.figure()
    plt.plot(V, dIdV, '.-')
    plt.grid()
    plt.ylabel("dI/dV")
    plt.xlabel("V (backgate voltage)")
    # plt.xlim([-0.2, 0.2])
    # plt.ylim([0.5, 1.5])
    # title = "Fl filter:A0="+str(A0) +",t2="+ str(t2)+",M="+str(M)+ ",phi="+ str(round(phi, 2))+ ", W="+ str(width)+ ",Ls="+str(length)+",Lf="+str(lengthf)
    # plt.title(title)

    plt.show()

def plot_density():

    midindex=trunc//2
    d_list = GNN(energy=0, backgate=backgateV)
    density = np.zeros((ns, length+2*lengthf))
    print(len(d_list))
    for i in range(length+2*lengthf):
        density[:, i] = -np.diag(d_list[i][midindex*ns:(midindex+1)*ns, midindex*ns:(midindex+1)*ns]).imag

    plt.figure()
    x = np.arange(length+2*lengthf)
    plt.plot(x, density[0, :])
    plt.plot(x, density[width, :])
    plt.plot(x, density[-1, :])
    plt.legend(["top edge", "middle", "bottom edge"])

    plt.figure()
    plt.imshow(density, cmap='coolwarm')
    plt.colorbar()

    plt.show()

def save_param(f, Estart, Eend, num, filename):
    if(f=="cond"):
        name = ["a", "t1", "t2", "M", "phi", "A0", "omega", "maxorder", "gammaL", "gammaR", "length", "width", "trunc", "filt", "lengthf", "tf", "tcl", "tcr", "Vbf", "Estart", "Eend", "num"]
        param = [a, t1, t2, M, phi, A0, omega, maxorder, gammaL, gammaR, length, width, trunc, filt, lengthf, tf, tcl, tcr, vbf, Estart, Eend, num]
    elif(f=="dist"):
        name = ["a", "t1", "t2", "M", "phi", "A0", "omega", "maxorder", "gammaL", "gammaR", "length", "width", "trunc", "filt", "lengthf", "tf", "tcl", "tcr","muL", "muR", "backgate", "Estart", "Eend", "num"]
        param = [a, t1, t2, M, phi, A0, omega, maxorder, gammaL, gammaR, length, width, trunc, filt, lengthf, tf, tcl, tcr, muL, muR, backgateV, Estart, Eend, num]

    l = []
    for i in range(len(name)):
        entry = name[i] + " = " + str(param[i]) + "\n"
        l.append(entry)
    file = open(filename, "w")
    file.writelines(l)
    file.close()


if __name__ == "__main__":


    plot_dIdV()
    # plot_distri()
    # plot_density()
    # Sig = Transmission(0, -1.553)
    # print(Sig, width, length, lengthf, tf, vbf)
