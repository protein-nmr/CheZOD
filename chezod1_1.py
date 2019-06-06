import sys
import string
from scipy.special import erfc
from scipy.optimize import curve_fit
import numpy as np
##from matplotlib import pyplot as pl

##from pylab import *
from numpy import *
##from jakob_util import *
import os
import time
import numpy
import math
##from random import normalvariate, randint, random
##from random import choice as randchoice
##from numpy.linalg import lstsq
##from numpy.random import rand

def initfil2(filename):
  file=open(filename,'r')
  buffer=file.readlines()
  file.close()
  for i in range(len(buffer)):
     buffer[i]=string.split(buffer[i])
  return buffer

aa13dict={'A': 'ALA', 'C': 'CYS', 'E': 'GLU', 'D': 'ASP', 'G': 'GLY', 'F': 'PHE', 'I': 'ILE', 'H': 'HIS', 'K': 'LYS', 'M': 'MET', 'L': 'LEU', 'N': 'ASN', 'Q': 'GLN', 'P': 'PRO', 'S': 'SER', 'R': 'ARG', 'T': 'THR', 'W': 'TRP', 'V': 'VAL', 'Y': 'TYR'}
VERB=False #change this to True for verbose logfile

#constants
R = 8.314472
e = 79.0
a = 5.0
b = 7.5
cutoff = 2
ncycles = 5

def smallmatrixlimits(ires, cutoff, len):
  ileft = max(1, ires - cutoff)
  iright = min(ileft + 2 * cutoff, len)
  if iright == len:
    ileft = max(1, iright - 2 * cutoff)
  return (ileft, iright)

def smallmatrixpos(ires, cutoff, len):
  resi = cutoff + 1
  if ires < cutoff + 1:
    resi = ires
  if ires > len - cutoff:
    resi = min(len, 2 * cutoff + 1) - (len - ires)
  return resi

def fun(pH, pK, nH):
  return (10 ** ( nH*(pK - pH) ) ) / (1. + (10 **( nH*(pK - pH) ) ) )

def W(r,Ion=0.1):
  k = np.sqrt(Ion) / 3.08 #Ion=0.1 is default
  x = k * r / np.sqrt(6)
  return 332.286 * np.sqrt(6 / np.pi) * (1 - np.sqrt(np.pi) * x * np.exp(x ** 2) * erfc(x)) / (e * r)

def w2logp(x,T=293.15):
  return x * 4181.2 / (R * T * np.log(10)) 

pK0 = {"n":8.23, "D":3.86, "E":4.34, "H":6.45, "C":8.49, "K":10.34, "R":13.9, "Y":9.76, "c":3.55}

def calc_pkas_from_seq(seq=None, T=293.15, Ion=0.1):
  #pH range
  pHs = np.arange(1.99, 10.01, 0.15)
  #titratable groups
  ##pK0 = {"n":7.5, "D":4.0, "E":4.4, "H":6.6, "C":8.6, "K":10.4, "R":12.0, "Y":9.6, "c":3.5} #was these values!
  pK0 = {"n":8.23, "D":3.86, "E":4.34, "H":6.45, "C":8.49, "K":10.34, "R":13.9, "Y":9.76, "c":3.55}

  pos = np.array([i for i in range(len(seq)) if seq[i] in pK0.keys()])
  N = pos.shape[0]
  I = np.diag(np.ones(N))
  sites = ''.join([seq[i] for i in pos])
  neg = np.array([i for i in range(len(sites)) if sites[i] in 'DEYc'])
  l = np.array([abs(pos - pos[i]) for i in range(N)])
  d = a + np.sqrt(l) * b

  tmp = W(d,Ion)
  tmp[I == 1] = 0

  ww = w2logp(tmp,T) / 2

  chargesempty = np.zeros(pos.shape[0])
  if len(neg): chargesempty[neg] = -1

  pK0s = [pK0[c] for c in sites]
  nH0s = [0.9 for c in sites]

  titration = np.zeros((N,len(pHs)))
  smallN = min(2 * cutoff+1, len(pos))
  smallI = np.diag(np.ones(smallN))

  alltuples =  [[int(c) for c in np.binary_repr(i, smallN)]
                for i in range(2 ** (smallN))]
  gmatrix = [np.zeros((smallN, smallN)) for p in range(len(pHs))]

  #perform iterative fitting.........................
  for icycle in range(ncycles):
    ##print (icycle)

    if icycle == 0:
      fractionhold = np.array([[fun(pHs[p], pK0s[i], nH0s[i]) for i in range(N)] for p in range(len(pHs))])
    else:
      fractionhold = titration.transpose()

    for ires in range(1, N+1):

      (ileft,iright) = smallmatrixlimits(ires, cutoff, N)
    
      resi = smallmatrixpos(ires, cutoff, N)
#    print ires, resi, (ileft, iright) 

      for p in range(len(pHs)):
  
        fraction = fractionhold[p].copy()
        fraction[ileft - 1 : iright] = 0
        charges = chargesempty + fraction  
        ww0 = np.diag(np.dot(ww, charges) * 2) 
        gmatrixfull =  ww + ww0 + pHs[p] * I - np.diag(pK0s) 
        gmatrix[p] = gmatrixfull[ileft - 1 : iright, ileft - 1 : iright]

      E_all = np.array([sum([10 ** -(gmatrix[p] * np.outer(c,c)).sum() for c in alltuples]) for p in range(len(pHs))])
      E_sel = np.array([sum([10 ** -(gmatrix[p] * np.outer(c,c)).sum() for c in alltuples if c[resi-1] == 1]) for p in range(len(pHs))])
      titration[ires-1] = E_sel/E_all
    sol=np.array([curve_fit(fun, pHs, titration[p], [pK0s[p], nH0s[p]],ftol=0.0001,maxfev=2000)[0] for p in range(len(pK0s))])
    (pKs, nHs) = sol.transpose()
    ##print (sol)

  dct={}
  for p,i in enumerate(pos):
    ##print (p,i,seq[i],pKs[p],nHs[p])
    dct[i-1]=(pKs[p],nHs[p],seq[i])

  return dct


##--------------- POTENCI core code and data tables from here -----------------

#AAstandard='ACDEFGHIKLMNPQRSTVY'
AAstandard='ACDEFGHIKLMNPQRSTVWY'

tablecent='''aa C CA CB N H HA HB
A 177.44069  52.53002  19.21113 125.40155   8.20964   4.25629   1.31544
C 174.33917  58.48976  28.06269 120.71212   8.29429   4.44261   2.85425
D 176.02114  54.23920  41.18408 121.75726   8.28460   4.54836   2.60054
E 176.19215  56.50755  30.30204 122.31578   8.35949   4.22124   1.92383
F 175.42280  57.64849  39.55984 121.30500   8.10906   4.57507   3.00036
G 173.83294  45.23929  None     110.09074   8.32746   3.91016   None
H 175.00142  56.20256  30.60335 120.69141   8.27133   4.55872   3.03080
I 175.88231  61.04925  38.68742 122.37586   8.06407   4.10574   1.78617
K 176.22644  56.29413  33.02478 122.71282   8.24902   4.25873   1.71982
L 177.06101  55.17464  42.29215 123.48611   8.14330   4.28545   1.54067
M 175.90708  55.50643  32.83806 121.54592   8.24848   4.41483   1.97585
N 174.94152  53.22822  38.87465 119.92746   8.37189   4.64308   2.72756
P 176.67709  63.05232  32.03750 137.40612   None      4.36183   2.03318
Q 175.63494  55.79861  29.44174 121.49225   8.30042   4.28006   1.97653
R 175.92194  56.06785  30.81298 122.40365   8.26453   4.28372   1.73437
S 174.31005  58.36048  63.82367 117.11419   8.25730   4.40101   3.80956
T 174.27772  61.86928  69.80612 115.48126   8.11378   4.28923   4.15465
V 175.80621  62.20156  32.77934 121.71912   8.06572   4.05841   1.99302
W 175.92744  57.23836  29.56502 122.10991   7.97816   4.61061   3.18540
Y 175.49651  57.82427  38.76184 121.43652   8.05749   4.51123   2.91782'''

def initcorcents():
    datc=string.split(tablecent,'\n')
    ##aas=string.split(datc[0],'\t')[1:]
    aas=string.split(datc[0])[1:]
    dct={}
    for i in range(20):
      ##vals=string.split(datc[1+i],'\t')
      vals=string.split(datc[1+i])
      aai=vals[0]
      dct[aai]={}
      for j in range(7):
	atnj=aas[j]
	dct[aai][atnj]=eval(vals[1+j])
    return dct
	

tablenei='''C A  0.06131 -0.04544  0.14646  0.01305
 C C  0.04502  0.12592 -0.03407 -0.02654
 C D  0.08180 -0.08589  0.22948  0.10934
 C E  0.05388  0.22264  0.06962  0.01929
 C F -0.06286 -0.22396 -0.34442  0.00950
 C G  0.12772  0.72041  0.16048  0.01324
 C H -0.00628 -0.03355  0.13309 -0.03906
 C I -0.11709  0.06591 -0.06361 -0.03628
 C K  0.03368  0.15830  0.04518 -0.01576
 C L -0.03877  0.11608  0.02535  0.01976
 C M  0.04611  0.25233 -0.00747 -0.01624
 C N  0.07068 -0.06118  0.10077  0.05547
 C P -0.36018 -1.90872  0.16158 -0.05286
 C Q  0.10861  0.19878  0.01596 -0.01757
 C R  0.01933  0.13237  0.03606 -0.02468
 C S  0.09888  0.28691  0.07601  0.01379
 C T  0.05658  0.41659 -0.01103 -0.00114
 C V -0.11591  0.09565 -0.03355 -0.03368
 C W -0.01954 -0.19134 -0.37965  0.01582
 C Y -0.08380 -0.24519 -0.32700 -0.00577
CA A  0.03588  0.03480 -0.00468 -0.00920
CA C  0.02749  0.15742  0.14376  0.03681
CA D -0.00751  0.12494  0.17354  0.14157
CA E  0.00985  0.13936  0.03289 -0.00702
CA F  0.01122  0.03732 -0.19586 -0.00377
CA G -0.00885  0.23403 -0.03184 -0.01144
CA H -0.02102  0.04621  0.03122 -0.02826
CA I -0.00656  0.05965 -0.10588 -0.04372
CA K  0.01817  0.11216 -0.00341 -0.02950
CA L  0.04507  0.07829 -0.03526  0.00858
CA M  0.07553  0.18840  0.04987 -0.01749
CA N -0.00649  0.11842  0.18729  0.06401
CA P -0.27536 -2.02189  0.01327 -0.08732
CA Q  0.06365  0.15281  0.04575 -0.01356
CA R  0.04338  0.11783  0.00345 -0.02873
CA S  0.02867  0.07846  0.09443  0.02061
CA T -0.01625  0.10626  0.03880 -0.00126
CA V -0.04935  0.04248 -0.10195 -0.03778
CA W  0.00434  0.16188 -0.08742  0.03983
CA Y  0.02782  0.02846 -0.24750  0.00759
CB A -0.00953  0.05704 -0.04838  0.00755
CB C -0.00164  0.00760 -0.03293 -0.05613
CB D  0.02064  0.09849 -0.08746 -0.06691
CB E  0.01283  0.05404 -0.01342  0.02238
CB F  0.01028  0.03363  0.18112  0.01493
CB G -0.02758  0.04383  0.06071 -0.02639
CB H -0.01760 -0.02367  0.00343  0.00415
CB I  0.02783  0.01052  0.00641  0.05090
CB K  0.00350  0.02852 -0.00408  0.01218
CB L  0.01223 -0.02940 -0.07268  0.00884
CB M -0.02925 -0.03912 -0.06587  0.03490
CB N -0.02242  0.03403 -0.09759 -0.08018
CB P  0.08431 -0.35696 -0.04680  0.05192
CB Q -0.01649 -0.01016 -0.03663  0.01723
CB R -0.01887  0.00618 -0.00385  0.02884
CB S -0.00921  0.07096 -0.06338 -0.03707
CB T  0.02601  0.04904 -0.01728  0.00781
CB V  0.03068  0.06325  0.01928  0.05011
CB W -0.07651 -0.11334  0.13806 -0.03339
CB Y  0.00082  0.01466  0.18107 -0.01181
 N A  0.09963 -0.00873 -2.31666 -0.14051
 N C  0.11905 -0.01296  1.15573  0.01820
 N D  0.11783 -0.11817 -1.16322 -0.37601
 N E  0.10825 -0.00605 -0.41856  0.01187
 N F -0.12280 -0.27542  0.34635  0.09102
 N G  0.10365 -0.05667 -1.50346 -0.00146
 N H -0.04145 -0.26494  0.26356  0.18198
 N I -0.09249  0.12136  2.75071  0.40643
 N K -0.02472  0.07224 -0.07057  0.12261
 N L  0.01542 -0.12800 -0.85172 -0.15460
 N M -0.11266 -0.27311 -0.33192  0.09384
 N N -0.00295 -0.20562 -1.00652 -0.30971
 N P  0.03252  1.35296 -1.17173  0.06026
 N Q  0.00900 -0.09950 -0.07389  0.08415
 N R -0.07819  0.00802 -0.04821  0.08524
 N S  0.12057  0.02242  0.48924 -0.25423
 N T  0.04631  0.09935  1.02269  0.20228
 N V -0.03610  0.21959  2.42228  0.39686
 N W -0.15643 -0.19285  0.05515 -0.53172
 N Y -0.10497 -0.25228  0.46023  0.01399
 H A  0.01337 -0.00605 -0.04371 -0.02485
 H C  0.01324  0.05107  0.12857  0.00610
 H D  0.02859  0.02436 -0.06510  0.02085
 H E  0.02737  0.01790  0.03740  0.01969
 H F -0.02633 -0.08287 -0.11364 -0.03603
 H G  0.02753  0.05640 -0.10477  0.06876
 H H -0.00124 -0.02861  0.04126  0.10004
 H I -0.02258 -0.00929  0.07962  0.01880
 H K -0.00512 -0.00744  0.04443  0.03434
 H L -0.01088 -0.01230 -0.03640 -0.03719
 H M -0.01961 -0.00749 -0.00097  0.02041
 H N  0.01134  0.02121 -0.01837 -0.00629
 H P -0.01246  0.02956  0.13007 -0.00810
 H Q  0.00783  0.00751  0.05643  0.02413
 H R -0.00734  0.00546  0.07003  0.04051
 H S  0.02133  0.03964  0.04978 -0.03749
 H T  0.00976  0.06072  0.03531  0.01657
 H V -0.01267  0.00994  0.09630  0.03420
 H W -0.02348 -0.09617 -0.24207 -0.18741
 H Y -0.01881 -0.07345 -0.14345 -0.06721
HA A  0.00350 -0.02371 -0.00654  0.00652
HA C  0.00660  0.01073  0.01921  0.00919
HA D  0.01717 -0.00854 -0.00802 -0.00597
HA E  0.01090 -0.01091  0.00472  0.00790
HA F -0.02271 -0.06316 -0.03057 -0.02350
HA G  0.02155 -0.00151  0.02477  0.01526
HA H -0.01132 -0.05617 -0.01514  0.01264
HA I  0.00459  0.00571  0.02984  0.00416
HA K  0.00492 -0.01788  0.00555  0.01259
HA L -0.00599 -0.01558  0.00358  0.00167
HA M  0.00100 -0.02037  0.00678  0.00930
HA N  0.00651 -0.01499 -0.00361  0.00203
HA P  0.01542  0.28350 -0.01496  0.00796
HA Q  0.00711 -0.02142  0.00734  0.00971
HA R -0.00472 -0.01414  0.00966  0.01180
HA S  0.01572  0.02791  0.03762  0.00133
HA T  0.01714  0.06590  0.03085  0.00143
HA V  0.00777  0.01505  0.02525  0.00659
HA W -0.06818 -0.08412 -0.09386 -0.06072
HA Y -0.02701 -0.05585 -0.03243 -0.02987
HB A  0.01473  0.01843  0.01428  0.00451
HB C  0.01180  0.03340  0.03081  0.00169
HB D  0.01786  0.01626  0.02221  0.01030
HB E  0.01796  0.01820  0.00835 -0.00045
HB F -0.04867 -0.09154 -0.04858 -0.00164
HB G  0.01718  0.03852  0.01043  0.00051
HB H -0.00817 -0.04557 -0.00820  0.00855
HB I  0.00446  0.00111  0.00049 -0.00283
HB K  0.01570  0.01156  0.00771  0.00646
HB L  0.00700  0.01236  0.00880  0.00150
HB M  0.01607  0.02294  0.01385 -0.00038
HB N  0.01893  0.01561  0.02760  0.01215
HB P -0.01199 -0.02752  0.00891 -0.00033
HB Q  0.01636  0.01861  0.01177 -0.00099
HB R  0.01324  0.01526  0.01082  0.00378
HB S  0.01859  0.03487  0.02890 -0.00477
HB T  0.01624  0.04073  0.01936 -0.00348
HB V  0.00380  0.00271 -0.00144 -0.00315
HB W -0.09045 -0.06895 -0.10934 -0.01948
HB Y -0.05069 -0.06698 -0.05666 -0.01193'''

tabletermcorrs='''C n -0.15238
C c -0.90166
CB n 0.12064
CB c 0.06854
CA n -0.04616
CA c -0.06680
N n 0.347176
N c 0.619141
H n 0.156786
H c 0.023189
HB n 0.0052692
HB c 0.0310875
HA n 0.048624
HA c 0.042019'''

def initcorneis():
    datc=string.split(tablenei,'\n')
    dct={}
    for i in range(20*7):
      vals=string.split(datc[i])
      atn=vals[0]
      aai=vals[1]
      if not aai in dct:dct[aai]={}
      dct[aai][atn]=[eval(vals[2+j]) for j in range(4)]
    datc=string.split(tabletermcorrs,'\n')
    for i in range(len(datc)):
      vals=string.split(datc[i])
      atn=vals[0]
      term=vals[1]
      if not term in dct:dct[term]={}
      if term=='n':  dct['n'][atn]=[None,None,None,eval(vals[-1])]
      elif term=='c':dct['c'][atn]=[eval(vals[-1]),None,None,None]
    return dct
	

tabletempk='''aa  CA   CB   C     N    H    HA
A  -2.2  4.7 -7.1  -5.3 -9.0  0.7
C  -0.9  1.3 -2.6  -8.2 -7.0  0.0
D   2.8  6.5 -4.8  -3.9 -6.2 -0.1
E   0.9  4.6 -4.9  -3.7 -6.5  0.3
F  -4.7  2.4 -6.9 -11.2 -7.5  0.4
G   3.3  0.0 -3.2  -6.2 -9.1  0.0
H   7.8 15.5  3.1   3.3 -7.8  0.4
I  -2.0  4.6 -8.7 -12.7 -7.8  0.4
K  -0.8  2.4 -7.1  -7.6 -7.5  0.4
L   1.7  4.9 -8.2  -2.9 -7.5  0.1
M   4.1  9.4 -8.2  -6.2 -7.1 -0.5
N   2.8  5.1 -6.1  -3.3 -7.0 -2.9
P   1.1 -0.2 -4.0   0.0  0.0  0.0
Q   2.3  3.6 -5.7  -6.5 -7.2  0.3
R  -1.4  3.5 -6.9  -5.3 -7.1  0.4
S  -1.7  4.4 -4.7  -3.8 -7.6  0.1
T   0.0  2.2 -5.2  -6.7 -7.3  0.0
V  -2.8  2.5 -8.1 -14.2 -7.6  0.5
W  -2.7  3.1 -7.9 -10.1 -7.8  0.4
Y  -5.0  2.9 -7.7 -12.0 -7.7  0.5'''

def gettempkoeff():
  datc=string.split(tabletempk,'\n')
  buf=[string.split(lin) for lin in datc]
  headers=buf[0][1:]
  dct={}
  for atn in headers:
    dct[atn]={}
  for lin in buf[1:]:
    aa=lin[0]
    for j,atn in enumerate(headers):
	dct[atn][aa]=eval(lin[1+j])
  return dct

tablecombdevs='''C -1 G r xrGxx  0.2742  1.4856
 C -1 G - x-Gxx  0.0522  0.2827
 C -1 P P xPPxx -0.0822  0.4450
 C -1 P r xrPxx  0.2640  1.4303
 C -1 r P xPrxx -0.1027  0.5566
 C -1 + P xP+xx  0.0714  0.3866
 C -1 - - x--xx -0.0501  0.2712
 C -1 p r xrpxx  0.0582  0.3151
 C  1 G r xxGrx  0.0730  0.3955
 C  1 P a xxPax -0.0981  0.5317
 C  1 P + xxP+x -0.0577  0.3128
 C  1 P p xxPpx -0.0619  0.3356
 C  1 r r xxrrx -0.1858  1.0064
 C  1 r a xxrax -0.1888  1.0226
 C  1 r + xxr+x -0.1805  0.9779
 C  1 r - xxr-x -0.1756  0.9512
 C  1 r p xxrpx -0.1208  0.6544
 C  1 + P xx+Px -0.0533  0.2886
 C  1 - P xx-Px  0.1867  1.0115
 C  1 p P xxpPx  0.2321  1.2574
 C -2 G r rxGxx -0.1457  0.7892
 C -2 r p pxrxx  0.0555  0.3008
 C  2 P P xxPxP  0.1007  0.5455
 C  2 P - xxPx-  0.0634  0.3433
 C  2 r P xxrxP -0.1447  0.7841
 C  2 a r xxaxr -0.1488  0.8061
 C  2 a - xxax- -0.0093  0.0506
 C  2 + G xx+xG -0.0394  0.2132
 C  2 + P xx+xP  0.1016  0.5502
 C  2 + a xx+xa  0.0299  0.1622
 C  2 + + xx+x+  0.0427  0.2312
 C  2 - a xx-xa  0.0611  0.3308
 C  2 p P xxpxP -0.0753  0.4078
CA -1 G P xPGxx -0.0641  0.3233
CA -1 G r xrGxx  0.2107  1.0630
CA -1 P P xPPxx -0.2042  1.0303
CA -1 P p xpPxx  0.0444  0.2240
CA -1 r G xGrxx  0.2030  1.0241
CA -1 r + x+rxx -0.0811  0.4093
CA -1 - P xP-xx  0.0744  0.3755
CA -1 - - x--xx -0.0263  0.1326
CA -1 p p xppxx -0.0094  0.0475
CA  1 G P xxGPx  1.3044  6.5813
CA  1 G - xxG-x -0.0632  0.3188
CA  1 P G xxPGx  0.2642  1.3329
CA  1 P P xxPPx  0.3025  1.5262
CA  1 P r xxPrx  0.1455  0.7343
CA  1 P - xxP-x  0.1188  0.5994
CA  1 P p xxPpx  0.1201  0.6062
CA  1 r P xxrPx -0.1958  0.9878
CA  1 r - xxr-x -0.0931  0.4696
CA  1 a P xxaPx -0.1428  0.7204
CA  1 a - xxa-x -0.0262  0.1324
CA  1 a p xxapx  0.0392  0.1977
CA  1 + P xx+Px -0.1059  0.5344
CA  1 + a xx+ax -0.0377  0.1901
CA  1 + + xx++x -0.0595  0.3001
CA  1 - P xx-Px -0.1156  0.5831
CA  1 - + xx-+x  0.0316  0.1593
CA  1 - p xx-px  0.0612  0.3090
CA  1 p r xxprx -0.0511  0.2576
CA -2 P - -xPxx -0.1028  0.5185
CA -2 r r rxrxx  0.1933  0.9752
CA -2 - G Gx-xx  0.0559  0.2818
CA -2 - p px-xx  0.0391  0.1973
CA -2 p a axpxx -0.0293  0.1479
CA -2 p + +xpxx -0.0173  0.0873
CA  2 G - xxGx-  0.0357  0.1802
CA  2 + G xx+xG -0.0315  0.1591
CA  2 - P xx-xP  0.0426  0.2150
CA  2 - r xx-xr  0.0784  0.3954
CA  2 - a xx-xa  0.1084  0.5467
CA  2 - - xx-x-  0.0836  0.4216
CA  2 p P xxpxP  0.0685  0.3456
CA  2 p - xxpx- -0.0481  0.2428
CB -1 P r xrPxx -0.2678  1.7345
CB -1 P p xpPxx  0.0355  0.2300
CB -1 r P xPrxx -0.1137  0.7367
CB -1 a p xpaxx  0.0249  0.1613
CB -1 + - x-+xx -0.0762  0.4935
CB -1 - P xP-xx -0.0889  0.5757
CB -1 - r xr-xx -0.0533  0.3451
CB -1 - - x--xx  0.0496  0.3215
CB -1 - p xp-xx -0.0148  0.0960
CB -1 p P xPpxx  0.0119  0.0768
CB -1 p r xrpxx -0.0673  0.4358
CB  1 P G xxPGx -0.0522  0.3379
CB  1 P P xxPPx -0.8458  5.4779
CB  1 P r xxPrx -0.1573  1.0187
CB  1 r r xxrrx  0.1634  1.0581
CB  1 a G xxaGx -0.0393  0.2544
CB  1 a r xxarx  0.0274  0.1777
CB  1 a - xxa-x  0.0394  0.2553
CB  1 a p xxapx  0.0149  0.0968
CB  1 + G xx+Gx -0.0784  0.5076
CB  1 + P xx+Px -0.1170  0.7580
CB  1 - P xx-Px -0.0913  0.5912
CB  1 - - xx--x  0.0284  0.1838
CB  1 p P xxpPx  0.0880  0.5697
CB  1 p p xxppx -0.0113  0.0733
CB -2 P - -xPxx  0.0389  0.2521
CB -2 P p pxPxx  0.0365  0.2362
CB -2 r + +xrxx  0.0809  0.5242
CB -2 a - -xaxx -0.0452  0.2927
CB -2 + - -x+xx -0.0651  0.4218
CB -2 - G Gx-xx -0.0883  0.5717
CB -2 p G Gxpxx  0.0378  0.2445
CB -2 p p pxpxx  0.0207  0.1341
CB  2 r G xxrxG -0.0362  0.2344
CB  2 r - xxrx- -0.0219  0.1419
CB  2 a - xxax- -0.0298  0.1929
CB  2 + p xx+xp  0.0189  0.1223
CB  2 - - xx-x- -0.0525  0.3400
 N -1 G P xPGxx  0.2411  0.5105
 N -1 G + x+Gxx -0.1773  0.3754
 N -1 G - x-Gxx  0.1905  0.4035
 N -1 P P xPPxx -0.9177  1.9434
 N -1 P p xpPxx  0.2609  0.5525
 N -1 r G xGrxx  0.2417  0.5119
 N -1 r a xarxx -0.0139  0.0295
 N -1 r + x+rxx -0.4122  0.8729
 N -1 r p xprxx  0.1440  0.3049
 N -1 a G xGaxx -0.5177  1.0963
 N -1 a r xraxx  0.0890  0.1885
 N -1 a a xaaxx  0.1393  0.2950
 N -1 a p xpaxx -0.0825  0.1747
 N -1 + G xG+xx -0.4908  1.0394
 N -1 + a xa+xx  0.1709  0.3619
 N -1 + + x++xx  0.1868  0.3955
 N -1 + - x-+xx -0.0951  0.2014
 N -1 - P xP-xx -0.3027  0.6410
 N -1 - r xr-xx -0.1670  0.3537
 N -1 - + x+-xx -0.3501  0.7414
 N -1 - - x--xx  0.1266  0.2681
 N -1 p G xGpxx -0.1707  0.3614
 N -1 p - x-pxx  0.0011  0.0023
 N  1 G G xxGGx  0.2555  0.5412
 N  1 G P xxGPx -0.9725  2.0595
 N  1 G r xxGrx  0.0165  0.0349
 N  1 G p xxGpx  0.0703  0.1489
 N  1 r a xxrax -0.0237  0.0503
 N  1 a r xxarx -0.1816  0.3845
 N  1 a - xxa-x -0.1050  0.2224
 N  1 a p xxapx -0.1196  0.2533
 N  1 - r xx-rx -0.1762  0.3731
 N  1 - a xx-ax  0.0006  0.0013
 N  1 p P xxpPx  0.2797  0.5923
 N  1 p a xxpax  0.0938  0.1986
 N  1 p + xxp+x  0.1359  0.2878
 N -2 G r rxGxx -0.5140  1.0885
 N -2 G - -xGxx -0.0639  0.1354
 N -2 P P PxPxx -0.4215  0.8927
 N -2 r P Pxrxx -0.3696  0.7828
 N -2 r p pxrxx -0.1937  0.4101
 N -2 a - -xaxx -0.0351  0.0743
 N -2 a p pxaxx -0.1031  0.2183
 N -2 - G Gx-xx -0.2152  0.4558
 N -2 - P Px-xx -0.1375  0.2912
 N -2 - p px-xx -0.1081  0.2290
 N -2 p P Pxpxx -0.1489  0.3154
 N -2 p - -xpxx  0.0952  0.2015
 N  2 G - xxGx-  0.1160  0.2457
 N  2 r p xxrxp -0.1288  0.2728
 N  2 a P xxaxP  0.1632  0.3456
 N  2 + + xx+x+ -0.0106  0.0226
 N  2 + - xx+x-  0.0389  0.0824
 N  2 - a xx-xa -0.0815  0.1726
 N  2 p G xxpxG -0.0779  0.1649
 N  2 p p xxpxp -0.0683  0.1447
 H -1 G P xPGxx -0.0317  0.4730
 H -1 G r xrGxx  0.0549  0.8186
 H -1 G + x+Gxx -0.0192  0.2867
 H -1 G - x-Gxx  0.0138  0.2055
 H -1 r P xPrxx -0.0964  1.4367
 H -1 r - x-rxx -0.0245  0.3648
 H -1 a G xGaxx -0.0290  0.4320
 H -1 a a xaaxx  0.0063  0.0944
 H -1 + G xG+xx -0.0615  0.9168
 H -1 + r xr+xx -0.0480  0.7153
 H -1 + - x-+xx -0.0203  0.3030
 H -1 - + x+-xx -0.0232  0.3455
 H -1 p G xGpxx -0.0028  0.0411
 H -1 p P xPpxx -0.0121  0.1805
 H  1 G P xxGPx -0.1418  2.1144
 H  1 G r xxGrx  0.0236  0.3520
 H  1 G a xxGax  0.0173  0.2580
 H  1 a - xxa-x  0.0091  0.1349
 H  1 + P xx+Px -0.0422  0.6290
 H  1 + p xx+px  0.0191  0.2842
 H  1 - P xx-Px -0.0474  0.7065
 H  1 - a xx-ax  0.0102  0.1515
 H -2 G G GxGxx  0.0169  0.2517
 H -2 G r rxGxx -0.3503  5.2220
 H -2 a P Pxaxx  0.0216  0.3227
 H -2 a - -xaxx -0.0276  0.4118
 H -2 + - -x+xx -0.0260  0.3874
 H -2 - G Gx-xx  0.0273  0.4073
 H -2 - a ax-xx -0.0161  0.2400
 H -2 - - -x-xx -0.0285  0.4255
 H -2 p P Pxpxx -0.0101  0.1503
 H -2 p a axpxx -0.0157  0.2343
 H -2 p + +xpxx -0.0122  0.1815
 H -2 p p pxpxx  0.0107  0.1601
 H  2 G G xxGxG -0.0190  0.2826
 H  2 r G xxrxG  0.0472  0.7036
 H  2 r P xxrxP  0.0337  0.5027
 H  2 a + xxax+ -0.0159  0.2376
 H  2 + G xx+xG  0.0113  0.1685
 H  2 + r xx+xr -0.0307  0.4575
 H  2 - P xx-xP -0.0088  0.1318
HA -1 P P xPPxx  0.0307  1.1685
HA -1 P r xrPxx  0.0621  2.3592
HA -1 r G xGrxx -0.0371  1.4092
HA -1 r + x+rxx  0.0125  0.4733
HA -1 r p xprxx -0.0199  0.7569
HA -1 a G xGaxx  0.0073  0.2779
HA -1 a a xaaxx  0.0044  0.1683
HA -1 - G xG-xx  0.0116  0.4409
HA -1 - r xr-xx  0.0228  0.8679
HA -1 - p xp-xx  0.0074  0.2828
HA  1 G G xxGGx  0.0175  0.6636
HA  1 G - xxG-x  0.0107  0.4081
HA  1 P a xxPax  0.0089  0.3369
HA  1 - r xx-rx  0.0113  0.4291
HA -2 G G GxGxx -0.0154  0.5847
HA -2 P - -xPxx  0.0136  0.5179
HA -2 r G Gxrxx -0.0159  0.6045
HA -2 + + +x+xx -0.0137  0.5190
HA -2 p - -xpxx -0.0068  0.2592
HA -2 p p pxpxx  0.0046  0.1763
HB -1 P r xrPxx  0.0460  2.1365
HB -1 a - x-axx  0.0076  0.3551
HB -1 + - x-+xx  0.0110  0.5122
HB -1 - r xr-xx  0.0233  1.0819
HB  1 a P xxaPx  0.0287  1.3310
HB  1 + P xx+Px  0.0324  1.5056
HB  1 + r xx+rx -0.0231  1.0709
HB  1 p r xxprx  0.0077  0.3586
HB  1 p + xxp+x -0.0074  0.3426
HB -2 a P Pxaxx -0.0026  0.1192
HB -2 a r rxaxx -0.0098  0.4559
HB -2 - - -x-xx  0.0016  0.0751
HB  2 P r xxPxr -0.0595  2.7608
HB  2 P + xxPx+ -0.0145  0.6744
HB  2 P - xxPx-  0.0107  0.4976
HB  2 a + xxax+ -0.0015  0.0691
HB  2 p r xxpxr  0.0262  1.2178'''

tablephshifts='''
D (pKa 3.86)
D H  8.55 8.38 -0.17 0.02 -0.03
D HA 4.78 4.61 -0.17 0.01 -0.01
D HB 2.93 2.70 -0.23
D CA 52.9 54.3 1.4 0.0 0.1
D CB 38.0 41.1 3.0
D CG 177.1 180.3 3.2
D C  175.8 176.9 1.1 -0.2 0.4
D N  118.7 120.2 1.5  0.3 0.1
D Np na na 0.1
E (pKa 4.34)
E H  8.45 8.57  0.12 0.00 0.02
E HA 4.39 4.29 -0.10 0.01 0.00
E HB 2.08 2.02 -0.06
E HG 2.49 2.27 -0.22
E CA 56.0 56.9 1.0 0.0 0.0
E CB 28.5 30.0 1.5
E CG 32.7 36.1 3.5
E CD 179.7 183.8 4.1
E C  176.5 177.0 0.6  0.1 0.1 
E N  119.9 120.9 1.0  0.2 0.1
E Np na na 0.1
H (pKa 6.45)
H H  8.55 8.35 -0.2  -0.01  0.0
H HA 4.75 4.59 -0.2  -0.01 -0.06
H HB 3.25 3.08 -0.17
H HD2 7.30 6.97 -0.33
H HE1 8.60 7.68 -0.92
H CA 55.1 56.7 1.6 -0.1 0.1
H CB 28.9 31.3 2.4
H CG 131.0 135.3 4.2
H CD2 120.3 120.0 -0.3
H CE1 136.6 139.2 2.6
H C 174.8 176.2 1.5  0.0 0.6
H N 117.9 119.7 1.8  0.3 0.5
H Np na na 0.5
H ND1 175.8 231.3 56
H NE2 173.1 181.1 8
C (pKa 8.49)
C H 8.49 8.49 0.0
C HA 4.56 4.28 -0.28 -0.01 -0.01
C HB 2.97 2.88 -0.09
C CA 58.5 60.6 2.1 0.0 0.1
C CB 28.0 29.7 1.7
C C 175.0 176.9 1.9 -0.4 0.5
C N 118.7 122.2 3.6  0.4 0.6
C Np na na 0.6
Y (pKa 9.76)
Y H  8.16 8.16 0.0
Y HA 4.55 4.49 -0.06
Y HB 3.02 2.94 -0.08
Y HD 7.14 6.97 -0.17
Y HE 6.85 6.57 -0.28
Y CA 58.0 58.2 0.3
Y CB 38.6 38.7 0.1
Y CG 130.5 123.8 -6.7
Y CD 133.3 133.2 -0.1
Y CE 118.4 121.7 3.3
Y CZ 157.0 167.4 10.4
Y C 176.3 176.7 0.4
Y N 120.1 120.7 0.6
K (pKa 10.34)
K H  8.4  8.4  0.0
K HA 4.34 4.30 -0.04
K HB 1.82 1.78 -0.04
K HG 1.44 1.36 -0.08
K HD 1.68 1.44 -0.24
K HE 3.00 2.60 -0.40
K CA 56.4 56.9 0.4
K CB 32.8 33.2 0.3
K CG 24.7 25.0 0.4
K CD 28.9 33.9 5.0
K CE 42.1 43.1 1.0
K C 177.0 177.5 0.5
K N 121.0 121.7 0.7
K Np na na 0.1
R (pKa 13.9)
R H  7.81 7.81 0.0
R HA 3.26 3.19 -0.07
R HB 1.60 1.55 0.05
R HG 1.60 1.55 0.05
R HD 3.19 3.00 -0.19
R CA 58.4 58.6 0.2
R CB 34.4 35.2 0.9
R CG 27.2 28.1 1.0
R CD 43.8 44.3 0.5
R CZ 159.6 163.5 4.0
R C 185.8 186.1 0.2
R N 122.4 122.8 0.4
R NE 85.6 91.5 5.9
R NG 71.2 93.2 22'''

def initcorrcomb():
    datc=string.split(tablecombdevs,'\n')
    buf=[string.split(lin) for lin in datc]
    dct={}
    for lin in buf:
      atn=lin[0]
      if not atn in dct:dct[atn]={}
      neipos=string.atoi(lin[1])
      centgroup=lin[2]
      neigroup= lin[3]
      key=(neipos,centgroup,neigroup)#(k,l,m)
      segment=lin[4]
      dct[atn][segment]=key,eval(lin[-2])
    return dct
	
TEMPCORRS=gettempkoeff()
##dct[atn][aa]=eval(lin[1+j])
CENTSHIFTS=initcorcents()
##dct[aai][atnj]=eval(vals[1+j])
NEICORRS =initcorneis()
##dct[aai][atn]=[eval(vals[2+j]) for j in range(4)]
COMBCORRS=initcorrcomb()
##dct[atn][segment]=key,eval(lin[-1])

def predPentShift(pent,atn):
    aac=pent[2]
    sh=CENTSHIFTS[aac][atn]
    allneipos=[2,1,-1,-2]
    for i in range(4):
	aai=pent[2+allneipos[i]]
	if aai in NEICORRS:
	  corr=NEICORRS[aai][atn][i]
	  sh+=corr
    groups=['G','P','FYW','LIVMCA','KR','DE']##,'NQSTHncX']
    labels='GPra+-p' #(Gly,Pro,Arom,Aliph,pos,neg,polar)
    grstr=''
    for i in range(5):
	aai=pent[i]
	found=False
	for j,gr in enumerate(groups):
	  if aai in gr:
	    grstr+=labels[j]
	    found=True
	    break
	if not found:grstr+='p'#polar
    centgr=grstr[2]
    for segm in COMBCORRS[atn]:
	key,combval=COMBCORRS[atn][segm]
        neipos,centgroup,neigroup=key#(k,l,m)
	if centgroup==centgr and grstr[2+neipos]==neigroup:
	 if (centgr,neigroup)<>('p','p') or pent[2] in 'ST':
	  #pp comb only used when center is Ser or Thr!
	  sh+=combval
    return sh
	
def gettempcorr(aai,atn,tempdct,temp):
    return tempdct[atn][aai]/1000*(temp-298)

def get_phshifts():
    datc=string.split(tablephshifts,'\n')
    buf=[string.split(lin) for lin in datc]
    dct={}
    na=None
    for lin in buf:
      if len(lin)>3:
	resn=lin[0]
	atn=lin[1]
	sh0=eval(lin[2])
	sh1=eval(lin[3])
	shd=eval(lin[4])
	if not resn in dct:dct[resn]={}
	dct[resn][atn]=shd
	if len(lin)>6:#neighbor data
	  for n in range(2):
	    shdn=eval(lin[5+n])
	    nresn=resn+'ps'[n]
	    if not nresn in dct:dct[nresn]={}
	    dct[nresn][atn]=shdn
    return dct

def initfilcsv(filename):
  file=open(filename,'r')
  buffer=file.readlines()
  file.close()
  for i in range(len(buffer)):
     buffer[i]=string.split(buffer[i][:-1],',')
  return buffer

def write_csv_pkaoutput(pkadct,seq,temperature,ion):
	seq=seq[:min(150,len(seq))]
	name='outpepKalc_%s_T%6.2f_I%4.2f.csv'%(seq,temperature,ion)
	out=open(name,'w')
	out.write('Site,pKa value,pKa shift,Hill coefficient\n')
	for i in pkadct:
	  pKa,nH,resi=pkadct[i]
	  reskey=resi+str(i+1)
	  diff=pKa-pK0[resi]
	  out.write('%s,%5.3f,%5.3f,%5.3f\n'%(reskey,pKa,diff,nH))
	out.close()

def read_csv_pkaoutput(seq,temperature,ion,name=None):
	seq=seq[:min(150,len(seq))]
        if VERB:print 'reading csv',name
	if name==None:name='outpepKalc_%s_T%6.2f_I%4.2f.csv'%(seq,temperature,ion)
	try:out=open(name,'r')
	except IOError:return None
	buf=initfilcsv(name)
	for lnum,data in enumerate(buf):
	  if len(data)>0 and data[0]=='Site':break
	pkadct={}
	for data in buf[lnum+1:]:
	  reskey,pKa,diff,nH=data
	  i=string.atoi(reskey[1:])-1
	  resi=reskey[0]
	  pKaval=eval(pKa)
	  nHval=eval(nH)
	  pkadct[i]=pKaval,nHval,resi
	return pkadct

def getphcorrs(seq,temperature,pH,ion,pkacsvfilename=None):
	bbatns=['C','CA','CB','HA','H','N','HB']
        dct=get_phshifts()
	Ion=max(0.0001,ion)
	pkadct=read_csv_pkaoutput(seq,temperature,ion,pkacsvfilename)
	if pkadct==None:
	  pkadct=calc_pkas_from_seq('n'+seq+'c',temperature,Ion)
	  write_csv_pkaoutput(pkadct,seq,temperature,ion)
	outdct={}
	for i in pkadct:
	  if VERB:print 'pkares: %6.3f %6.3f %1s'%pkadct[i],i
	  pKa,nH,resi=pkadct[i]
	  frac =fun(pH,pKa,nH)
	  frac7=fun(7.0,pK0[resi],nH)
	  if resi in 'nc':jump=0.0#so far
	  else:
	   for atn in bbatns:
	    if not atn in outdct:outdct[atn]={}
	    if VERB:print 'data:',atn,pKa,nH,resi,i,atn,pH
	    dctresi=dct[resi]
	    try:
		delta=dctresi[atn]
		jump =frac *delta
		jump7=frac7*delta
		key=(resi,atn)
	    except KeyError:
	        ##if not (resi in 'RKCY' and atn=='H') and not (resi == 'R' and atn=='N'):
		print 'warning no key:',resi,i,atn
		delta=999;jump=999;jump7=999
	    if delta<99:
	     jumpdelta=jump-jump7
	     if not i in outdct[atn]:outdct[atn][i]=[resi,jumpdelta]
	     else:
		outdct[atn][i][0]=resi
		outdct[atn][i][1]+=jumpdelta
	     if VERB:print '%3s %5.2f %6.4f %s %3d %5s %8.5f %8.5f %4.2f'%(atn,pKa,nH,resi,i,atn,jump,jump7,pH)
	     if resi+'p' in dct and atn in dct[resi+'p']:
	      for n in range(2):
	        ni=i+2*n-1
	        ##if ni is somewhere in seq...
		nresi=resi+'ps'[n]
		ndelta=dct[nresi][atn]
		jump =frac *ndelta
		jump7=frac7*ndelta
		jumpdelta=jump-jump7
	        if not ni in outdct[atn]:outdct[atn][ni]=[None,jumpdelta]
	        else:outdct[atn][ni][1]+=jumpdelta
	return outdct

def getpredshifts(seq,temperature,pH,ion,usephcor=True,pkacsvfile=None,identifier=''):
        tempdct=gettempkoeff()
        bbatns =['C','CA','CB','HA','H','N','HB']
	if usephcor:
	  phcorrs=getphcorrs(seq,temperature,pH,ion,pkacsvfile)
	else:phcorrs={}
	shiftdct={}
	for i in range(1,len(seq)-1):
	  if seq[i] in AAstandard:#else: do nothing
	    res=str(i+1)
	    trip=seq[i-1]+seq[i]+seq[i+1]
	    phcorr=None
	    shiftdct[(i+1,seq[i])]={}
	    for at in bbatns:
	      if not (trip[1],at) in [('G','CB'),('G','HB'),('P','H')]:
		if i==1:
		  pent='n'+     trip+seq[i+2]
		elif i==len(seq)-2:
		  pent=seq[i-2]+trip+'c'
		else:
		  pent=seq[i-2]+trip+seq[i+2]
	        shp=predPentShift(pent,at)
		if shp<>None:
	          if at<>'HB':shp+=gettempcorr(trip[1],at,tempdct,temperature)
		  if at in phcorrs and i in phcorrs[at]:
		     phdata=phcorrs[at][i]
		     resi=phdata[0]
		     ##assert resi==seq[i]
		     if seq[i] in 'CDEHRKY' and resi<>seq[i]:
			print 'WARNING: residue mismatch',resi,seq[i],i,phdata,at
		     phcorr=phdata[1]
		     if abs(phcorr)<9.9:
		       shp-=phcorr
		  shiftdct[(i+1,seq[i])][at]=shp
		  if VERB:print 'predictedshift: %5s %3d %1s %2s %8.4f'%(identifier,i,seq[i],at,shp),phcorr
	return shiftdct

def writeOutput(name,dct):
	try:out=open(name,'w')
	except IOError:
	  print 'warning file name too long!',len(name),name
	  subnames=name.split('_')
	  name=string.join([subnames[0]]+[subnames[1][:120]]+subnames[2:],'_')
	  out=open(name,'w')
        bbatns =['N','C','CA','CB','H','HA','HB']
	out.write('#NUM AA   N ')
	out.write(' %7s %7s %7s %7s %7s %7s\n'%tuple(bbatns[1:]))
	reskeys=dct.keys();reskeys.sort()
	for resnum,resn in reskeys:
	  shdct=dct[(resnum,resn)]
	  if len(shdct)>0:
	    out.write('%-4d %1s '%(resnum,resn))
	    for at in bbatns:
	      shp=0.0
	      if at in shdct:shp=shdct[at]
	      out.write(' %7.3f'%shp)
	  out.write('\n')
	out.close()

def potenci(seq,pH=7.0,temp=298,ion=0.1,pkacsvfile=None,doreturn=True):
    ##README##......
    #requires: python with numpy and scipy
    #usage: potenci1_0.py seqstring pH temp ionicstrength [pkacsvfile] > logfile
    #optional filename in csv format contained predicted pKa values and Hill parameters,
    #the format of the pkacsvfile must be the same as the output for pepKalc,
    #only lines after "Site" is read. If this is not found no pH corrections are applied.
    #Output: Table textfile in SHIFTY format (space separated)
    #average of methylene protons are provided for Gly HA2/HA3 and HB2/HB3.
    #NOTE:pH corrections is applied if pH is not 7.0
    #NOTE:pKa predictions are stored locally and reloaded if the seq, temp and ion is the same.
    #NOTE:at least 5 residues are required. Chemical shift predictions are not given for terminal residues.
    #NOTE:change the value of VERB in the top of this script to have verbose logfile
    ##name='outPOTENCI_%s_T%6.2f_I%4.2f_pH%4.2f.txt'%(seq,temp,ion,pH)
    name='outPOTENCI_%s_T%6.2f_I%4.2f_pH%4.2f.txt'%(seq[:min(150,len(seq))],temp,ion,pH)
    usephcor = pH<6.99 or pH>7.01
    if len(seq)<5:
	print 'FAILED: at least 5 residues are required (exiting)' 
	raise SystemExit
    #------------- now ready to generate predicted shifts ---------------------
    print 'predicting random coil chemical shift with POTENCI using:',seq,pH,temp,ion,pkacsvfile
    shiftdct=getpredshifts(seq,temp,pH,ion,usephcor,pkacsvfile)
    #------------- write output nicely is SHIFTY format -----------------------$
    writeOutput(name,shiftdct)
    print 'chemical shift succesfully predicted, see output:',name
    if doreturn: return shiftdct

histdct_flattened=(2.0737054913103337e-10, 0.029518200092680934, 0.97048179969994852, 8.9102609389174977e-10, 0.049071251975264775, 0.95092874713370923, 3.6911845844839106e-09, 0.079951554923860677, 0.92004844138495467, 1.4645276032946968e-08, 0.12682756718735982, 0.87317241816736413, 5.5170502882584017e-08, 0.19418179262213037, 0.80581815220736663, 1.9535866625876679e-07, 0.28408625538027543, 0.71591354926105832, 6.4415744419331008e-07, 0.39341966288684982, 0.60657969295570613, 0.52492556339882823, 0.24358165501654738, 0.2314927815846245, 5.5564750876404044e-06, 0.62885124891949329, 0.37114319460541911, 2.1057352549240146e-05, 0.61171750797158275, 0.38826143467586793, 3.0971176979542517e-05, 0.51981670536451852, 0.48015232345850201, 7.915310450907783e-05, 0.61884065214048178, 0.38108019475500915, 0.00045690835118482965, 0.79503735616235505, 0.20450573548646017, 0.00037479162478264302, 0.35106708082817562, 0.64855812754704179, 0.00095386218989570288, 0.57450903465177372, 0.4245371031583306, 0.0020603975788089112, 0.61761448046266132, 0.38032512195852974, 0.10697471251615311, 0.740936476567855, 0.15208881091599186, 0.19327399498223335, 0.66933460014069035, 0.13739140487707624, 0.012958657581034067, 0.8553594104147666, 0.13168193200419931, 0.091851181267882026, 0.77756153682517193, 0.13058728190694613, 0.22921634513601896, 0.70560705274133872, 0.065176602122642202, 0.17138260006224124, 0.74739765730864161, 0.081219742629117173, 0.39611931387854205, 0.53348400777148219, 0.070396678349975719, 0.32077386008589764, 0.67887372861950723, 0.00035241129459517807, 0.34547036052878238, 0.53173852579655434, 0.12279111367466328, 0.15654016376325999, 0.84329817825782072, 0.00016165797891921677, 0.609236196608406, 0.39071653468646333, 4.7268705130734661e-05, 0.56507190622112824, 0.43487160797697183, 5.6485801899947119e-05, 0.72211884215624933, 0.27786639061478047, 1.4767228970310703e-05, 0.89076020079356311, 0.1092234770391385, 1.6322167298468004e-05, 0.9651593658187213, 0.034836824638462209, 3.8095428165094679e-06, 0.98971313476752076, 0.010286035965388558, 8.2926709062859595e-07, 0.96981453223812519, 0.03018366029636706, 1.8074655077167692e-06, 0.9800419677234179, 0.019957137967335396, 8.9430924670088857e-07, 0.9866363147631777, 0.013363233763595815, 4.5147322651919685e-07, 1.9109838135778465e-10, 0.023630638204507205, 0.97636936160439436, 8.2442113579148138e-10, 0.039442231792554197, 0.96055776738302479, 3.4371676755121813e-09, 0.064675160889006281, 0.93532483567382607, 1.3771494331934358e-08, 0.10360311557528777, 0.89639687065321794, 5.2622165192033297e-08, 0.16089625102505495, 0.83910369635277982, 1.8996806314544746e-07, 0.23997930062335224, 0.76002050940858468, 6.4159584961185574e-07, 0.34040915035607228, 0.65959020804807811, 1.6142671821825171e-06, 0.3656723330296458, 0.63432605270317199, 7.9306224762809292e-06, 0.42167471692381381, 0.57831735245370997, 1.4331997234628106e-05, 0.62168294461968965, 0.37830272338307569, 8.5079831053122454e-05, 0.35116882723558307, 0.64874609293336394, 0.00023739564627928063, 0.60858211393349881, 0.39118049042022185, 0.00027350017133546272, 0.35110265425567283, 0.64862384557299169, 0.0013094397503175168, 0.80436154464227105, 0.19432901560741142, 0.0011593081625392108, 0.6181721553663625, 0.38066853647109822, 0.0017483116290761852, 0.60143525528397734, 0.39681643308694647, 0.062346515643269242, 0.67173435620272981, 0.26591912815400082, 0.12117764676552746, 0.79268160196400617, 0.086140751270466262, 0.025364944414347012, 0.96071846892327084, 0.013916586662382227, 0.1253815227369563, 0.86842714038053614, 0.0061913368825075822, 0.035564561379828476, 0.71161999636365836, 0.25281544225651315, 0.05127083074821575, 0.94697591712442863, 0.0017532521273556747, 0.33683647246616599, 0.66246322641254274, 0.00070030112129126204, 0.20613528440390783, 0.7931944731511591, 0.00067024244493308304, 0.25475130826701098, 0.74500186152316994, 0.00024683020981902096, 0.38096153535572541, 0.618941435995089, 9.7028649185720704e-05, 0.4380554386001887, 0.56186912046014503, 7.5440939666278716e-05, 0.72201381857408975, 0.27782597825880734, 0.00016020316710293287, 0.46415121975875845, 0.53580664263808608, 4.2137603155417023e-05, 0.83863471128469635, 0.16135044999873116, 1.483871657248392e-05, 0.9127086014213116, 0.087279403938147634, 1.1994640540753855e-05, 0.95968276815616971, 0.040313147387399932, 4.0844564303048067e-06, 0.97959790349402753, 0.020400561247868244, 1.5352581042162526e-06, 0.98243937381474755, 0.017559637299381859, 9.8888587070698638e-07, 0.98837030712191354, 0.01162919912319507, 4.9375489140841599e-07, 1.7586070645157005e-10, 0.018919495073917988, 0.98108050475022135, 7.6114140518621012e-10, 0.031681081386595179, 0.96831891785226343, 1.5738979951325724e-09, 0.025765327484998785, 0.97423467094110328, 1.2883779943905633e-08, 0.084325190757747803, 0.91567479635847215, 4.9822664459746072e-08, 0.13253373783046729, 0.86746621234686827, 1.938707366951566e-07, 0.21307266355241578, 0.78692714257684759, 6.3126935162206797e-07, 0.29139135982217446, 0.70860800890847397, 3.1778932524635995e-06, 0.35119759109025267, 0.64879923101649484, 7.3222033521025827e-06, 0.62630845676289149, 0.37368422103375648, 3.5011493538620542e-05, 0.51981460507542765, 0.48015038343103372, 7.1272798885618945e-05, 0.15284352413112626, 0.84708520306998814, 0.00015475183905515138, 0.72141337200465983, 0.27843187615628495, 0.00027694019364964242, 0.43099593807618158, 0.56872712173016882, 0.0005940761539064874, 0.47400377256513898, 0.52540215128095458, 0.002552483169753832, 0.70714312419390568, 0.29030439263634056, 0.048136277614296952, 0.81499045482398769, 0.13687326756171533, 0.090692589975098353, 0.65142724668963636, 0.25788016333526537, 0.03745858302666627, 0.74951793816703616, 0.21302347880629754, 0.029535710506623215, 0.88648086977008478, 0.083983419723291999, 0.10213292270407413, 0.89080113308337361, 0.0070659442125522078, 0.08366492548264258, 0.8370359658293498, 0.079299108688007519, 0.24465858755181566, 0.7531436066998346, 0.0021978057483498241, 0.33657682568722619, 0.66195257374585037, 0.0014706005669234679, 0.40230659369205496, 0.59710432564384208, 0.00058908066410307588, 0.29238446867227558, 0.70719027286905833, 0.00042525845866613841, 0.40491597418577552, 0.59490669737192015, 0.00017732844230438293, 0.56505048739799879, 0.4348551243437338, 9.4388258267336206e-05, 0.56506660125667107, 0.4348675253489056, 6.5873394423309603e-05, 0.7958164435359204, 0.2041498156516921, 3.3740812387630269e-05, 0.68409674991869873, 0.31588289954646637, 2.0350534834945435e-05, 0.94542022483429133, 0.054570363018285849, 9.4121474229599436e-06, 0.92863027799932507, 0.071360647917449158, 9.0740832258402936e-06, 0.93479652168789862, 0.065197320514038481, 6.1577980629425805e-06, 0.98484176376230692, 0.015157164952703448, 1.071284989674963e-06, 0.98799996363359788, 0.011999396959327391, 6.3940707466488445e-07, 1.6165617676271101e-10, 0.015153116260222027, 0.98484688357812189, 7.0147936153724939e-10, 0.025440079913643904, 0.97455991938487674, 2.9521305956832856e-09, 0.042107899156463249, 0.95789209789140617, 1.2002430192798381e-08, 0.068446641981493875, 0.93155334601607587, 3.9989238031081283e-08, 0.23965735406034891, 0.76034260595041314, 1.8473809999672171e-07, 0.17690531070607751, 0.82309450455582256, 1.1311414665106574e-06, 0.45493338944441802, 0.54506547941411554, 3.5240316153704533e-06, 0.60512861536107743, 0.39486786060730722, 9.7525092748002545e-06, 0.72682822772674927, 0.27316201976397597, 2.0861086226669762e-05, 0.6859466903146072, 0.31403244859916624, 5.332729240920751e-05, 0.18829335186004104, 0.81165332084754971, 0.00025945729180097395, 0.47416247745093659, 0.52557806525726247, 0.00035725454767968995, 0.43592108982262995, 0.56372165562969045, 0.0010897843172447664, 0.38177825445219243, 0.6171319612305628, 0.0018591181203997816, 0.53394779913297052, 0.46419308274662957, 0.0045928219568831148, 0.57925391139005111, 0.41615326665306585, 0.031283424727716821, 0.74633416562103239, 0.22238240965125072, 0.10385850320845873, 0.85922689359944548, 0.036914603192095684, 0.11120289139689665, 0.77022201413565816, 0.11857509446744517, 0.043321716017784687, 0.83349499849331521, 0.12318328548890009, 0.21041134479899304, 0.69754328950803046, 0.092045365692976513, 0.2123196322567944, 0.7625253259234076, 0.0251550418197981, 0.3312304844007723, 0.64398429234572929, 0.024785223253498315, 0.3273376571679506, 0.67177221993330294, 0.00089012289874644697, 0.48962085422104135, 0.50979569087113141, 0.00058345490782740013, 0.46406754194472771, 0.53571004668683031, 0.00022241136844202664, 0.47346135887494539, 0.5263113326065374, 0.00022730851851715877, 0.56504995961065041, 0.43485471816581145, 9.5322223538082067e-05, 0.92115472651613373, 0.078767563857399928, 7.7709626466389205e-05, 0.88629770933571506, 0.11368043186533899, 2.185879894596154e-05, 0.83862783079609726, 0.16134912621625858, 2.3042987644162106e-05, 0.96631295876816758, 0.033681673271623441, 5.3679602088754854e-06, 0.838644851080249, 0.16135240086071348, 2.7480590375089156e-06, 0.98440741136075338, 0.015591207496872617, 1.3811423740147136e-06, 0.98108244712858661, 0.018916289515771071, 1.2633556422733213e-06, 1.4846190684672449e-10, 0.01214339852203244, 0.98785660132950559, 6.4556439373447552e-10, 0.02042954543601691, 0.97957045391841868, 2.7259985755764897e-09, 0.033928854284672787, 0.96607114298932861, 1.1142560576130246e-08, 0.055447698327182209, 0.94455229053025735, 1.0107812694009762e-07, 0.35119867166374696, 0.64880122725812617, 1.654000159869412e-07, 0.13820873971878719, 0.86179109488119687, 2.3223672356741317e-06, 0.35119789154988373, 0.64879978608288058, 2.4408824174268045e-06, 0.36573819315220912, 0.63425936596537358, 7.2097582121844249e-06, 0.063374404788280117, 0.93661838545350773, 3.0386781128102376e-05, 0.071776381334351963, 0.92819323188451985, 6.5650760385645707e-05, 0.08959463451860368, 0.91033971472101072, 0.00044744139811345109, 0.47407331930863594, 0.52547923929325069, 0.00067487507718573005, 0.38193683059140787, 0.61738829433140652, 0.045779702813143329, 0.56370292717454962, 0.39051737001230707, 0.026856114715831315, 0.74405154428792653, 0.22909234099624223, 0.035269922704626547, 0.81429749900443538, 0.15043257829093798, 0.048924812447079517, 0.76558794795279728, 0.18548723960012309, 0.079046798759930509, 0.87600002175553215, 0.04495317948453733, 0.096444017823476252, 0.8349974619163345, 0.068558520260189354, 0.19376178818706802, 0.76688445204086342, 0.039353759772068501, 0.18454410738301449, 0.81036436229132403, 0.005091530325661486, 0.28381399090869897, 0.68256051372628801, 0.033625495365013074, 0.32660290681419762, 0.64137363863754115, 0.032023454548261152, 0.36082178603061182, 0.61944782985149038, 0.01973038411789782, 0.55761760296151508, 0.44175649223864227, 0.00062590479984271999, 0.55656155484308267, 0.44309188757699419, 0.00034655757992309489, 0.47465596158444351, 0.5251025551443339, 0.00024148327122261258, 0.81529190442053157, 0.18454043976402412, 0.00016765581544424501, 0.89404820563457432, 0.10585342570845434, 9.8368656971371839e-05, 0.94785806600754141, 0.052104185674751766, 3.7748317706797921e-05, 0.83863502030838277, 0.16135050945383145, 1.4470237785757608e-05, 0.94231374952028113, 0.057674745437563832, 1.1505042155123695e-05, 0.9230730445097739, 0.076915559070109427, 1.139642011672311e-05, 0.98243351081584129, 0.017564541661950692, 1.9475222079740559e-06, 0.98569633242627486, 0.014302471968853316, 1.1956048718005995e-06, 1.36240635668259e-10, 0.0097385449777252827, 0.99026145488603401, 5.9340775954517885e-10, 0.01641098147654356, 0.9835890179300486, 2.5125697989399905e-09, 0.02732901032883624, 0.9726709871585939, 3.9446330598836948e-08, 0.17154075963781867, 0.82845920091585068, 1.6504487039990168e-07, 0.35119864919870936, 0.64880118575642032, 2.1344063721831815e-07, 0.1558616494389477, 0.84413813712041508, 2.2998920896228269e-06, 0.21300195540160005, 0.78699574470631029, 4.5359774978689756e-06, 0.097684860082922875, 0.90231060393957918, 1.499302138667569e-05, 0.40355941577024274, 0.59642559120837046, 3.0550408219921344e-05, 0.26516747696249077, 0.73480197262928926, 7.9715726522888093e-05, 0.19987691221010465, 0.8000433720633725, 0.00023722907701973362, 0.43101305814738139, 0.56874971277559883, 0.00066614456557445014, 0.51948652140013785, 0.4798473340342877, 0.046027174247046672, 0.49590637009413346, 0.45806645565881976, 0.057860713654857072, 0.66793220408137521, 0.27420708226376761, 0.028272367394338957, 0.75065193623726156, 0.22107569636839949, 0.036174255405063901, 0.72381951305071224, 0.24000623154422374, 0.074301679440815213, 0.7596978145870138, 0.16600050597217098, 0.10088294103963727, 0.84696143117085221, 0.052155627789510592, 0.15153295257912125, 0.79177271768957602, 0.056694329731302603, 0.16258775171205039, 0.78487693995163177, 0.052535308336317817, 0.30110694619777884, 0.69518335856616831, 0.0037096952360529539, 0.39164241310540737, 0.59625305434697995, 0.012104532547612675, 0.41870766798824777, 0.58001704689659284, 0.0012752851151593718, 0.48371653941491172, 0.51543918103731157, 0.00084427954777679742, 0.52190589673625576, 0.47763978531906004, 0.00045431794468420123, 0.49624968038443801, 0.50342281692885249, 0.00032750268670954889, 0.59544595576665271, 0.40433564200905803, 0.00021840222428923535, 0.89028496661588385, 0.1096241807748886, 9.0852609227485534e-05, 0.88760593168518243, 0.11234214495144711, 5.192336337035488e-05, 0.76460113373795835, 0.23537062864316166, 2.8237618879884617e-05, 0.9285295070485976, 0.071458360491606879, 1.2132459795584331e-05, 0.96713858224879545, 0.03285533272768458, 6.0850235199357034e-06, 0.9073905786752221, 0.092596587895467833, 1.283342930992485e-05, 0.96023262996968806, 0.039763215133852226, 4.1548964598639039e-06, 1.249460449550858e-10, 0.0078166157623807755, 0.99218338411267315, 5.4493784644288464e-10, 0.013189767582097526, 0.98681023187296457, 4.1046417517442006e-09, 0.039074179331734717, 0.96092581656362352, 5.7494043326152375e-08, 0.21882271798998215, 0.78117722451597449, 2.5511722740456355e-07, 0.35119861756541404, 0.64880112731735862, 4.599285302560988e-07, 0.29394179850740626, 0.70605774156406342, 2.2129124459681669e-06, 0.26517499137431694, 0.73482279571323705, 3.8534353451592905e-06, 0.10737338509950629, 0.89262276146514852, 1.7683866271391761e-05, 0.43110770745363519, 0.56887460868009343, 3.5597241834713196e-05, 0.19988573117960604, 0.80007867157855916, 6.5297850503057936e-05, 0.28245446610122588, 0.71748023604827116, 0.00025963522600652424, 0.43596365935187936, 0.56377670542211411, 0.00055629085636108022, 0.45216340899019891, 0.54728030015344009, 0.0011350170661250966, 0.5137607624461642, 0.48510422048771068, 0.012020945557803323, 0.62907855606863516, 0.35890049837356147, 0.0069552027468725254, 0.68284458437285678, 0.31020021288027061, 0.055921114734906352, 0.76235421097882172, 0.18172467428627198, 0.070118303391193959, 0.82022372185086045, 0.10965797475794549, 0.10214531120142367, 0.7953445321949375, 0.1025101566036388, 0.15660989240593035, 0.78964544141622484, 0.053744666177844874, 0.21502193493428201, 0.75718693435163997, 0.027791130714078106, 0.31085898742543339, 0.65152769006198941, 0.037613322512577166, 0.36271872557927254, 0.62807257947832196, 0.0092086949424054591, 0.46811728543474374, 0.53037785806420867, 0.0015048565010475092, 0.50987561748906873, 0.46555137209677477, 0.024573010414156545, 0.63361890611508076, 0.36571833095040768, 0.00066276293451154661, 0.66071512117548392, 0.33898487843992936, 0.00030000038458678053, 0.88366574495923411, 0.11610730419318148, 0.00022695084758433308, 0.86392095832233629, 0.13599444881463338, 8.4592863030316664e-05, 0.88628043827939951, 0.11367821660389803, 4.1345116702401145e-05, 0.93970992205548809, 0.060265658278344798, 2.4419666167040562e-05, 0.96209674471090179, 0.037891461227038382, 1.1794062059785199e-05, 0.94788907403898592, 0.052105890200234789, 5.0357607793574558e-06, 0.98065622512876816, 0.019340428830497556, 3.3460407343359303e-06, 0.94694421969391585, 0.053048860846764551, 6.9194593196676287e-06, 1.1452663474520966e-10, 0.0062799794920764271, 0.99372002039339691, 2.853257157209107e-09, 0.060532214515358576, 0.93946778263138431, 1.6214598025741835e-08, 0.13529329344145699, 0.86470669034394498, 2.1189337446152076e-08, 0.21300244077085834, 0.78699753803980421, 7.3955721368779058e-08, 0.10027259913410227, 0.89972732691017632, 2.7127367296178246e-07, 0.051350744882100251, 0.94864898384422691, 1.1373150305372755e-06, 0.13970426155079299, 0.86029460113417644, 5.6386873784185915e-06, 0.10737319341021356, 0.89262116790240809, 9.1443181671856631e-06, 0.1958725816555755, 0.80411827402625724, 3.0780835544700696e-05, 0.23623548224942459, 0.76373373691503066, 0.018866399425582821, 0.20327052027286135, 0.77786308030155582, 0.00016186564366182682, 0.39934577771479773, 0.60049235664154044, 0.00055400522477567474, 0.50935418305662183, 0.49009181171860233, 0.0011543244279179205, 0.60305481615717316, 0.39579085941490899, 0.0093830174487906804, 0.65711480876058237, 0.33350217379062697, 0.02712460037708728, 0.68364712401760941, 0.28922827560530329, 0.071627369982289954, 0.76713459519396177, 0.16123803482374829, 0.039927699342942116, 0.83843027544955162, 0.12164202520750628, 0.14656321780515202, 0.74588950576415913, 0.10754727643068876, 0.16376035719753274, 0.78848118510568288, 0.047758457696784468, 0.19493473739530479, 0.77867060244504371, 0.026394660159651598, 0.36120498883352231, 0.61929368977117705, 0.019501321395300596, 0.33948133935494107, 0.63170375299979131, 0.028814907645267729, 0.51829821941476095, 0.47954661106164709, 0.0021551695235920544, 0.58059215907068418, 0.40958113371002164, 0.0098267072192940765, 0.71672312276462669, 0.28251674384286046, 0.00076013339251280619, 0.77942164030452332, 0.22019157835612072, 0.00038678133935605036, 0.86461414507544643, 0.13515840341285848, 0.00022745151169498992, 0.87144894223037694, 0.12842333294968455, 0.0001277248199384536, 0.92979811341202045, 0.070152932060478179, 4.895452750144501e-05, 0.95126278805944253, 0.048711574494744239, 2.563744581323103e-05, 0.97227436758017605, 0.02771292781793892, 1.2704601884907007e-05, 0.98237127145535186, 0.01762364926357245, 5.0792810756628159e-06, 0.9883308747746975, 0.011666609041297193, 2.51618400519803e-06, 0.98235119700233953, 0.017645933714761562, 2.8692828988985315e-06, 6.150922647499844e-10, 0.02960697961527442, 0.97039301976963332, 3.901427491963202e-09, 0.072655969185578692, 0.92734402691299378, 2.1823351141247776e-08, 0.15984300523259043, 0.84015697294405844, 1.6425441738108356e-08, 0.11919570971067156, 0.88080427386388671, 9.7614611940378502e-08, 0.26517555229964312, 0.73482435008574487, 5.701319637312147e-07, 0.28035220426049173, 0.71964722560754446, 9.3319247507560906e-07, 0.082751612748402772, 0.91724745405912222, 3.5530676313383773e-06, 0.097684956098771203, 0.90231149083359752, 1.0912076402876782e-05, 0.16873535134981291, 0.8312537365737841, 2.3104211625557428e-05, 0.19200984631769741, 0.8079670494706771, 8.8396020164666851e-05, 0.25937649579303113, 0.74053510818680424, 0.00022977833115246573, 0.31406919157580399, 0.68570103009304362, 0.00051050879418069637, 0.45177723595921199, 0.54771225524660727, 0.013015917668960585, 0.50585204409539097, 0.48113203823564832, 0.026157531993968221, 0.63914270022932174, 0.33469976777671007, 0.016468280264379884, 0.71817991354020883, 0.26535180619541132, 0.061697392653401247, 0.76286893962150992, 0.17543366772508873, 0.062288225164186743, 0.84915493858483237, 0.088556836250980861, 0.080077719163564112, 0.86570864472610753, 0.054213636110328381, 0.125712880508489, 0.82960477806789368, 0.04468234142361735, 0.23580485020524711, 0.72863832476703816, 0.035556825027714609, 0.27782924455626395, 0.70394011219599584, 0.01823064324774025, 0.36138756245183318, 0.62576757838464525, 0.012844859163521632, 0.55509185834700769, 0.44218026106753838, 0.0027278805854539867, 0.59550747330712761, 0.39558043292726031, 0.0089120937656120107, 0.68655511497381561, 0.31247299230750236, 0.00097189271868207243, 0.7954393293525992, 0.20405307501303099, 0.00050759563436990134, 0.90561026643893916, 0.094181793366656483, 0.00020794019440424204, 0.93326029567241164, 0.06663937315513993, 0.00010033117244839944, 0.97155646884745128, 0.028393540482592489, 4.9990669956191124e-05, 0.98709918457349344, 0.01287555978671461, 2.5255639792054125e-05, 0.96891974492996114, 0.031069474141213794, 1.0780928825086651e-05, 0.96761573797445677, 0.032376688412043095, 7.5736135001059529e-06, 0.90094551641916587, 0.099050763302711287, 3.7202781226453157e-06, 0.99133691074756192, 0.0086613359312496859, 1.7533211885125555e-06, 9.6097254311358254e-11, 0.0040664325175955333, 0.99593356738630712, 4.2020893703699151e-10, 0.0068795779582577097, 0.99312042162153336, 3.9879837830057039e-09, 0.063374861452971831, 0.93662513455904439, 3.0080683037737539e-08, 0.077439687097355894, 0.92256028282196101, 7.7186008207421418e-08, 0.039974282905150094, 0.96002563990884171, 3.6885786560369713e-07, 0.076875499296983479, 0.92312413184515085, 7.7055439693921474e-07, 0.069475346224996848, 0.9305238832206062, 2.3788787821454687e-06, 0.12468665514378201, 0.87531096597743596, 6.8106732164348888e-06, 0.1427741686108622, 0.85721902071592138, 2.5733347849939136e-05, 0.16308427171890702, 0.83688999493324312, 6.6605014300140638e-05, 0.31161863783095967, 0.68831475715474022, 0.00018680201424775851, 0.38941372835540933, 0.61039946963034297, 0.00061803537808386767, 0.55393249902621355, 0.44544946559570259, 0.0054109434599216352, 0.5330160712634614, 0.46157298527661689, 0.0097375594561176539, 0.68569116673948893, 0.30457127380439342, 0.0039141582307427852, 0.72897239469063568, 0.2671134470786215, 0.028926609312672959, 0.77915329683199297, 0.19192009385533398, 0.053433332039870884, 0.8452765375847382, 0.10129013037539092, 0.064694683146133941, 0.85436453433854931, 0.080940782515316792, 0.13413179047744975, 0.81527466221209133, 0.050593547310458876, 0.17498194500100919, 0.79836342861812759, 0.026654626380863269, 0.30176405042849586, 0.68253987551331341, 0.015696074058190668, 0.44444174379002854, 0.53878282121133103, 0.016775434998640436, 0.56975274670981035, 0.41169688556828721, 0.018550367721902401, 0.62105948045207948, 0.37692692092350022, 0.0020135986244202821, 0.73838910877870134, 0.26044982141749318, 0.0011610698038053676, 0.84383025163554393, 0.15567813415622619, 0.00049161420822986111, 0.89605580056509737, 0.10369808792428638, 0.00024611151061605928, 0.94064348769367923, 0.059228702074037368, 0.00012781023228344892, 0.97198351665198623, 0.02796355948371226, 5.2923864301402124e-05, 0.98155001689620114, 0.018424084485191394, 2.589861860750302e-05, 0.98846266666253302, 0.01152586885723272, 1.1464480234292309e-05, 0.98167521430374161, 0.018316614933378158, 8.1707628801724633e-06, 0.99179140859534576, 0.008205852149880663, 2.7392547735057506e-06, 0.99230972099150416, 0.0076883440461310909, 1.9349623647716529e-06, 9.9365586388011134e-10, 0.0370197924262224, 0.96298020658012173, 3.263280530733794e-09, 0.047037725201323412, 0.95296227153539603, 9.9735532843536177e-09, 0.056541359702757379, 0.94345863032368926, 1.3430512891640197e-08, 0.059870119660389826, 0.94012986690909728, 5.3626335247782195e-08, 0.049400900691735312, 0.95059904568192943, 1.8251676122974967e-07, 0.12256942989678073, 0.87743038758645808, 5.2922450677873055e-07, 0.076875486968713325, 0.92312398380677985, 1.8404048442577517e-06, 0.12432874015114775, 0.875669419444008, 6.2614999735686165e-06, 0.14538888263629424, 0.85460485586373214, 2.1023550747395797e-05, 0.18484238552335236, 0.81513659092590041, 0.0063906682531892755, 0.22131768840776816, 0.77229164333904254, 0.00017232974892336268, 0.31131130515430244, 0.68851636509677416, 0.0084904521815259079, 0.49659432357304756, 0.49491522424542644, 0.013467487739868876, 0.507855321126509, 0.47867719113362212, 0.011177896815878621, 0.67098380378005851, 0.31783829940406289, 0.018719412343690768, 0.71070604571793838, 0.27057454193837094, 0.029818801988092305, 0.81987452935681904, 0.15030666865508854, 0.043634440046455995, 0.85188345323689274, 0.10448210671665126, 0.071265570563405545, 0.86336657885860391, 0.065367850577990577, 0.090578007986616166, 0.8572150282070562, 0.052206963806327632, 0.17442474222818516, 0.76673152015876533, 0.058843737613049507, 0.33994821113117157, 0.64376032368387026, 0.016291465184958123, 0.41266805186909161, 0.56313808378591601, 0.024193864344992419, 0.56348149705322992, 0.43025977856840303, 0.00625872437836712, 0.68925181451467832, 0.3082279318432869, 0.0025202536420348394, 0.80676473084094369, 0.19209395330517381, 0.0011413158538825963, 0.88637676130418119, 0.1130178461429087, 0.00060539255291009798, 0.927608488602367, 0.072146921450505927, 0.00024458994712709065, 0.93409735164382557, 0.065778820831101786, 0.00012382752507264516, 0.95955613130039441, 0.040384600727647969, 5.9267971957485057e-05, 0.97011376253539017, 0.029863451160091054, 2.2786304518811963e-05, 0.97785636843210832, 0.022133673016754722, 9.9585511369828196e-06, 0.98883089378959566, 0.011162923574077121, 6.1826363272136429e-06, 0.99132115144217314, 0.0086752529754306317, 3.5955823961140894e-06, 0.99391937637817918, 0.0060787241602671032, 1.8994615537603742e-06, 2.1883111168533595e-10, 0.097685303159883413, 0.90231469662128561, 1.9619342309380787e-09, 0.024935516727982106, 0.97506448131008361, 3.5286318780874051e-09, 0.076875527381861097, 0.92312446908950707, 1.1174111479027537e-08, 0.021193277834474996, 0.97880671099141348, 3.9014624280688761e-08, 0.027700502337816992, 0.97229945864755873, 1.9855753294600787e-07, 0.056732691047394423, 0.94326711039507272, 5.5649301216555306e-07, 0.094581880780753844, 0.90541756272623397, 2.1048608818736715e-06, 0.10587348637527356, 0.89412440876384458, 6.3394180477664409e-06, 0.1480302505364014, 0.85196341004555076, 1.7495901445102562e-05, 0.17312172990195959, 0.82686077419659532, 0.0049196783937341055, 0.2536698512655457, 0.74141047034072016, 0.0045293250465165205, 0.34508495218051799, 0.65038572277296547, 0.00048661010877909847, 0.45371542150162869, 0.54579796838959216, 0.0071844113231145897, 0.46167355915152009, 0.53114202952536527, 0.005889581720519464, 0.60056175055146765, 0.39354866772801289, 0.010782119472985954, 0.70945947440517587, 0.27975840612183822, 0.01642973426724444, 0.82006011621068664, 0.16351014952206883, 0.041788042636438386, 0.85815102554979483, 0.10006093181376677, 0.06441789360108971, 0.85876918013974235, 0.076812926259168024, 0.1083871738371005, 0.82396050886044858, 0.067652317302450873, 0.24327610456501764, 0.71505255245120514, 0.041671342983777108, 0.33455571864246691, 0.62408368863533703, 0.041360592722195916, 0.4376036981926188, 0.55108443115460148, 0.0113118706527797, 0.61343631730047987, 0.36718281992332857, 0.019380862776191631, 0.74969640877153532, 0.24368332139839324, 0.0066202698300714558, 0.85119990263626566, 0.14753869354684521, 0.0012614038168891181, 0.91305101156710244, 0.086394045069231262, 0.00055494336366611167, 0.94159826524719181, 0.058182865450185708, 0.00021886930262248174, 0.96696107098195949, 0.032927434102496211, 0.00011149491554430661, 0.95834583136945961, 0.041604247394788012, 4.9921235752383312e-05, 0.96462998331574712, 0.035350763692445421, 1.9252991807457744e-05, 0.98934203780505781, 0.010648728623461537, 9.2335714806298587e-06, 0.98241227853886326, 0.017582582526883196, 5.1389342535582088e-06, 0.9931869095510869, 0.0068095909833367435, 3.4994655764008678e-06, 0.99244970892018214, 0.0075473668794043521, 2.9242004134331896e-06, 3.339859439043867e-10, 0.0096885070803255629, 0.9903114925856884, 6.6121992840305402e-10, 0.039974285964173864, 0.96002571337460618, 5.7939805196932679e-09, 0.056732701983391663, 0.94326729222262784, 1.293760096272363e-08, 0.022056853662269983, 0.97794313340012906, 3.0749706047563064e-08, 0.0392496563128246, 0.96075031293746938, 1.2920621726186205e-07, 0.04977673026423924, 0.95022314052954349, 4.0424534540435191e-07, 0.056144245701243024, 0.94385535005341159, 1.3477013411494762e-06, 0.091401519396331718, 0.90859713290232713, 4.0981074768474016e-06, 0.089326336031237741, 0.91066956586128545, 1.8237107152639052e-05, 0.1740783281776366, 0.82590343471521077, 0.0038803579363370817, 0.18514876347033832, 0.81097087859332462, 0.00013533082420921915, 0.31868429584681496, 0.68118037332897585, 0.0065781577655604339, 0.39487189693243979, 0.5985499453019999, 0.012532913921255872, 0.50637085192594344, 0.48109623415280062, 0.01536184158953059, 0.53909519798637417, 0.4455429604240953, 0.0051958393538176079, 0.66977335234010082, 0.32503080830608172, 0.015235236726309416, 0.78053866575827613, 0.2042260975154144, 0.038499342715402084, 0.82466179329259126, 0.1368388639920067, 0.08425963752955834, 0.80963694495948102, 0.10610341751096065, 0.12191915369695405, 0.79141298491078516, 0.086667861392260737, 0.22601809958995775, 0.73642314957685306, 0.037558750833189151, 0.32756897296451915, 0.64823811415012966, 0.02419291288535114, 0.44785970630845934, 0.5258832162877094, 0.026257077403831305, 0.61337403998392526, 0.38035221696858235, 0.0062737430474923347, 0.79559588376442836, 0.20095333978995925, 0.0034507764456123725, 0.84284029889699708, 0.15540294105933913, 0.0017567600436638351, 0.93424840284558153, 0.065115603521040649, 0.00063599363337777368, 0.94134185197668685, 0.058422887925821496, 0.00023526009749169207, 0.9666607629679429, 0.033233937014115193, 0.00010530001794187947, 0.98350184818462183, 0.016454121011921057, 4.403080345709061e-05, 0.98772145143229284, 0.012260272830276267, 1.8275737430953548e-05, 0.99106996043374562, 0.0089206302010749125, 9.4093651794501396e-06, 0.99271879949982789, 0.0072760267255096494, 5.1737746624565255e-06, 0.99500086432646317, 0.0049959565442079202, 3.1791293289832205e-06, 0.99413373450184239, 0.0058634524709917264, 2.8130271657988598e-06, 1.4890074595651799e-10, 0.056732702303652287, 0.94326729754744687, 5.5164699221405322e-10, 0.0054754415857801572, 0.99452455786257288, 1.6855214298597994e-09, 0.015671207920404839, 0.98432879039407373, 6.3441770089935905e-09, 0.051350758486430906, 0.94864923516939204, 2.4237545120856208e-08, 0.036720221326562538, 0.96327975443589242, 9.9809191992937716e-08, 0.066937067907946388, 0.93306283228286169, 3.5070831958991804e-07, 0.060125873813705412, 0.93987377547797502, 1.4107925778350402e-06, 0.06489431342917816, 0.93510427577824407, 0.0042461469762375631, 0.11437218166870533, 0.88138167135505718, 1.469222381146675e-05, 0.17250839273092008, 0.82747691504526844, 4.4989382244150631e-05, 0.17401496905303238, 0.82594004156472345, 0.00012918933912872213, 0.27215846728718202, 0.7277123433736894, 0.0031919472999223496, 0.36601536672116297, 0.63079268597891469, 0.0088518902292733757, 0.39965455862973115, 0.59149355114099544, 0.0086857091673266263, 0.53029596667999723, 0.46101832415267618, 0.0081001904119084676, 0.62337957179697701, 0.36852023779111459, 0.01796662965674432, 0.72294789834748074, 0.25908547199577486, 0.03036864014526687, 0.77337723160873328, 0.19625412824599966, 0.045280776900217222, 0.80584766031253818, 0.14887156278724464, 0.1079457271478957, 0.8020894941362412, 0.089964778715863017, 0.23833738414678105, 0.71604810718646128, 0.045614508666757772, 0.32946417908479775, 0.64924454644743579, 0.021291274467766488, 0.53905579612338017, 0.44198968296084262, 0.018954520915777207, 0.59742330889998618, 0.38780499192689893, 0.014771699173114914, 0.79456694214993939, 0.19966950131706829, 0.0057635565329923139, 0.88375940901024241, 0.11098341778069606, 0.0052571732090615081, 0.903451493353706, 0.0959012287837505, 0.00064727786254343048, 0.96367159898140298, 0.036067038447164058, 0.00026136257143289087, 0.98275802786434341, 0.017140331617144153, 0.00010164051851241006, 0.98577611361797401, 0.014180183366550192, 4.3703015475690965e-05, 0.98791129714047454, 0.012067984414869278, 2.0718444656275322e-05, 0.99412999128844415, 0.0058617103010831116, 8.2984104727447807e-06, 0.99377704428800751, 0.0062178635519437699, 5.0921600487546242e-06, 0.99507421822732978, 0.0049219087753953842, 3.8729972747183382e-06, 0.99389431209247481, 0.006102067798638167, 3.6201088870785132e-06, 1.3436663246182434e-10, 0.051350758805309378, 0.94864924106032411, 3.0245333555268103e-10, 0.034830021882150085, 0.9651699778153966, 1.6293431669884804e-09, 0.045584883390177723, 0.95441511498047915, 5.7935011524760405e-09, 0.0094072243189571164, 0.99059276988754164, 2.2920191518573447e-08, 0.034830021094373775, 0.96516995598543476, 7.1552742834805665e-08, 0.021878552531952946, 0.97812137591530424, 2.4066169516794339e-07, 0.041384826584195365, 0.95861493275410947, 9.1208389540814795e-07, 0.098191497692471347, 0.90180759022363333, 3.7366780764952134e-06, 0.097684938162729204, 0.90231132515919432, 1.3322018292023556e-05, 0.11010237579519749, 0.88988430218651049, 4.4816354764194115e-05, 0.18457279033563193, 0.81538239330960383, 0.00012953164247918807, 0.22809125560451687, 0.77177921275300398, 0.00043620745611727978, 0.3562993175278219, 0.64326447501606077, 0.0030147186049377028, 0.37121379078132843, 0.62577149061373394, 0.0031459523393467916, 0.54958558651158618, 0.4472684611490671, 0.0062650069699376719, 0.50384295133666768, 0.48989204169339479, 0.016546717599393645, 0.71056431338426873, 0.27288896901633775, 0.053285192434476346, 0.63106070864449748, 0.3156540989210263, 0.065835201188771772, 0.79376529836335186, 0.14039950044787641, 0.13616974495645434, 0.81543121262958773, 0.048399042413957975, 0.2175380165935713, 0.70686029889198354, 0.075601684514445128, 0.38976360615743039, 0.56761044406685002, 0.042625949775719561, 0.48429319255322578, 0.49849349271620019, 0.017213314730574213, 0.65577844118514184, 0.33645207752045642, 0.0077694812944017625, 0.81328991779179638, 0.18213927858367787, 0.0045708036245257289, 0.90155922851848225, 0.096272735785017216, 0.0021680356965005066, 0.91640021854328324, 0.082813337815603849, 0.00078644364111294686, 0.9700403361305322, 0.029691524984771501, 0.0002681388846962445, 0.98137240893801769, 0.018521136275595661, 0.00010645478638659134, 0.97974481001879721, 0.020214428991367224, 4.07609898355343e-05, 0.98917056485208699, 0.0108132405303822, 1.6194617530872471e-05, 0.98619159540055856, 0.013799264712834254, 9.1398866072316556e-06, 0.99671927679643746, 0.0032764685165851859, 4.2546869772530814e-06, 0.99595685930891742, 0.004039215597667763, 3.9250934147206801e-06, 0.96762031774846635, 0.032376841652539595, 2.840598993960128e-06, 7.2097396335411564e-11, 0.029194492376581511, 0.97080550755132111, 4.0179039697768676e-10, 0.049025365025077261, 0.95097463457313236, 9.5066065405223261e-10, 0.028181215348281424, 0.97181878370105801, 3.7831465448186402e-09, 0.02603511334043445, 0.97396488287641891, 1.2897453100992321e-08, 0.033226557647600172, 0.96677342945494671, 5.5540205241171822e-08, 0.035987832474261745, 0.96401211198553305, 2.4998164072527338e-07, 0.042044199734704649, 0.95795555028365453, 8.1670004618803465e-07, 0.055895756937254425, 0.94410342636269939, 3.5557183280460435e-06, 0.095593906464581158, 0.90440253781709079, 1.185475439547221e-05, 0.098620850863316845, 0.9013672943822878, 0.0032133768311381028, 0.16074310494565341, 0.83604351822320844, 0.0032911711794620145, 0.21529142647254981, 0.78141740234798807, 0.0067920595635109242, 0.29794333851398391, 0.69526460192250528, 0.0034341497501990908, 0.39114545017058466, 0.60542040007921616, 0.0034709908480684542, 0.44876614745630133, 0.54776286169563015, 0.0038885669818436775, 0.59253241363734921, 0.40357901938080715, 0.026798279846251247, 0.6874529842694449, 0.28574873588430377, 0.046865210150335936, 0.74658326805472752, 0.20655152179493647, 0.050973212054560915, 0.73820502975669444, 0.21082175818874463, 0.11582050725305687, 0.7975136432918819, 0.086665849455061283, 0.23845744267710173, 0.73405434168222672, 0.027488215640671592, 0.38642095964740025, 0.58414770015039585, 0.02943134020220398, 0.59032075382646843, 0.39956750147623415, 0.010111744697297306, 0.7188743196947609, 0.27129834517111084, 0.009827335134128222, 0.8880468278643705, 0.10514287521247102, 0.006810296923158439, 0.89523876139976111, 0.10220887125492693, 0.0025523673453119598, 0.93299311307193045, 0.066093774773800032, 0.00091311215426958592, 0.96948075560435842, 0.030215852435740143, 0.00030339195990135022, 0.98743151006175234, 0.012457602960085305, 0.00011088697816232922, 0.98863307660542143, 0.011327620194609483, 3.9303199969077409e-05, 0.99192771052491746, 0.0080567176245565537, 1.5571850525886394e-05, 0.9915449244966672, 0.0084473735206991758, 7.701982633683089e-06, 0.98896457303458574, 0.01103034209933974, 5.0848660745897837e-06, 0.99739916086639624, 0.0025977259776725765, 3.113155931097272e-06, 0.9959521110731242, 0.0040442348735576072, 3.6540533181947174e-06, 4.4199695987302964e-11, 0.00079135070310652029, 0.9992086492526937, 2.2304041455284082e-10, 0.015230278483834668, 0.9847697212931249, 6.8854433767571221e-10, 0.015230278476744913, 0.9847697208347107, 2.8192807958977872e-09, 0.027144862577934682, 0.9728551346027845, 1.0778809988163945e-08, 0.031080299962911805, 0.96891968925827832, 4.8193427306082821e-08, 0.031456581535049252, 0.96854337027152348, 1.807015419342082e-07, 0.031182021533312645, 0.96881779776514543, 7.5681741952125855e-07, 0.071778508130552518, 0.92822073505202796, 2.5644348903010699e-06, 0.084181389562409448, 0.91581604600270017, 1.0818550821281452e-05, 0.10868723168293189, 0.89130194976624688, 0.0035761069920280547, 0.13210190469280431, 0.8643219883151676, 0.00012820012897510902, 0.20853946141899798, 0.791332338452027, 0.00039227128709937755, 0.27635328768831097, 0.72325444102458969, 0.0040385455059335235, 0.33566501172958918, 0.66029644276447741, 0.0036193142363509698, 0.45968037923984117, 0.53670030652380785, 0.019002973720688467, 0.5081988933496111, 0.4727981329297003, 0.016204891781588775, 0.71500761241990485, 0.26878749579850642, 0.042688688205912047, 0.71454471112403395, 0.2427666006700539, 0.1382598686175068, 0.77437668498850887, 0.087363446393984384, 0.12441612787888208, 0.73407528189227178, 0.14150859022884615, 0.31785277970529957, 0.61153787636852763, 0.070609343926172824, 0.45274998924439597, 0.49360944632781095, 0.053640564427793115, 0.55663135865760371, 0.40947694225532671, 0.033891699087069516, 0.79868554174638928, 0.18237089641179172, 0.018943561841819025, 0.84486569202216188, 0.14752370666445122, 0.0076106013133868094, 0.91693074199889102, 0.080284369619351315, 0.0027848883817575515, 0.93265969205335908, 0.066400902707302881, 0.00093940523933805028, 0.95821883063149993, 0.041428769783705847, 0.00035239958479424047, 0.9774688774728677, 0.022424697623625742, 0.00010642490350651342, 0.98300434716428664, 0.016952531015567419, 4.3121820145886123e-05, 0.98810034637639721, 0.01188169543503511, 1.7958188567716007e-05, 0.98737955654446385, 0.012612013242308748, 8.430213227496385e-06, 0.99709656907768007, 0.0028977155863713871, 5.7153359486050302e-06, 0.98712043664391691, 0.012875836995233881, 3.7263608491763499e-06, 0.9970893397129047, 0.0029074249700652114, 3.2353170301639181e-06, 4.3888955063189543e-11, 0.00069907988458812315, 0.99930092007152294, 1.1754043539367822e-10, 0.000724357199774671, 0.99927564268268498, 6.1289453853422841e-10, 0.0080144137737966602, 0.99198558561330874, 2.4333146707723345e-09, 0.02770050335113769, 0.97229949421554762, 8.8905864030767153e-09, 0.034098665837255612, 0.965901325272158, 4.0007630845511952e-08, 0.024013869669660209, 0.975986090322709, 1.7184897608261227e-07, 0.063748025836775649, 0.93625180231424832, 6.9712268733252152e-07, 0.039086123597828377, 0.96091317927948428, 2.5831802294727531e-06, 0.05848385440586986, 0.94151356241390061, 1.0526998699058674e-05, 0.10979251231594027, 0.89019696068536069, 3.4459274098236631e-05, 0.15284915124105034, 0.84711638948485146, 0.0044856204022698971, 0.21748054175923309, 0.77803383783849689, 0.00038571747310749794, 0.25769559343018633, 0.74191868909670622, 0.011963826053637832, 0.40972025970841741, 0.57831591423794482, 0.011437298748728635, 0.43569843754256626, 0.55286426370870512, 0.0089947583370318662, 0.6184467122975541, 0.37255852936541406, 0.035309177334161806, 0.60325080739663639, 0.36144001526920166, 0.040297966404177196, 0.63886293029668306, 0.32083910329913978, 0.10125893136242717, 0.71551602800859582, 0.18322504062897702, 0.223448691256009, 0.58317905835789197, 0.19337225038609901, 0.34548155099428157, 0.58009718165983104, 0.07442126734588736, 0.43013936485315829, 0.53680438414769005, 0.033056250999151639, 0.61905110162159371, 0.33782031038761423, 0.043128587990792062, 0.75968307867829277, 0.21924063889519718, 0.021076282426509937, 0.85766715474075783, 0.13445440720263788, 0.0078784380566042384, 0.90288678811449841, 0.093749511328208177, 0.0033637005572933125, 0.96811646495403314, 0.030678495898637168, 0.0012050391473296292, 0.95089363517350589, 0.048786315463027113, 0.0003200493634669237, 0.97573817535568441, 0.024145168815375446, 0.00011665582894015838, 0.99367518240783481, 0.0062853602086005173, 3.9457383564631756e-05, 0.99114421679596987, 0.0088351856626138828, 2.0597541416095451e-05, 0.98326681111255509, 0.016722843872560747, 1.03450148842026e-05, 0.99220181813099639, 0.0077916831060220303, 6.4987629815080154e-06, 0.98241270058943386, 0.017582590080472613, 4.7093300936084676e-06, 0.99712459543069476, 0.0028714745167780892, 3.9300525271690069e-06, 2.7378520631881889e-11, 0.018323631138197263, 0.98167636883442433, 1.2052212251421604e-10, 0.012152853843749001, 0.98784714603572887, 4.7293100527401347e-10, 0.015447597425960255, 0.98455240210110873, 2.3770398395229041e-09, 0.013518639561767774, 0.9864813580611923, 7.8637346462345538e-09, 0.033483469953700086, 0.96651652218256523, 3.5647561754668079e-08, 0.026723682033058294, 0.9732762823193799, 1.4844428637083042e-07, 0.034387486461628987, 0.96561236509408466, 6.0889521621561119e-07, 0.042638615750026365, 0.95736077535475739, 2.1224683916353647e-06, 0.097169346848809554, 0.90282853068279878, 9.4610359789014307e-06, 0.051350272980832536, 0.94864026598318862, 0.0049279203459382456, 0.14032102283869916, 0.85475105681536268, 0.00012291526472504904, 0.15959566862730168, 0.84028141610797324, 0.00043392494435007972, 0.23511412741233956, 0.76445194764331037, 0.011874908686974274, 0.33813438856531086, 0.64999070274771487, 0.0072754448510199445, 0.40313396290282921, 0.58959059224615096, 0.021887809915287919, 0.58394471380190371, 0.39416747628280835, 0.032295125140880981, 0.49707772923103155, 0.47062714562808733, 0.074250520848829632, 0.66937988124340253, 0.25636959790776775, 0.1099557710066984, 0.68905229273932522, 0.20099193625397641, 0.19159918058635889, 0.71760025897447011, 0.090800560439171021, 0.42917814742742499, 0.55048271740748989, 0.020339135165085273, 0.51832997904470879, 0.4593390027965169, 0.02233101815877454, 0.57265786634794857, 0.40591679127907632, 0.021425342372975154, 0.82240898436026277, 0.15277251440782993, 0.024818501231907313, 0.86864798604493454, 0.12154538284049969, 0.0098066311145658461, 0.93836792759720689, 0.057772409501247034, 0.0038596629015461925, 0.96660059357272565, 0.032110039952553335, 0.0012893664747210358, 0.96312795810933927, 0.036452955712746496, 0.00041908617791427548, 0.97942566613234527, 0.020450651737463313, 0.00012368213019137324, 0.9917480209001196, 0.0082068355725797709, 4.5143527300700856e-05, 0.99655519962274208, 0.0034238180679699425, 2.0982309288017537e-05, 0.99722361082503641, 0.0027652958623038217, 1.1093312659843271e-05, 0.99787731128663626, 0.0021163825452772747, 6.3061680865358122e-06, 0.99775001773541105, 0.0022449764528887466, 5.0058117001726669e-06, 0.9934580702019119, 0.0065309506133623433, 1.0979184725732311e-05, 1.7511407197797245e-11, 0.00022175643672111715, 0.9997782435457675, 6.9504524600977934e-11, 0.009246530576511024, 0.99075346935398434, 4.7326191949812714e-10, 0.020394776296029635, 0.97960522323070853, 1.6481905805489844e-09, 0.006183412077557863, 0.99381658627425162, 7.4661109877518518e-09, 0.031456582816191458, 0.96854340971769759, 3.3062905832505243e-08, 0.042044208854881547, 0.95795575808221256, 1.6292748863877625e-07, 0.024897488666784786, 0.97510234840572663, 5.5029117087644341e-07, 0.039107851461809104, 0.96089159824701997, 2.7196647548860221e-06, 0.062809030530917118, 0.93718824980432802, 1.1264414578094582e-05, 0.086039056972500028, 0.91394967861292198, 3.2859306410116049e-05, 0.10422421809251206, 0.8957429226010778, 0.0075239783960994455, 0.17950087585610025, 0.81297514574780028, 0.00041462532727106041, 0.25490163266672039, 0.7446837420060084, 0.0097942997080261596, 0.29396480311563605, 0.69624089717633775, 0.020116220796997084, 0.37928862346267783, 0.60059515574032507, 0.029684742810971144, 0.54827932642773602, 0.42203593076129275, 0.045373111520344275, 0.67509149981043781, 0.27953538866921779, 0.056440911995126582, 0.62258486931532042, 0.32097421868955284, 0.067054063161592389, 0.77405834410019403, 0.15888759273821346, 0.16059537517490682, 0.74155202126060504, 0.097852603564488147, 0.27688185792734227, 0.59190136025604712, 0.13121678181661051, 0.4881873237408641, 0.45397359461573883, 0.057839081643396968, 0.7289080071501608, 0.2445199924432194, 0.026572000406619858, 0.81789665319636884, 0.1573605185546515, 0.024742828248979696, 0.78582066612439938, 0.20158561113891454, 0.012593722736686163, 0.96249124557469046, 0.032678827686928716, 0.004829926738380951, 0.95445948645458389, 0.04407233773830041, 0.0014681758071157095, 0.97946770234799052, 0.02010093181615474, 0.00043136583585478948, 0.98130479288638905, 0.018532484487115006, 0.00016272262649596838, 0.98162352424793353, 0.018313794767152655, 6.2680984913820494e-05, 0.98575174919864261, 0.014224144867249884, 2.4105934107445664e-05, 0.99652195739581984, 0.0034610116781912952, 1.70309259889367e-05, 0.97276885264424784, 0.027222894546339853, 8.2528094122116463e-06, 0.99695740758510432, 0.0030342932228727597, 8.2991920229771441e-06, 0.99446164937779757, 0.0055269535040442571, 1.1397118157986199e-05, 1.4187068837724371e-11, 0.00016054974935304818, 0.99983945023645981, 5.715327212215367e-11, 0.021193278070079785, 0.97880672187276696, 3.2999678230222699e-10, 0.029728955965103266, 0.97027104370490003, 1.5979272319802827e-09, 0.058484005387143365, 0.94151599301492939, 7.1179770126793832e-09, 0.034830021644765251, 0.96516997123725778, 2.4371933812835587e-08, 0.033594794856526007, 0.96640518077154014, 1.1942460612760051e-07, 0.050868191962605856, 0.94913168861278807, 5.884704560091193e-07, 0.052456570500484724, 0.94754284102905928, 2.8410714601984615e-06, 0.077374795730142401, 0.92262236319839741, 9.3211675545631986e-06, 0.11782901767715663, 0.88216166115528882, 3.9345881457697885e-05, 0.10586954353028988, 0.89409111058825241, 0.0001405174272693304, 0.20967228324452339, 0.79018719932820725, 0.00054998713503999442, 0.2301343614117039, 0.7693156514532562, 0.0012577130436678496, 0.35075699996733722, 0.64798528698899494, 0.028674514115622787, 0.44135027504055413, 0.52997521084382315, 0.0097800513521887865, 0.51474881366893599, 0.47547113497887511, 0.032067924665140454, 0.6052776622641477, 0.36265441307071189, 0.0950402629788107, 0.52662013986132783, 0.37833959715986143, 0.058359927550077247, 0.65123852594302745, 0.29040154650689526, 0.30676510578518651, 0.511511494855586, 0.18172339935922752, 0.39807812005322452, 0.47131611518985894, 0.13060576475691657, 0.69469774016045083, 0.26731493814817442, 0.037987321691374823, 0.61569958793272406, 0.3230686221530723, 0.061231789914203603, 0.72417678120376272, 0.24150383528473066, 0.034319383511506589, 0.91453093874707181, 0.072185692130575421, 0.01328336912235286, 0.94061063280727397, 0.054632533887668429, 0.0047568333050575827, 0.95769458075676495, 0.040201562542849685, 0.0021038567003853945, 0.9627929884309181, 0.036590237600538186, 0.00061677396854360668, 0.96993609453164731, 0.029857981925494666, 0.00020592354285794284, 0.99409111054581212, 0.0058435971952659543, 6.5292258921788846e-05, 0.99566306061244092, 0.0043017670863138048, 3.5172301245249531e-05, 0.98665977709230013, 0.01332140114790183, 1.882175979812223e-05, 0.99726662916713416, 0.0027211872188618036, 1.2183614004143976e-05, 0.93972420864926431, 0.060266574508937097, 9.216841798684859e-06, 0.98591063450261096, 0.014053864888318803, 3.5500609070171877e-05, 1.5722910771986251e-11, 0.00015924225241121373, 0.99984075773186587, 5.4960143058987886e-11, 0.030003593882343266, 0.96999640606269655, 2.6307862914140925e-10, 0.011630614333934525, 0.9883693854029868, 1.2982856773412406e-09, 0.039974285938707624, 0.96002571276300674, 5.4748242528577134e-09, 0.023663936054151687, 0.976336058471024, 3.1316150502647088e-08, 0.027235915413429776, 0.97276405327041982, 1.1321209554098173e-07, 0.042595620639255725, 0.95740426614864882, 5.9795690136218268e-07, 0.017438173155925336, 0.98256122888717334, 2.4649363394370085e-06, 0.053907602548449743, 0.94608993251521079, 7.216653795107126e-06, 0.098960216184005897, 0.90103256716219904, 4.5191500880007834e-05, 0.10229605010773089, 0.89765875839138898, 0.00010975898091841597, 0.14466735403110442, 0.85522288698797722, 0.00038007082176860416, 0.27872926403735859, 0.72089066514087285, 0.0011900061186975064, 0.35078077855185269, 0.64802921532944968, 0.02805540116460423, 0.45341211080212007, 0.51853248803327578, 0.010764024114413793, 0.45329345171126378, 0.53594252417432242, 0.065488732167637911, 0.65519001976821012, 0.27932124806415198, 0.072212199387054393, 0.72245576876949025, 0.20533203184345539, 0.13561030828027243, 0.80012275719790971, 0.06426693452181792, 0.38372846175915604, 0.47988228303541325, 0.13638925520543058, 0.43719277419040659, 0.33645757754889161, 0.22634964826070161, 0.62347075615045644, 0.28788873833801987, 0.088640505511523623, 0.58109227420307807, 0.351371950279949, 0.067535775516973026, 0.79400292795233607, 0.18105297486625949, 0.024944097181404411, 0.9139607174946488, 0.072762585910215269, 0.013276696595135992, 0.95135494406111376, 0.043067632291405651, 0.0055774236474804657, 0.92596556385245343, 0.071261042931099375, 0.0027733932164471053, 0.97306551752294235, 0.026275716114251176, 0.00065876636280652915, 0.98640080969260957, 0.013347995496412926, 0.00025119481097760158, 0.99161869525162583, 0.0082682879375069861, 0.00011301681086729397, 0.98789635835861556, 0.01206780192785129, 3.5839713533250897e-05, 0.99694085974943014, 0.0030367468212366923, 2.2393429333304457e-05, 0.99800256657017805, 0.0019865525553096376, 1.0880874512223259e-05, 0.99821633293335454, 0.0017763860429823425, 7.2810236631001788e-06, 0.99826454295271017, 0.0017301106657655668, 5.3463815241317353e-06, 1.4776740137602711e-11, 0.00013414022858605908, 0.99986585975663733, 4.6300513708786933e-11, 0.01965426030242572, 0.9803457396512737, 2.6234330573059661e-10, 0.00036238377872868313, 0.99963761595892808, 1.613504112766891e-09, 0.00089109455206720872, 0.99910890383442863, 6.1096903584786802e-09, 0.013689582780224219, 0.98631041111008544, 3.806569167025583e-08, 0.017161748835659331, 0.98283821309864894, 1.0076978395624472e-07, 0.019654258322780158, 0.98034564090743592, 4.9559954932050257e-07, 0.044953909537167498, 0.95504559486328322, 2.2987489996242469e-06, 0.013030469946320946, 0.98696723130467956, 7.397591693764927e-06, 0.1464892998856431, 0.85350330252266304, 4.3292944525184965e-05, 0.19050426515518376, 0.80945244190029109, 0.019035109959053019, 0.08789496116983575, 0.89306992887111114, 0.00064689527722181612, 0.23608992967694434, 0.76326317504583385, 0.048395461253001142, 0.29795587884907565, 0.6536486598979232, 0.0046861036728838915, 0.50318430900855238, 0.49212958731856371, 0.01122471857926274, 0.49987869011216157, 0.4888965913085756, 0.12382094198014931, 0.52409996588084928, 0.3520790921390014, 0.1605005119790715, 0.53524899187843811, 0.30425049614249033, 0.20965967811493125, 0.59162113930744409, 0.19871918257762458, 0.29276594099134395, 0.54074079047904344, 0.16649326852961249, 0.49837013587232309, 0.38353883809910699, 0.11809102602856991, 0.69088569763685714, 0.21267847220991304, 0.096435830153229818, 0.63999631213235397, 0.24626620284270148, 0.11373748502494455, 0.72830635497980567, 0.16814838193316983, 0.10354526308702455, 0.92298165842769053, 0.064574005138227406, 0.012444336434082095, 0.95846926195585125, 0.035124991936709524, 0.0064057461074392376, 0.96505439295829154, 0.032290985104160036, 0.0026546219375483543, 0.97242160835875646, 0.026726080921534993, 0.00085231071970847565, 0.98862713876212316, 0.011117265350910083, 0.00025559588696679353, 0.99229982509509329, 0.0075737021569505401, 0.00012647274795616972, 0.97319556615527103, 0.026748499746583349, 5.593409814555021e-05, 0.99710974235865635, 0.002864451988807614, 2.5805652536018087e-05, 0.99759973342109121, 0.0023843118993613281, 1.5954679547526123e-05, 0.9969172245390302, 0.0030674155400210357, 1.5359920948821713e-05, 0.99874827087156437, 0.0012470212936870357, 4.7078347486403233e-06, 9.7450028416373407e-12, 7.9407963503867098e-05, 0.9999205920267511, 5.543497048108002e-11, 0.00017476716041905143, 0.99982523278414592, 1.6522628019573371e-10, 0.00020487056732963453, 0.99979512926744407, 1.1004130796729861e-09, 0.00054552057801935685, 0.99945447832156764, 4.3147419317708209e-09, 0.031764233305951857, 0.9682357623793062, 4.4183598096215622e-08, 0.032724381013353274, 0.96727557480304871, 1.0213216506373892e-07, 0.032724379117022202, 0.96727551875081275, 4.6748150228075886e-07, 0.034830005610293577, 0.96516952690820412, 2.8944680850101307e-06, 0.053907579393366038, 0.94608952613854891, 6.8401858457495187e-06, 0.051350407563475607, 0.94864275225067862, 4.2288229609097081e-05, 0.15284795455060832, 0.84710975721978254, 0.045890069806887993, 0.10594910963041623, 0.84816082056269582, 0.0004039828679425948, 0.27248430824822867, 0.7271117088838287, 0.0014317249600511064, 0.50059847338920349, 0.49796980165074534, 0.0031682920577561806, 0.32875625853129753, 0.66807544941094632, 0.14007213011769412, 0.39525784810192865, 0.46467002178037708, 0.060335162020955097, 0.5107642673509557, 0.42890057062808923, 0.068901196818126897, 0.69698801831644763, 0.23411078486542539, 0.12448441105338233, 0.47900749804456111, 0.39650809090205658, 0.33672714414871358, 0.32392575399099149, 0.33934710186029493, 0.63664263410065269, 0.22270520952530348, 0.14065215637404377, 0.27610128479574603, 0.53120942822384698, 0.19268928698040699, 0.74549517920020447, 0.19124096346480865, 0.063263857334986989, 0.87030326117595458, 0.10304207426413499, 0.026654664559910449, 0.93537072989748393, 0.051417751173010354, 0.013211518929505771, 0.90809120890068074, 0.085373083665776825, 0.0065357074335423357, 0.9565376945801991, 0.04122098491339371, 0.0022413205064071603, 0.95807364609933843, 0.040962242784464434, 0.00096411111619706068, 0.92099342804345152, 0.078753771290981531, 0.00025280066556699032, 0.992442566968611, 0.0074065361038368879, 0.00015089692755214875, 0.99572231550936152, 0.0042148468997504377, 6.2837590888111967e-05, 0.99134726004376938, 0.0085586688949944659, 9.4071061236363082e-05, 0.99802897803884261, 0.0019550608975151362, 1.5961063642100496e-05, 0.99327938162255047, 0.0066798092837178948, 4.0809093731748351e-05, 0.9986311122982684, 0.0013626115141493135, 6.2761875822913219e-06, 6.7388632162049671e-12, 4.9364762207106814e-05, 0.99995063523105399, 4.2433006481895921e-11, 0.051350758810030234, 0.94864924114753679, 1.7257304034797493e-10, 0.00019236307682332141, 0.99980763675060369, 7.5234801343775437e-10, 0.051350758773575561, 0.9486492404740765, 4.9803189007011353e-09, 0.00090208264868298982, 0.9990979123709981, 2.4009290881476715e-08, 0.030858820153509676, 0.9691411558371994, 1.2528099340963286e-07, 0.034830017529144795, 0.96516985718986181, 5.2063787722667725e-07, 0.067315410303138395, 0.93268406905898438, 1.4129672385283775e-06, 0.11416757860757935, 0.88583100842518214, 6.0682881578408014e-06, 0.026351816486843552, 0.97364211522499866, 3.7386115182993796e-05, 0.046899702306341141, 0.95306291157847589, 8.016968117097043e-05, 0.24985439330176415, 0.75006543701706485, 0.00034565980934394023, 0.58441035377936745, 0.41524398641128868, 0.0011304260154092413, 0.31656987858998814, 0.68229969539460267, 0.0035604164419432553, 0.64111987709890883, 0.35531970645914801, 0.0091238402813121691, 0.61324298895716323, 0.37763317076152447, 0.12806014139478231, 0.689873329307228, 0.18206652929798983, 0.046441352413898636, 0.37057007675116349, 0.58298857083493782, 0.29705346121765586, 0.30481103947455324, 0.39813549930779096, 0.32875701438496641, 0.25300690037021911, 0.41823608524481454, 0.58872081446699387, 0.26077382736263455, 0.1505053581703715, 0.8091949032455994, 0.088963618072826628, 0.10184147868157394, 0.8273979804226429, 0.10612569427653497, 0.066476325300822092, 0.79151970053582321, 0.17532851569116747, 0.033151783773009404, 0.90960582233468579, 0.077780021785032058, 0.012614155880282082, 0.92994619328513295, 0.065061260660114972, 0.0049925460547520645, 0.90948288095894547, 0.087490697758068184, 0.0030264212829863866, 0.97536224692630613, 0.023522644634195223, 0.001115108439498745, 0.95494297937251682, 0.043568029563507277, 0.001488991063975965, 0.99252839457658693, 0.0072906289943522121, 0.00018097642906096838, 0.99371392777951295, 0.0061739244113479192, 0.00011214780913929365, 0.996387276765903, 0.0035649814270718389, 4.7741807025164776e-05, 0.99591720423002128, 0.0040425840764449416, 4.0211693533666997e-05, 0.99824821565752853, 0.0017388410567994045, 1.2943285671922303e-05, 0.99858038976050756, 0.0014116878842659047, 7.9223552264642187e-06)

aa31dict={'CYS': 'C', 'GLN': 'Q', 'ILE': 'I', 'SER': 'S', 'VAL': 'V', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'LYS': 'K', 'THR': 'T', 'PHE': 'F', 'ALA': 'A', 'HIS': 'H', 'GLY': 'G', 'ASP': 'D', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'GLU': 'E', 'TYR': 'Y'}
aa3s=aa31dict.keys();aa3s.sort()#introduce ordering
aa1s=aa13dict.keys();aa1s.sort()
aa3s=['ALA','ARG','ASP' ,'ASN' ,'CYS' ,'GLU' ,'GLN' ,'GLY' ,'HIS' ,'ILE' ,'LEU' ,'LYS' ,'MET' ,'PHE' ,'PRO' ,'SER' ,'THR' ,'TRP' ,'TYR' ,'VAL']
aa1s3=[aa31dict[k] for k in aa3s]


class ShiftDict(dict):

    def __init__(self):
        dict.__init__(self)
        self.counts={'H':0,'C':0,'N':0,'P':0}#P is for DNA

    def writeTable(self,pdbID,pref='predictions_',col=0):
	if pref<>None:
	  pdbID=string.join(string.split(pdbID),'')##+'R'
	  fil=open(pref+pdbID+'.out','w')
	  fil.write(' NUM    RES      HA       H        N       CA        CB       C   \n')
	  fil.write(' ----  ------ -------- ------- -------- -------- -------- --------\n')
	else:fil=open('junk','w')
	ress=self.keys()
	ress.sort(lambda x,y: cmp(eval(x),eval(y)))
	##avedct =dict(zip(['HA','H','N','CA','CB','C'],[SubContainer() for i in range(6)]))
	lavedct=dict(zip(['HA','H','N','CA','CB','C'],[SubContainer() for i in range(6)]))
	for res in ress:
	  if len(self[res])>0:
	    resn=None
	    fil.write('  %3s  '%res)
	    fil.write('  %1s  '%self[res][self[res].keys()[0]][2])#generalize
	    for at in ['HA','H','N','CA','CB','C']:
	      if at in self[res]:
		vals=self[res][at];shift=vals[col]
		fil.write(' %8.4f'%shift)
		if shift>0.0:
		  ##avedct[at].update(shift)
		  lavedct[at].update(log(shift))
	      else:
		fil.write('   0.0000')
	    fil.write('\n')
	fil.close()
	##return avedct,lavedct
	return lavedct

		
class Parser:

    def __init__(self,buf):
        self.buffer=buf
        self.start=0
        self.numlines=len(buf)
        self.terminated=False

    def search(self,s='',s2=None,stopstr=None,stop=-1,numhits=1,changeStart=True,
               doReturn=True,positions=None,positions2=None,incr=1,verb=False):
        hits=0
        for k in range(self.start,len(self.buffer)):
            match=False
            if stopstr=='':
                if self.buffer[k]==[]:return None
            if s=='':#to nearest new-line
                if self.buffer[k]==[]:match=True
            elif s=='noneblank':
                if len(self.buffer[k])>0:
                    match=True
            if s<>'' or stopstr<>None:
                if positions==None:searchlist=range(len(self.buffer[k]))
                else:searchlist=positions
                for j in searchlist:
                  if len(self.buffer[k])>j:
                    if self.buffer[k][j]==s:
                        if verb:print self.buffer[k]
                        if s2==None:match=True
                        else:
                            if positions2==None:searchlist2=range(len(self.buffer[k]))
                            else:searchlist2=positions2
                            for j2 in searchlist2:
                                if self.buffer[k][j2]==s2:
                                    match=True
                    if self.buffer[k][j]==stopstr:return None
            if match:
                if verb:print 'matching'
                if verb:print self.buffer[k]
                if changeStart:self.start=k+incr
                hits+=1
                if hits==numhits:
                    if doReturn:return k+incr
                    break
        self.terminated=True

    def getExcerpt(self,end=None):
	if end==None:end=self.numlines
	return self.buffer[self.start:end]

    def getCurrent(self):
	return self.buffer[self.start]

    def findShiftData(self,ch,verb=True,skipCO=False):
        self.search(s="_Saveframe_category",s2='assigned_chemical_shifts',
                    positions=[0],positions2=[1])
	print 'finding NMR shifts for chain',ch
        if verb:print self.start
        li,mi=None,None
        while mi==None:
            li=self.search(s='loop_')
            mi=self.search(s='_Chem_shift_value',stopstr='')
            if verb:print li,mi
            if self.terminated:
                return ShiftDict()#empty
                print 'WARNING: no _Chem_shift_value entry found'
        pos=mi-li-1
        idict={}
        self.search()#nearest blank
        for i in range(self.start-li-1):
            if verb:print '**',i,self.buffer[li+i]
            key=self.buffer[li+i][0][1:]
            idict[key]=i
        if verb:print idict
        startI=self.search('noneblank',incr=0)
        endI=self.search(incr=0)
        return getShiftDBA(startI,endI,self.buffer,ch,idict,skipCO=skipCO)

    def findShiftData31(self,ch,verb=True):
        if verb:print self.start
        li,mi=None,None
        while mi==None:
            li=self.search(s='loop_')
            mi=self.search(s='_Atom_chem_shift.Val',stopstr='')
            if verb:print li,mi,self.start
        pos=mi-li-1
        idict={}
        self.search()#nearest blank
        for i in range(self.start-li-1):
            if verb:print '**',i,self.buffer[li+i]
            key=self.buffer[li+i][0][1:]
            idict[key]=i
        if verb:print idict
        startI=self.search('noneblank',incr=0)
        endI=self.search(incr=0)
	if verb:print startI,endI
        return getShiftDBA31(startI,endI,self.buffer,ch,idict)

    def findReference(self,verb=True):
        self.search(s="_Saveframe_category",s2='chemical_shift_reference',
                    positions=[0],positions2=[1])
	print 'finding reference'
        if verb:print self.start
        li,mi=None,None
        while mi==None:
            li=self.search(s='loop_')
            mi=self.search(s='_Mol_common_name',stopstr='')
            if verb:print li,mi
            if self.terminated:
                print 'WARNING: no _Mol_common_name entry found'
                return None
        pos=mi-li-1
        idict={}
        self.search()#nearest blank
        for i in range(self.start-li-1):
            if verb:print '**',i,self.buffer[li+i]
            key=self.buffer[li+i][0][1:]
            idict[key]=i
        if verb:print idict
        startI=self.search('noneblank',incr=0)
        endI=self.search(incr=0)
	##return [self.buffer[i][idict['Mol_common_name']] for i in range(startI,endI)]
	return [self.buffer[i][pos] for i in range(startI,endI)]

    def findSampleConditions(self,verb=True):
        self.search(s="_Saveframe_category",s2='sample_conditions',
                    positions=[0],positions2=[1])
	print 'finding sample conditions'
        if verb:print self.start
        li,mi=None,None
        while mi==None:
            li=self.search(s='loop_')
            mi=self.search(s='_Variable_type',stopstr='')
            if verb:print li,mi
            if self.terminated:
                print 'WARNING: no _Variable_type entry found'
                return None
        pos=mi-li-1
        idict={}
        self.search()#nearest blank
        for i in range(self.start-li-1):
            if verb:print '**',i,self.buffer[li+i]
            key=self.buffer[li+i][0][1:]
            idict[key]=i
        if verb:print idict
        startI=self.search('noneblank',incr=0)
        endI=self.search(incr=0)
	dct={}
	for i in range(startI,endI):
	   lin=self.buffer[i]
	   key=lin[pos]
	   if key[0]=="'":
	     key=key[1:]
	     for k in range(1,10):
		wordk=lin[pos+k]
		key+=(' '+wordk)
		if wordk[-1]=="'":
		  key=key[:-1]
		  break
	     ##dct[key]=lin[pos+k+1]
	     dct[key]=(lin[pos+k+1],lin[pos+k+1+2])#value,unit
	   else:
	     dct[key]=lin[pos+1]
	return dct
	
    def findDatabaseMatches(self,verb=False):
	dct={}
	self.start=0
        k=self.search('_Database_name')
	if k==None:
	  print 'warning no database matches'
	  return {}
	for i in range(len(self.buffer)-k):
	  lin=self.buffer[k+i]
	  if len(lin)>0:
	    if lin[0]=='stop_':return dct
	  if len(lin)>1:
	    dbnam=lin[0]
	    if not dbnam in dct:dct[dbnam]=[]
	    if dbnam in ['BMRB','PDB','DBJ','EMBL','GB','REF','SP','TPG']:
	      dbid=lin[1]
	      dct[dbnam].append(dbid)

def getShiftDBA31(start,end,buf,chnum,idict,includeLabel=True,includeAmb=False):
    dba=ShiftDict()
    for lst in buf[start:end]:
        ambc=lst[idict['Atom_chem_shift.Ambiguity_code']]
        ##if ambc in ['1','.']:
        if ambc in ['1','.','2']:#to include GLY
        ##if ambc in ['1','2','3','.']:#unamb, geminal, arom (sym degenerate), None
            val=lst[idict['Atom_chem_shift.Val_err']]
            if val=='.':std=None
            else:std=string.atof(val)
            if std==None or std<=1.3:#was 1.3
                elem=lst[idict['Atom_chem_shift.Atom_type']]
                res=lst[idict['Atom_chem_shift.Seq_ID']]
                rl3=lst[idict['Atom_chem_shift.Comp_ID']]
                at= lst[idict['Atom_chem_shift.Atom_ID']]
                avg=string.atof(lst[idict['Atom_chem_shift.Val']])
                if not dba.has_key(res):dba[res]={}
		dba[res][at]=[avg,std,rl3,ambc]
                dba.counts[elem]+=1
	    else:print 'skipping',std
    return dba  

def getShiftDBA(start,end,buf,chnum,idict,includeLabel=True,includeAmb=True,skipCO=False):#was includeAmb=False
    dba=ShiftDict()
    for lst in buf[start:end]:
        ambc=lst[idict['Chem_shift_ambiguity_code']]
        ##if ambc in ['1','.']:
        if ambc in ['1','.','2']:#to include GLY
        ##if ambc in ['1','2','3','.']:#unamb, geminal, arom (sym degenerate), None
            val=lst[idict['Chem_shift_value_error']]
            ##if val=='.':std=None
            if val in ['.','@']:std=None
            else:std=string.atof(val)
            if std==None or std<=1.3:#was 1.3
                elem=lst[idict['Atom_type']]
                res=lst[idict['Residue_seq_code']]
                ##res=lst[idict['Residue_author_seq_code']]
                rl3=lst[idict['Residue_label']]
                at= lst[idict['Atom_name']]
                avg=string.atof(lst[idict['Chem_shift_value']])
                if not dba.has_key(res):dba[res]={}
		if not (skipCO and at=='C'):
                 if includeLabel:
		  if includeAmb:
                    dba[res][at]=[avg,std,rl3,ambc]
		  else:
                    dba[res][at]=[avg,std,rl3]
                 else:
		  if includeAmb:
                    dba[res][at]=[avg,std,'X',ambc]
		  else:
                    dba[res][at]=[avg,std,'X']
                dba.counts[elem]+=1
		##print res,rl3,at,avg,ambc
		##print dba[res][at]
	    else:print 'skipping',std
    ##dba.show()
    if False:#includeAmb:
	dba.defineambiguities()
	dba.show()
    return dba  

def convChi2CDF(rss,k):
	return ((((rss/k)**(1.0/6))-0.50*((rss/k)**(1.0/3))+1.0/3*((rss/k)**(1.0/2)))\
		- (5.0/6-1.0/9/k-7.0/648/(k**2)+25.0/2187/(k**3)))\
		/ sqrt(1.0/18/k+1.0/162/(k**2)-37.0/11664/(k**3))



class ShiftGetter:

    def __init__(self,bmrID):
	self.bmrID=bmrID
	path=''
	bmrname=path+'bmr'+self.bmrID+'.str'
	print 'opening bmr shift file',bmrname
	try:open(bmrname)
	except IOError:
	  ##try:
	  ##  bmrname='new'+bmrname
	  ##  open(bmrname)
	  ##except IOError:
	    print 'getting bmrfile',self.bmrID
	    bmrpath='http://www.bmrb.wisc.edu/ftp/pub/bmrb/entry_directories/'
	    platform=sys.platform
	    if platform.startswith('linux'):
	      os.system('wget %sbmr%s/bmr%s_21.str'%(bmrpath,self.bmrID,self.bmrID))
	      os.system('mv bmr%s_21.str %sbmr%s.str'%(self.bmrID,path,self.bmrID))
	    elif 'os' in platform or platform.startswith('darwin'):
	      os.system('curl -O %sbmr%s/bmr%s_21.str'%(bmrpath,self.bmrID,self.bmrID))
	      os.system('mv bmr%s_21.str %sbmr%s.str'%(self.bmrID,path,self.bmrID))
	    elif 'win' in platform:
	    ##else:
	      os.system('C:\Windows\System32\curl.exe -O %sbmr%s/bmr%s_21.str'%(bmrpath,self.bmrID,self.bmrID))#if problems: -> manual download file...
	      os.system('ren bmr%s_21.str %sbmr%s.str'%(self.bmrID,path,self.bmrID))
	buf=initfil2(bmrname)
	parser=Parser(buf)
        parser.search(s="Polymer",s2='residue',positions=[1],positions2=[2])
        self.dbdct={}
        self.title='no title...'
	seq=''
	if True:
          k=parser.search('_Mol_residue_sequence')
	  if k==None:raise SystemExit, "no sequence found (bmrID might not exist) %s"%self.bmrID
          k-=1
	  print 'try buf[k]',buf[k]
	  if len(buf[k])>1:
	    seq=buf[k][1]
	    print 'found sequence',seq
	if len(seq)==0:
         i0=parser.search(s=";",positions=[0])
	 print i0
         if i0==None:
	  print 'warning no residues found (maybe missing semicolons?)',self.bmrID
	  self.moldba=ShiftDict()#counts er all 0
	  return
         i1=parser.search(s=";",positions=[0])
	 for i in range(i0,i1-1):
	  if len(buf[i])>0:#some lines can be blank
	    seq+=buf[i][0]
	##print i0,i1,seq
	seq=string.upper(seq)
	self.seq=seq
	if len(seq)<10:
	  print 'WARNING small or empty sequence!'
	  self.moldba=ShiftDict()#counts er all 0
	  return
	print 'seq: ',seq
	xcount=seq.count('X')
	xcount+=seq.count('U')#RNA
	print 'xcount:',self.bmrID,xcount
	if xcount>5 or xcount*1.0/len(seq)>0.15:
	  print 'warning: high xcount',xcount,seq
	  self.moldba=ShiftDict()#counts er all 0
	  return
	moldba=parser.findShiftData('none',verb=False,skipCO=self.bmrID in ['15719','15274','15506'])
	print moldba.counts
	self.moldba=moldba
	parser.start=0
        k=parser.search('_Entry_title')
	self.title=string.join(buf[k+1],' ')
	if buf[k+2][0]<>';':
	  self.title+=('  '+string.join(buf[k+2],' '))
        k2=parser.search('_Details')
	if k2<>None and len(buf[k2-1])>1 and buf[k2-1][1]<>'.':
	  print 'details:',k2,buf[k2-1]
	  self.title+=(' :'+string.join(buf[k2-1][1:],' '))
	print self.title
        self.references=parser.findReference(verb=False)
        print self.references
	parser.start=0
        parser.terminated=False#OK?
        conddct=parser.findSampleConditions(verb=False)
	print conddct
	##self.pH=-9.99;self.temperature=999.9
	self.pH=7.0;self.temperature=298.0
	if 'pH' in conddct:
	   if conddct['pH']=='.':self.pH=-9.99
	   else:self.pH=eval(conddct['pH'])
	if 'temperature' in conddct:
	   if conddct['temperature']=='.':self.temperature=298.0
	   else:self.temperature=eval(conddct['temperature'])
	if 'ionic strength' in conddct:
	   ionstr=conddct['ionic strength'][0]
	   if ionstr in ['.',' ','']:self.ion=0.1
	   else:self.ion=eval(conddct['ionic strength'][0])
	   unit=conddct['ionic strength'][1]
	   if unit=='mM':self.ion/=1000.0
	else:self.ion=0.1
	parser.start=0
        k=parser.search('_System_physical_state')
        if k==None:
	  print 'warning: phys_state not defined in bmrbfile',self.bmrID
	  self.phys_state='unknown'
	else: self.phys_state=string.join(buf[k-1][1:],' ')
	print self.phys_state
	print 'summary_info: %5s %4.2f %5.1f %3d'%(self.bmrID,self.pH,self.temperature,len(self.seq)),
	print self.phys_state,
	print self.title
        self.dbdct=parser.findDatabaseMatches(verb=True)
        print self.dbdct
        
    def write_rereferenced(self,lacsoffs):
	out=open('rereferenced/bmr%s.str'%self.bmrID,'w')
	out.write('data_%s\n'%self.bmrID)
	out.write('''
#######################
#  Entry information  #
#######################

save_entry_information
   _Saveframe_category      entry_information

   _Entry_title
;
''')
	out.write('%s\n'%self.title)
	out.write(';\n')
	out.write('''
        ##############################
        #  Polymer residue sequence  #
        ##############################

''')
	s=self.seq
	out.write('_Residue_count   %d\n'%len(s))
	out.write('''_Mol_residue_sequence
;
''')
	for i in range((len(s)-1)/20+1):
	  out.write('%s\n'%s[i*20:min(len(s),(i+1)*20)])
	out.write('''

        ###################################
        #  Assigned chemical shift lists  #
        ###################################

save_assigned_chem_shift_list_1
   _Saveframe_category               assigned_chemical_shifts

   loop_
      _Atom_shift_assign_ID
      _Residue_author_seq_code
      _Residue_seq_code
      _Residue_label
      _Atom_name
      _Atom_type
      _Chem_shift_value
      _Chem_shift_value_error
      _Chem_shift_ambiguity_code

''')
	cnt=1
	for i in range(len(s)):
	  res=str(i+1)
	  if res in self.moldba:
	    shdct=self.moldba[res]
	    for at in shdct:
		shave,shstd,rl3,ambc=shdct[at]
		if at in lacsoffs:refsh=shave+lacsoffs[at]
		else:refsh=shave
		if shstd==None:
		  if at[0] in 'CN':shstd=0.3
		  else:shstd=0.05
		data=(cnt,res,res,rl3,at,at[0],refsh,shstd,ambc);print data
		out.write('     %4d %3s %3s %3s %-4s %1s %7.3f %4.2f %1s\n'%data)
		cnt+=1
	out.write('''

   stop_

save_
''')
	out.close()

    def writeShiftY(self,out):
	header='#NUM AA HA CA CB CO N HN'
        bbatns =['HA','CA','CB','C','N','H']##,'HA3','HB','HB2','HB3']
	out.write(header+'\n')
	seq=self.seq
	for i in range(1,len(seq)-1):
	  res=str(i+1)
	  resi=seq[i]
	  out.write('%s %1s'%(res,resi))
	  if res in self.moldba:
	    shdct=self.moldba[res]
	    for at in bbatns:
	      sho=0.0
	      if resi=='G' and at=='HA':
		if 'HA2' in shdct and 'HA3' in shdct:
		  sho=(shdct['HA2'][0]+shdct['HA3'][0])/2
	      if at in shdct:
		sho=shdct[at][0]
	      out.write(' %7.3f'%sho)
	  out.write('\n')

    def cmp2pred1(self,verb=False):
	seq=self.seq
        predshiftdct=potenci(seq,self.pH,self.temperature,self.ion)
        bbatns0=['C','CA','CB','HA','H','N']
        bbatns =['C','CA','CB','HA','H','N','HB']##'HA2','HA3','HB','HB2','HB3']
        ##bbatns =['C','CA','CB','H','N']##'HA2','HA3','HB','HB2','HB3']
	cmpdct={}
	self.shiftdct={}
	for i in range(1,len(seq)-1):
	  res=str(i+1)
	  ##if res in self.moldba:
	  if res in self.moldba and seq[i] in aa1s:
	    trip=seq[i-1]+seq[i]+seq[i+1]
	    shdct=self.moldba[res]
	    for at in bbatns:
	      sho=None
	      if at in shdct:
		sho=shdct[at][0]
	      elif seq[i]=='G' and at=='HA' or at=='HB':
		shs=[]
		for pref in '23':
		  atp=at+pref
		  if atp in shdct:
		    shs.append(shdct[atp][0])
		if len(shs)>0:sho=average(shs)
	      if sho<>None:
		if i==1:
		  pent='n'+     trip+seq[i+2]
		elif i==len(seq)-2:
		  pent=seq[i-2]+trip+'c'
		else:
		  pent=seq[i-2]+trip+seq[i+2]
		##shp=refinedpred(paramdct[at],pent,at,tempdct,self.temperature)
		shp=predshiftdct[(i+1,seq[i])][at]
		if shp<>None:
		  self.shiftdct[(i,at)]=[sho,pent]
		  diff=sho-shp
		  if verb:print 'diff is:',self.bmrID,i,seq[i],at,sho,shp,abs(diff),diff
		  if not at in cmpdct:cmpdct[at]={}
		  cmpdct[at][i]=diff
	return cmpdct

    def visresults(self,dct,doplot=True,dataset=None,offdct=None,label='',minAIC=999.0,lacsoffs=None,cdfthr=6.0):##6.0):#was minAIC=9.0
	shout=open('shifts%s.txt'%self.bmrID,'w')
        bbatns=['C','CA','CB','HA','H','N','HB']
	cols='brkgcmy'
	refined_weights={'C':0.1846,'CA':0.1982,'CB':0.1544,'HA':0.02631,'H':0.06708,'N':0.4722,'HB':0.02154}
	outlivals={'C':5.0000,'CA':7.0000,'CB':7.0000,'HA':1.80,   'H':2.30,   'N':12.00, 'HB':1.80}
	dats={}
	maxi=max([max(dct[at].keys()) for at in dct])
	mini=min([min(dct[at].keys()) for at in dct])#is often 1
	nres=maxi-mini+1
	resids=range(mini+1,maxi+2)
	self.mini=mini
	tot=zeros(nres)
	newtot=zeros(nres)
	newtotsgn=zeros(nres)
	newtotsgn1=zeros(nres)
	newtotsgn2=zeros(nres)
	totnum=zeros(nres)
	allrmsd=[]
	totbbsh=0
	oldct={}
	allruns=zeros(nres)
	rdct={}
	sgnw={'C':1.0,'CA':1.0,'CB':-1.0,'HA':-1.0,'H':-1.0,'N':-1.0,'HB':1.0}#was 'HB':0.0
	##wbuf=initfil2('weights_oplsda123new7');wdct={}
        wbuf=[['weights:', 'N', '-0.0626', '0.0617', '0.2635'], ['weights:', 'C', '0.2717', '0.2466', '0.0306'], ['weights:', 'CA', '0.2586', '0.2198', '0.0394'], ['weights:', 'CB', '-0.2635', '0.1830', '-0.1877'], ['weights:', 'H', '-0.3620', '1.3088', '0.3962'], ['weights:', 'HA', '-1.0732', '0.4440', '-0.4673'], ['weights:', 'HB', '0.5743', '0.2262', '-0.3388']]
        wdct={}
	for lin in wbuf:wdct[lin[1]]=[eval(lin[n]) for n in (2,3,4)]#lin[2] is first component
	for at in dct:
	  vol=outlivals[at]
	  subtot=zeros(nres)
	  subtot1=zeros(nres)
	  subtot2=zeros(nres)
	  if dataset<>None:dataset[at][self.bmrID]=[]
	  A=array(dct[at].items())
	  totbbsh+=len(A)
	  I=bbatns.index(at)
	  w=refined_weights[at]
	  shw=A[:,1]/w
	  off=average(shw)
	  rms0=sqrt(average(shw**2))
	  if offdct<>None:
            shw-=offdct[at]#offset correction
            print 'using predetermined offset correction',at,offdct[at],offdct[at]*w
	  ##shw-=off
	  shwl=list(shw)
	  for i in range(len(A)):
	    resi=int(A[i][0])-mini#minimum value for resi is 0
	    ashwi=abs(shw[i])
	    if ashwi>cdfthr:oldct[(at,resi)]=ashwi
	    tot[resi]+=(min(4.0,ashwi)**2)
	    for k in [-1,0,1]:
	      if 0<=resi+k<len(subtot):##maxi:
	        ##subtot[resi+k]+=(shw[i]*w*wdct[at][0])
	        subtot[resi+k]+=(clip(shw[i]*w,-vol,vol)*wdct[at][0])
	        ##subtot1[resi+k]+=(shw[i]*w*wdct[at][1])
	        subtot1[resi+k]+=(clip(shw[i]*w,-vol,vol)*wdct[at][1])
	        subtot2[resi+k]+=(clip(shw[i]*w,-vol,vol)*wdct[at][2])
	    totnum[resi]+=1
	    if offdct==None:
	      if 3<i<len(A)-4:
		vals=shw[i-4:i+5]
		runstd=std(vals)
		allruns[resi]+=runstd
		if not resi in rdct:rdct[resi]={}
		rdct[resi][at]=average(vals),sqrt(average(vals**2)),runstd
	  dats[at]=shw
	  stdw=std(shw)
	  dAIC=log(rms0/stdw)*len(A)-1
	  print 'rmsd:',at,stdw,off,dAIC
	  allrmsd.append(std(shw))
	  if doplot:
	    subplot(211)
	    sca=scatter(A[:,0]+1,shw,alpha=0.5,s=25,edgecolors='none',c=cols[I])
	    plot((mini+1,maxi+1),[0.0,0.0],'k--')
	    axis([mini+1,max(resids),-20,20])
	    ylabel('Weighted Sec Chem Shifts')
	  newtot+=((subtot/3.0)**2)
	  newtotsgn+=subtot
	  newtotsgn1+=subtot1
	  newtotsgn2+=subtot2
	T0=list(tot/totnum)
	cdfs=convChi2CDF(tot,totnum)
	Th=list(tot/totnum*0.5)
	tot3=array([0,0]+Th)+array([0]+T0+[0])+array(Th+[0,0])
	Ts=list(tot)
	Tn=list(totnum)
	tot3f=array([0,0]+Ts)+array([0]+Ts+[0])+array(Ts+[0,0])
	totn3f=array([0,0]+Tn)+array([0]+Tn+[0])+array(Tn+[0,0])
	cdfs3=convChi2CDF(tot3f[1:-1],totn3f[1:-1])
	newrms=(newtot*3)/totn3f[1:-1]
	newcdfs=convChi2CDF(newtot*3,totn3f[1:-1])
        avc=average(cdfs3[cdfs3<20.0])
        numzs=len(cdfs3[cdfs3<20.0])
        numzslt3=len(cdfs3[cdfs3<cdfthr])
        stdcp=std(cdfs3[cdfs3<20.0])
	atot=sqrt(tot3/2)[1:-1]
	aresids=array(resids)
	if offdct==None:
	  tr=(allruns/totnum)[4:-4]
	  offdct={}
	  mintr=None;minval=999
	  for j in range(len(tr)):
	    if j+4 in rdct and len(rdct[j+4])==len(dct):#all ats must be represented for this res
	    ##if j+4 in rdct and len(rdct[j+4])>=len(dct)-1:#all ats (except one as max) must be represented for this res
	      if tr[j]<minval:
		minval=tr[j]
		mintr=j
	  if mintr==None:return None#still not found
	  print len(tr),len(resids[4:-4]),len(atot),mintr+4,min(tr),tr[mintr]##,tr
	  for at in rdct[mintr+4]:
	    roff,std0,stdc=rdct[mintr+4][at]
	    dAIC=log(std0/stdc)*9-1
	    print 'minimum running average',at,roff,dAIC
	    if dAIC>minAIC:
	      print 'using offset correction:',at,roff,dAIC,self.bmrID,label
	      offdct[at]=roff
	    else:
	      print 'rejecting offset correction due to low dAIC:',at,roff,dAIC,self.bmrID,label
	      offdct[at]=0.0
	  return offdct #with the running offsets
	if dataset<>None:
	  csgns= newtotsgn/totn3f[1:-1]*10
	  csgnsq=newtotsgn/sqrt(totn3f[1:-1])*10
	  for I in range(len(resids)):
	    pass##print 'datapoints:',self.bmrID,resids[I],csgns[I],cdfs3[I],csgnsq[I],totn3f[1:-1][I]
	if doplot:
	  subplot(212)
	  ##plot(resids,newtotsgn/sqrt(totn3f[1:-1])*8.0,'r-')
	  ##plot(resids,newtotsgn1/sqrt(totn3f[1:-1])*8.0,'b-')
	  plot(resids,cdfs3,'k-')
	  ##plot((mini+1,maxi+1),[cdfthr,cdfthr],'g--')
	  plot((mini+1,maxi+1),[8.0,8.0],'g--')
	  plot((mini+1,maxi+1),[3.0,3.0],'k--')
	  ##plot((mini+1,maxi+1),[0.0,0.0],'k--')
	  ##axis([mini+1,max(resids),-16,16])
	  axis([mini+1,max(resids),-4,16])
	  ylabel('CheZOD Z-scores')
	  xlabel('Residue number')
	if True:
	 sferr3=0.0
	 for at in dats:
	  I=bbatns.index(at)
	  ashw=abs(dats[at])
	  Terr=linspace(0.0,5.0,26)
	  ferr=array([sum(ashw>T) for T in Terr])*1.0/len(ashw)+0.000001
	  sferr3+=ferr[15]#3.0std-fractile
	 aferr3=sferr3/len(dats)
	 F=zeros(2)
	 for at in dats:
	  ashw=abs(dats[at])
	  fners=sum(ashw>1.0)*1.0/len(ashw),sum(ashw>2.0)*1.0/len(ashw)
	  ##print 'fnormerr:',at,fners[0],fners[1]
	  F+=fners
	 totnorm=sum(atot>1.5)*1.0/len(atot)
	 outli0=aresids[atot>1.5]
	 outli1=aresids[cdfs>cdfthr]
	 outli3=aresids[cdfs3>cdfthr]
	 newoutli3=aresids[newcdfs>cdfthr]
	 finaloutli=[i+mini+1 for i in range(nres) if cdfs[i]>cdfthr or cdfs3[i]>cdfthr and cdfs[i]>0.0 and totnum[i]>0]
	 print 'outliers:',self.bmrID,len(outli0),len(outli1),len(outli3),len(finaloutli),sum(totnum==0)##,finaloutli
	 Fa=F/len(dats)
	 fout=len(finaloutli)*1.0/nres
	 print len(oldct),mini,maxi,nres,aresids[totnum==0]
	 print 'summary_stat: %5s %5d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %4d'%\
	  (self.bmrID,sum(self.moldba.counts.values()),average(allrmsd),Fa[0],Fa[1],fout,aferr3,totnorm,totbbsh)

	#now accumulate the validated data
	atns=dct.keys()
	accdct=dict(zip(atns,[[] for _ in atns]))
	numol=0
	iatns=self.shiftdct.keys();iatns.sort()
	for i,at in iatns:#i is seq enumeration (starting from 0, but terminal allways excluded)
	  I=bbatns.index(at)
	  w=refined_weights[at]
	  ol=False
	  if i+1 in finaloutli:ol=True
	  elif (at,i-mini) in oldct:ol=True
	  if not ol:
		accdct[at].append(dct[at][i])
	  else:numol+=1
	  if dataset<>None:
	    dataset[at][self.bmrID].append(self.shiftdct[(i,at)]+[ol])
	    vals=dataset[at][self.bmrID][-1]
	    ##shout.write('%3d %2s %7.3f %5s %6.3f\n'%(i+1,at,vals[0],vals[1],dct[at][i]))
	    shout.write('%3d %1s %2s %7.3f %5s %6.3f\n'%(i+1,vals[1][2],at,vals[0],vals[1],dct[at][i]))
	sumrmsd=0.0;totsh=0
	newoffdct={}
	for at in accdct:
	  I=bbatns.index(at)
	  w=refined_weights[at]
	  vals=accdct[at]
	  vals=array(vals)/w
	  anum=len(vals)
	  if anum==0:newoffdct[at]=0.0
	  else:
	    aoff=average(vals)
	    astd0=sqrt(average(array(vals)**2))
	    astdc=std(vals)
	    adAIC=log(astd0/astdc)*anum-1
	    if adAIC<minAIC or anum<4:
	      print 'rejecting offset correction due to low adAIC:',at,aoff,adAIC,anum,self.bmrID,label,
	      if lacsoffs<>None and at in lacsoffs:print 'LACS',lacsoffs[at],-aoff*w
	      else:print
	      astdc=astd0
	      aoff=0.0
	      shout.write('off %2s   0.0\n'%at)
	    else:
	      print 'using offset correction:',at,aoff,adAIC,anum,self.bmrID,label,
	      if lacsoffs<>None and at in lacsoffs:print 'LACS',lacsoffs[at],-aoff*w
	      else:print
	      shout.write('off %2s %7.3f\n'%(at,aoff*w))
	    sumrmsd+=(astdc*anum);totsh+=anum
	    ##print 'accepted stats: %2s %3d %6.3f %5.3f %5.3f %6.3f'%(at,anum,aoff,astd0,astdc,adAIC)
	    newoffdct[at]=aoff
        compl=calc_complexity(cdfs3,self.bmrID,thr=cdfthr)
	fullrmsd=average(allrmsd)
	ps=self.phys_state
	ps6=ps.strip("'")[:6]
	fraczlt3=numzslt3*1.0/numzs
	if totsh==0:avewrmsd,fracacc=9.99,0.0
	else:avewrmsd,fracacc=sumrmsd/totsh,totsh/(0.0+totsh+numol)
	allsh=sum(totnum)
	ratsh=allsh*1.0/numzs
	print 'finalstats %5s %8s %6s %7.4f %6.4f %6.4f %4d %4d %4d %7.3f %3d %3d %4d %6.4f %6.4f %7.3f %8.5f'\
	 %(self.bmrID,label,ps6,avewrmsd,fullrmsd,fracacc,nres,totsh,numol,avc,numzs,numzslt3,allsh,fraczlt3,ratsh,stdcp,compl)##,
	if dataset<>None:
	  if len(newoffdct)>6 or len(newoffdct)==6 and 'HB' not in newoffdct:
	    print 'testoff:',
	    for atn in ['CA','C','CB','N','H','HA']:print newoffdct[atn],
	    print
	  fracol3=len(outli3)*1.0/len(totnum>0)
	  newfracol3=len(newoutli3)*1.0/len(totnum>0)
	  if newfracol3<=0:lratf=0.0
	  else:lratf=log(fracol3/newfracol3)
	  print 'fraccdfs3gt3 %5s %7.4f %7.4f %6.3f'%(self.bmrID,fracol3,newfracol3,lratf)
	  if doplot:
	    subplot(211)
	    ##title('%5s %5.3f %5.3f '%(self.bmrID,1-fracol3,compl)+self.title[:60])
	    title('Secondary chemical shifts and CheZOD Z-scores for %s'%self.bmrID)
	  ##return cdfs3,newtotsgn/sqrt(totn3f[1:-1])*8.0,newtotsgn1/sqrt(totn3f[1:-1])*8.0
	  return resids,cdfs3,newtotsgn/sqrt(totn3f[1:-1])*8.0,newtotsgn1/sqrt(totn3f[1:-1])*8.0,newtotsgn2/sqrt(totn3f[1:-1])*8.0
	print
	return avewrmsd,fracacc,newoffdct,cdfs3 #offsets from accepted stats
	
    def savedata(self,cdfs3,pc1ws,pc2ws,pc3ws):
	out=open('zscores%s.txt'%self.bmrID,'w')
	s=self.seq
	for i,x in enumerate(cdfs3):
	  if x<99:#not nan
	    I=i+self.mini
	    aai=s[I]
	    pci1=pc1ws[i]
	    pci2=pc2ws[i]
	    pci3=pc3ws[i]
	    ##out.write('%s %3d %6.3f %6.3f %6.3f\n'%(aai,I+1,x,pci1,pci2))
	    ##out.write('%s %3d %6.3f %6.3f %6.3f %6.3f\n'%(aai,I+1,x,pci1,pci2,pci3))
	    out.write('%s %3d %6.3f\n'%(aai,I+1,x))
	out.close()

def calc_borders(c,ID,thr=3.0,lim=10,lim2=10,verb=False):
    a=''
    for x in c:
	if x<thr:a+='0'
	elif x>=thr:a+='1'
	else:a+='-'
    ##a='00000000000000000111111100000----0000000111111111111111111111110000-0000-00000011111--1111111000'
    borders=[]
    prev='-'
    nancounts=[0]
    for i,x in enumerate(a):
      if x<>prev and prev<>'-' and x<>'-':
	borders.append(i)
	nancounts.append(0)
      if x<>'-':prev=x
      else:nancounts[-1]+=1
    N=len(a)
    first=a[0]
    ni=first<>'0'
    b0=array([0]+borders)
    b1=array(borders+[N])
    d=b1-b0
    if verb:
      print borders
      print d
      print nancounts
    lst=[]
    for j,dj in enumerate(d):
      if j%2==ni and dj>lim:
       if dj-nancounts[j]>lim2:
	if verb:print 'idrs:',ID,j,dj,b0[j:j+2],dj-nancounts[j]
	lst.append((dj,dj-nancounts[j],b0[j:j+2]))
	if nancounts[j]>0:
	  print 'testidrs:',ID,lim,lim2,j,dj,b0[j:j+2],dj-nancounts[j]
    return lst
##calc_borders([0,6],'test',lim=15,lim2=11,verb=True)
##1/0

def calc_complexity(c,ID,thr=3.0,ret=1,verb=False):
    ##c=array([random.rand() for _ in range(1000)])
    ##a=c<0.5
    if isinstance(c[0],str):a=c
    else:
     a=''
     for x in c:
	if x<thr:a+='0'
	elif x>=thr:a+='1'
	else:a+='-'
    ##a='000000000000000001111111000000000----000000000001111111111111111111111100000000000000111111111111'
    ##print 'binstr:',a
    ##print a.count('-')
    borders=[]
    prev='-'
    for i,x in enumerate(a):
      if x<>prev and prev<>'-' and x<>'-':
	borders.append(i)
      if x<>'-':prev=x
    if verb:print len(borders)
    if verb:print borders
    N=len(a)
    b0=array([0]+borders)
    b1=array(borders+[N])
    d=b1-b0
    f=d*1.0/N
    s=sum(f*log(f))
    entr=exp(-s)-1
    ##if isinstance(c[0],str):nonnans=array([True]*len(a))
    if isinstance(c[0],str):avc=9.999
    else:
	nonnans=c<10.0
	avc=average(c[nonnans])
    if verb:print 'entropy: %5s %7.4f %8.5f %6.3f'%(ID,entr,entr/N,avc)
    if ret==1:return entr/N
    else:return entr/N,d

def getCheZODandPCs(ID,usetcor=True,minAIC=6.0,doplot=True):
       bbatns=['C','CA','CB','HA','H','N','HB']
       dataset=dict(zip(['HA','H','N','CA','CB','C','HB'],[{} for _ in range(7)]))
       refined_weights={'C':0.1846,'CA':0.1982,'CB':0.1544,'HA':0.02631,'H':0.06708,'N':0.4722,'HB':0.02154}
       sg=ShiftGetter(ID)
       if not usetcor:sg.temperature=298.0
       totsh=sum(sg.moldba.counts.values())
       print totsh,sg.seq
       if sg.moldba.counts['P']>1 or 'U' in sg.seq:
	raise SystemExit,'skipping DNA/RNA %s %s'%(ID,sg.title)
       if len(sg.seq)<5:## or totsh/len(sg.seq)<1.5:
	raise SystemExit,'too short sequence or too few shifts %s %s'%(ID,sg.title)
       dct=sg.cmp2pred1()
       totbbsh=sum([len(dct[at].keys()) for at in dct])
       print 'total backbone shifts:',totbbsh
       offr=sg.visresults(dct,False,minAIC=minAIC)
       if offr<>None:
         atns=offr.keys()
         off0=dict(zip(atns,[0.0 for _ in atns]))
         armsdc,frac,noffc,cdfs3c=sg.visresults(dct,False,offdct=offr,label='ofcor',minAIC=minAIC)
       else:
	 print 'warning: no running offset could be estimated',ID
         off0=dict(zip(bbatns,[0.0 for _ in bbatns]))
	 armsdc=999.9;frac=0.0
       armsd0,fra0,noff0,cdfs30=sg.visresults(dct,False,offdct=off0,label='nocor',minAIC=minAIC)
       usefirst=armsd0/(0.01+fra0)<armsdc/(0.01+frac)
       av0=average(cdfs30[cdfs30<20.0])#to avoid nan
       if offr<>None:
	avc=average(cdfs3c[cdfs3c<20.0])
	orusefirst=av0<avc
	if usefirst<>orusefirst:print #'warning hard decission',usefirst,orusefirst
	print 'decide',orusefirst,ID,armsd0,fra0,av0,armsdc,frac,avc
       else:orusefirst=True
       if orusefirst: #was usefirst
         ##resids,cdfs3,pc1ws,pc2ws=sg.visresults(dct,True,dataset,offdct=noff0,label='nocornew',minAIC=minAIC)
         resids,cdfs3,pc1ws,pc2ws,pc3ws=sg.visresults(dct,doplot,dataset,offdct=noff0,label='nocornew',minAIC=minAIC)
       else:
         resids,cdfs3,pc1ws,pc2ws,pc3ws=sg.visresults(dct,doplot,dataset,offdct=noffc,label='ofcornew',minAIC=minAIC)
       sg.savedata(cdfs3,pc1ws,pc2ws,pc3ws)
       ##show()

def getseccol(pc1,pc2):
    i=(clip(pc1,-12,12)+12)/24 #from 0 to 1
    j=(clip(pc2,-8,8)+8)/16
    return (i,1-j,1-i)

def getprobs(pc1,pc2):
    N=array(histdct_flattened).reshape((25, 35, 3))
    secs='HCS'
    ind1=18+clip(pc1,-18,16.999)
    ind2=12+clip(pc2,-12,12.999)
    ##pr0=[array([dct[ss][0][int(ind2[i]),int(ind1[i])] for ss in secs]) for i in range(len(ind1))]
    ##nprobs=[prob/sum(prob) for prob in pr0]
    nprobs=[N[int(ind2[i]),int(ind1[i])] for i in range(len(ind1))]
    return nprobs

def viscolentry(ID,pref='',delta=0,doreturn=False,pdbID='',returnpcs=False):
    zbuf=initfil2('%szscores%s.txt'%(pref,ID))
    ##zbuf=initfil2('CheZOD%s/zscoresmod%s.txt'%(pref,ID))
    resi=[eval(zlin[1])+delta for zlin in zbuf]
    zsco=[eval(zlin[2]) for zlin in zbuf]
    ##pc1s=[eval(zlin[3]) for zlin in zbuf]
    ##pc2s=[eval(zlin[4]) for zlin in zbuf]
    if returnpcs: return resi,pc1s,pc2s,zsco##,C
    ##rgbs=getseccol(array(pc1s),array(pc2s))
    ##C=array(rgbs).transpose()
    if doreturn:
	return resi,C,zsco
	C3=zeros((len(C),1,3))
	C3[:,0,:]=C
	imshow(rot90(C3),interpolation='none');show();1/0
    for i,ri in enumerate(resi):
	zi=zsco[i]
	coli=C[i]
	##bari=bar(ri+0.5,zi,width=1.0,fc=coli,ec='none')
	bari=bar(ri-0.5,zi,width=1.0,fc=coli,ec='none')
    title(ID+'  '+pdbID)
    return resi,pc1s,pc2s,C
    ##axis([-7,105,0,16])

def getramp(buf):
    dct={}
    for i in range(64):
      for j in range(4):
        rgb=[string.atof(x)/255 for x in buf[i][4*j+1:4*j+4]]
        dct[i+j*64]=rgb
    return dct

def getColor(f,ramp):
    return ramp[int(255*f)]

def gencolpml(ID,pdbid='',delta=0):
    ramp1=getramp(initfil2('ramp1.dat'))
    resi,coli,zsco=viscolentry(ID,'',delta,doreturn=True)
    ##pmlfile=open('colCheZOD%s_%s.pml'%(ID,pdbid),'w')
    pmlfile=open('colCheZOD%s.pml'%ID,'w')
    for i,ri in enumerate(resi):
	rgbi=coli[i]
	##print i,ri,rid,delta,rgbi,1/0
	##pmlfile.write('set_color coluser%d, '%ri)
	pmlfile.write('set_color coluser%d, '%ri)
	pmlfile.write('[%5.3f, %5.3f, %5.3f]\n'%tuple(rgbi))
	##pmlfile.write('color coluser%d, resi %s and chain A and %s\n'%(ri,ri,pdbid))
	pmlfile.write('color coluser%d, resi %s\n'%(ri,ri))
    pmlfile.close()

def visprobs(resi,probs):
    seccols='rgb'
    for i,ri in enumerate(resi):
	probi=probs[i]
	bottoms=[0.0,probi[0],probi[0]+probi[1]]
	for n in range(3):
	  bari=bar(ri-0.5,probi[n],bottom=bottoms[n],width=1.0,fc=seccols[n],ec='none')

def lineplot2dpcs2(resi,pc1s,pc2s,cols,ID):
   clf()
   title('Principal components '+ID)
   scatter(pc1s,pc2s,c=cols,s=100,faceted=False)
   plot(pc1s,pc2s,'k')
   plot([-18,18],[0,0],'k--')
   plot([0,0],[-16,16],'k--')
   axis([-18,18,-16,16])
   show();1/0


ID=sys.argv[1]
doplot=True
if "-n" in sys.argv:doplot=False
if doplot: from pylab import *
getCheZODandPCs(ID,doplot=doplot)
if doplot:
  tight_layout()
  show()
