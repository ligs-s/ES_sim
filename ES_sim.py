#Imports
from chroma import make, view
from chroma.geometry import Geometry, Material, Solid, Surface, Mesh
from chroma import optics
from chroma.transform import make_rotation_matrix
from chroma.demo.optics import glass, water, vacuum
from chroma.demo.optics import black_surface, r7081hqe_photocathode
from chroma.loader import create_geometry_from_obj
from chroma.detector import Detector
from chroma.pmt import build_pmt
from chroma.event import Photons
from chroma.sim import Simulation
from chroma.sample import uniform_sphere
import numpy as np
from chroma.transform import make_rotation_matrix, get_perp, rotate, rotate_matrix, normalize
#from scipy import stats
from matplotlib.ticker import NullFormatter
#import pyparsing
import time
from chroma.stl import mesh_from_stl
import setupMaterials as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ROOT import TFile, TTree
from array import array

#if __name__ == '__main__':
from chroma.sim import Simulation
from chroma import sample
from chroma.event import Photons
import chroma.event as chromaev
from chroma.loader import load_bvh
from chroma.generator import vertex
from chroma.io.root import RootWriter

import sys

print "===================================="
print "ES simulation"
print "===================================="
setup = Detector(sm.Vac)

#Make TFiles and a  TTree
file = None
source_type = None
source_position = None
sphere = None
nphotons = None
outdir = './data'
if len(sys.argv)<=1:
    file = TFile('test.root','recreate')
elif len(sys.argv)<=6:
    source_type = sys.argv[1]
    source_position = sys.argv[2]
    sphere = sys.argv[3]
    if len(sys.argv)>=5:
        nphotons = int(sys.argv[4])
    if len(sys.argv)>=6:
        outdir = sys.argv[5]
    filename = '%s/stype_%s_spos_%s_sphere_%s.root' % (outdir, source_type, source_position, sphere)
    file = TFile(filename, 'recreate')
else:
    print 'unknown argmuents, exit ...'
    sys.exit(1)

#Read in stl files from solid works
allstls = []
with open('./fnames.txt') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#'):
            print 'skip parts', line
            continue
        if sphere!=None and not bool(int(sphere)) and line.find('sphere')>0:
            print 'skip parts ', line, 'by user from cmd line arguments'
            continue
        allstls.append(line)


#List of meshes, solids
meshes = []
solids = []
scenters = []
dcenters = []

#Shift detector origin
xshift = -4.3977036476135254
yshift = 95.611244201660156
zshift = -1.3024694919586182
#xshift = 0
#yshift = 0
#zshift = 0
global_shift = np.array([xshift, yshift, zshift])
det_shift = global_shift + np.array([0, 0, 0]) # additonal shift for detector for sanity check

# rotation if needed
rotation=make_rotation_matrix(-np.pi*0/2., (1,0,0))

# source position, shift w.r.t detector
x_source = 0
y_source = -120
z_source = 0

if not source_position==None:
    y_source = int(source_position)


#Values for tree - although int values, must be stored in array form b/c python
maxn=100000
z = array('f', maxn*[0])
x = array('f', maxn*[0])
y = array('f', maxn*[0])
t = array('f', maxn*[0])
n = array('i', [0])
p = array('i', [0])
d = array('i', [0])
ad = array('f',maxn*[0])
ar = array('f',maxn*[0])
evt = array('i',[0])
zi = array('f', maxn*[0])
xi = array('f', maxn*[0])
yi = array('f', maxn*[0])
ti = array('f', maxn*[0])
flags = array('i',maxn*[0])



tree = TTree('tree', 'data from Photon events')
tree.Branch('num',n,'num/I')
tree.Branch('z_det', z, 'z_det[num]/F')
tree.Branch('x_det', x, 'x_det[num]/F')
tree.Branch('y_det', y, 'y_det[num]/F')
tree.Branch('t_det', t, 't_det[num]/F')
tree.Branch('n_gen', p, 'n_gen/I')
tree.Branch('n_det', d, 'n_det/I')
tree.Branch('theta_gen',ar,'theta_gen[num]/F')
tree.Branch('theta_det',ad,'theta_det[num]/F')
tree.Branch('nevt',evt,'nevt/I')
tree.Branch('z_gen', zi, 'z_gen[num]/F')
tree.Branch('x_gen', xi, 'x_gen[num]/F')
tree.Branch('y_gen', yi, 'y_gen[num]/F')
tree.Branch('t_gen', ti, 't_gen[num]/F')
tree.Branch('pFlags',flags,'pFlags[num]/I')

#Build the geomtry****************************************************************************************************************************

#note: stl files correctly oriented, center at 0 0 0 
# build detector. place parts, rotate if possible, assign materials
for xx in range(0,len(allstls)):
    print 'index', xx, allstls[xx]
    meshes.append(mesh_from_stl(allstls[xx]))
    if 'detector' in allstls[xx]:
        #solids.append(Solid(meshes[xx],sm.fullAbsorb,sm.LXe,surface=sm.SiPMSurface,color=0x3300ee22))
        solids.append(Solid(meshes[xx],sm.fullAbsorb,sm.Vac,surface=sm.iSiPMSurface,color=0x3300ee22))
        setup.add_pmt(solids[xx],rotation=rotation, displacement=det_shift)

        #Getting centers of the detector SiPM
        trianglecenters = meshes[xx].get_triangle_centers()
        xPos = np.mean(trianglecenters[:,0])
        yPos = np.mean(trianglecenters[:,1])
        zPos = np.mean(trianglecenters[:,2])
        dcenters.append([xPos + xshift,yPos + yshift,zPos + zshift])
        #dcenters.append(rotation*np.array([xPos, yPos, zPos])+np.array(xshift,yshift,zshift))
        #print np.inner(rotation, np.array(xPos,yPos,zPos))
    else:
        # add all other parts
        if 'sphere' in allstls[xx]:
            solids.append(Solid(meshes[xx],sm.fullAbsorb,sm.Vac,surface=sm.teflonSurface,color=0x3365737e))
        if 'filter' in allstls[xx]:
            solids.append(Solid(meshes[xx],sm.fullAbsorb,sm.Vac,surface=sm.teflonSurface,color=0x3365737e))
        elif 'copper' in allstls[xx]:
            #solids.append(Solid(meshes[xx],sm.fullAbsorb,sm.Vac,surface=sm.copperSurface,color=0x3365737e))
            solids.append(Solid(meshes[xx],sm.fullAbsorb,sm.Vac,surface=sm.genericSurface,color=0x3365737e))
        elif 'source_surface' in allstls[xx]:
            # photons coming out from this window
            #solids.append(Solid(meshes[xx],sm.fullAbsorb,sm.Vac,surface=sm.genericSurface,color=0x3365737e))
            solids.append(Solid(meshes[xx],sm.fullAbsorb,sm.Vac,surface=sm.genericSurface,color=0x3365737e))
            trianglecenters = meshes[xx].get_triangle_centers()
            xPos = np.mean(trianglecenters[:,0])
            yPos = np.mean(trianglecenters[:,1])
            zPos = np.mean(trianglecenters[:,2])
            scenters.append([xPos + xshift,yPos + yshift,zPos + zshift])
        else:
            solids.append(Solid(meshes[xx],sm.fullAbsorb,sm.Vac,surface=sm.genericSurface,color=0x3365737e))

        setup.add_solid(solids[xx],rotation=rotation, displacement=global_shift)
    
print 'detector center position: ', dcenters
print 'source window center position: ', scenters

#rarr = np.array([rcenters[0][0],rcenters[0][1],rcenters[0][2]])
#mag_r = np.sqrt(rarr.dot(rarr))
#darr = np.array([0,0,0])#[dcenters[0][0],dcenters[0][1],dcenters[0][2]])
darr = np.array([0,1,0])
mag_d = np.sqrt(darr.dot(darr))
norm_z = np.array([0,0.,1])
norm_y = np.array([0,1,0])
#norm_y_15 = np.array([np.sin(-15.*np.pi/180.),-1.*np.cos(-15.*np.pi/180.),0])
norm_y_15 = np.array([0,-1.*np.cos(-15.*np.pi/180.),0])
##print rarr
print "Mesh Imports Successful"

setup.flatten()
#view (setup)
setup.bvh = load_bvh(setup)
sim = Simulation(setup, geant4_processes=0)


# photon bomb, isotropic from position center
def photon_bomb(n,wavelength,pos):
	pos = np.tile(pos,(n,1))
	direction = uniform_sphere(n)
	#direction = np.tile((-0.9999995,0.01,0),(n,1))
	pol = np.cross(direction,uniform_sphere(n))
	wavelengths = np.repeat(wavelength,n)
	return Photons(pos,direction,pol,wavelengths) 

def photon_gun(n,wavelength,pos,_dir):
	pos = np.tile(pos,(n,1))
	direction = np.tile(_dir,(n,1))
	pol = np.cross(direction,uniform_sphere(n))
	wavelengths = np.repeat(wavelength,n)
	return Photons(pos,direction,pol,wavelengths) 

def pb_z_axis(n,wavelength,zStart,zEnd):
	zPoint = zStart
	photons = []
	while zPoint < zEnd:
		zPoint += (zEnd-zStart)/10
		pos = np.tile([x_source,y_source,zPoint],(n,1))
		direction = uniform_sphere(n)
		pol = np.cross(direction, uniform_sphere(n))
		wavelengths = np.repeat(wavelength,n)
		photons.append(Photons(pos,direction,pol,wavelengths))
	return photons

def photon_uniform_circle(n,wavelength, pos, radius, _dir=None):

        rsq = np.random.uniform(0, radius*radius, n)
        r = np.sqrt(rsq)
	theta = np.random.uniform(0.,2.*np.pi,n)
        
        x = r*np.cos(theta)
        z = r*np.sin(theta)
        y = np.repeat(0, n)

	pos = (np.vstack((x+x_source,y+y_source,z+z_source))).T

	direction = uniform_sphere(n)
        if not _dir==None:
            direction = np.tile(_dir, (n, 1))
	pol = np.cross(direction,uniform_sphere(n))
	wavelengths = np.repeat(wavelength,n)
	return Photons(pos,direction,pol,wavelengths)

def photon_uniform_col(n,wavelength):
	s = np.random.uniform(0,1,n)
	theta = np.random.uniform(0.,2.*np.pi,n)
	z = np.random.uniform(0.0001,0.015,n) #fission fragment range in 3.0 g/cm3 xenon
	r = np.sqrt(s[:])*3.
	x = r[:]*np.cos(theta[:])
	y = r[:]*np.sin(theta[:])
	pos = (np.vstack((x+x_source,y+y_source,z+z_source))).T
	direction = uniform_sphere(n)
	pol = np.cross(direction,uniform_sphere(n))
	wavelengths = np.repeat(wavelength,n)
	return Photons(pos,direction,pol,wavelengths)

def photon_uniform_muon(n,wavelength):
	num_z = 100
	s = np.random.uniform(0,1,n/num_z)
	theta = np.random.uniform(0.,2.*np.pi,n/num_z)
	r = np.sqrt(s[:])*50.
	x = r[:]*np.cos(theta[:])
	y = r[:]*np.sin(theta[:])
	direction = uniform_sphere(n)
	pol = np.cross(direction,uniform_sphere(n))
	wavelengths = np.repeat(wavelength,n)

	pos = np.empty((0,3))
	for ii in xrange(num_z):	
		z_ii = 50.*(1.-float(ii)/float(num_z))
		z = np.repeat(z_ii,n/num_z)
		pos = np.append(pos,(np.vstack((x+xshift,y+yshift,z+zshift))).T,axis=0)

	return Photons(pos,direction,pol,wavelengths)

	
	




#Simulation stuff***************************************************************************************************************************
evt[0]=0
#numPhotons = 6000000 #90 MeV fragment, 15 eV per photon
#numPhotons = 800000 #1.3 MeV*cm2/g, 3 g/cm3, 23.7 eV per photon, 5 cm, round off
numPhotons = 10000
if nphotons!=None:
    numPhotons = nphotons
photonsource = photon_uniform_circle(numPhotons,178,(x_source,y_source,z_source), 22)
#photonsource = pb_z_axis(numPhotons,178,z_source,0.5)
#photonsource = photon_uniform_col(numPhotons,178)
#photonsource = photon_gun(numPhotons,178,(x_source,y_source,z_source),norm_y)
#photonsource = photon_uniform_muon(numPhotons,178)
radius = 5.8 if y_source<-110 else 22
if source_type=='area_iso':
    photonsource = photon_uniform_circle(numPhotons,178,(x_source,y_source,z_source), radius)
if source_type=='area_norm':
    photonsource = photon_uniform_circle(numPhotons,178,(x_source,y_source,z_source), radius, norm_y)
if source_type=='point_iso':
    photonsource = photon_bomb(numPhotons,178,(x_source,y_source,z_source))



for ev in sim.simulate(photonsource, keep_photons_beg=True,keep_photons_end=True, run_daq=True,max_steps=500):
	
	detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)#Detection condition
        #detected = np.ones(numPhotons, dtype=bool)
	numDetected = len(ev.photons_end[detected])#Number of detected photons
	print numPhotons
	print numDetected
	
	#Store various configurations of detected photons
	startPos = ev.photons_beg.pos[detected]
	startDir = ev.photons_beg.dir[detected]
        startTime = ev.photons_beg.t[detected]
        endPos = ev.photons_end.pos[detected]
	endDir = ev.photons_end.dir[detected]
        endTime = ev.photons_end.t[detected]
	endFlags = ev.photons_end.flags[detected]
	
	n[0] = numDetected
	for i in range(n[0]):
		z[i] = endPos[i][2] 
		x[i] = endPos[i][0] 
		y[i] = endPos[i][1] 
                t[i] = endTime[i]
		zi[i] = startPos[i][2] 
		xi[i] = startPos[i][0] 
		yi[i] = startPos[i][1] 
                ti[i] = startTime[i]
		flags[i] = endFlags[i]
        
	
                #print endPos[i], np.dot(endDir[i], endDir[i])

		projDir = endDir[i] - np.dot(endDir[i],norm_z)*norm_z
                projDir = endDir[i]

		uv_det = np.dot(darr, endDir[i])
		uv_ref = np.dot(norm_y,startDir[i])
		mag_p = np.sqrt(projDir.dot(projDir))
		cos_d = uv_det/mag_d/mag_p
		cos_r = uv_ref/1./mag_p
		ad[i] = np.arccos(cos_d)*180./np.pi
		ar[i] = np.arccos(cos_r)*180./np.pi
	
	p[0] = numPhotons
	d[0] = numDetected

	#evt[0]=int(iter)

	tree.Fill()
		
	print "Num Detected:", numDetected
	print "Detection efficiency, %:", float(float(numDetected)*100.0/float(numPhotons))
	print 

tree.Write()
file.Close()
