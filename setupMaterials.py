from chroma.geometry import Material, Solid, Surface
import numpy as np

# Materials
copper= Material('copper')
copper.set('refractive_index', 1.)#3)
copper.set('absorption_length', 1e-16)
copper.set('scattering_length',1e6)
copper.density = 8.96
copper.composition = {'Cu' : 1.00}

LXe = Material('LXe')
LXe.set('refractive_index', 1.)#7)
LXe.set('absorption_length', 20000000)
LXe.set('scattering_length', 400)

Vac = Material('Vac')
Vac.set('refractive_index', 1.)#7)
Vac.set('absorption_length', 1e9)
Vac.set('scattering_length', 1e9)

fullAbsorb= Material('fullAbsorb')
fullAbsorb.set('refractive_index', 1.)#5)
fullAbsorb.set('absorption_length', 1.e-16)
fullAbsorb.set('scattering_length', 1.e6)

# Surfaces
#generic
genericSurface = Surface('genericSurface')
genericSurface.set('detect', 0.)
genericSurface.set('reflect_specular', 0.)
genericSurface.set('reflect_diffuse',0.0)
genericSurface.set('absorb', 1.)

teflonSurface = Surface('teflonSurface')
teflonSurface.set('absorb', 0.3)
teflonSurface.set('reflect_diffuse', 0.65)
teflonSurface.set('reflect_specular', 0.05)

copperSurface = Surface('copperSurface')
copperSurface.set('absorb', 0.3)
copperSurface.set('reflect_diffuse', 0.25)
copperSurface.set('reflect_specular',0.45)

dark_copperSurface = Surface('dark_copperSurface')
dark_copperSurface.set('absorb', 1.)
dark_copperSurface.set('reflect_diffuse', 0.)
dark_copperSurface.set('reflect_specular',0.)

rough2_copperSurface = Surface('rough2_copperSurface')
rough2_copperSurface.set('absorb', 0.6)
rough2_copperSurface.set('reflect_diffuse', 0.2)
rough2_copperSurface.set('reflect_specular',0.2)

#SiPM
SiPMSurface = Surface('SiPMSurface')
SiPMSurface.set('detect', 0.15)
SiPMSurface.set('reflect_specular', 0.60)
SiPMSurface.set('reflect_diffuse',0.0)
SiPMSurface.set('absorb', 0.25)

#deadSiPM
deadSiPMSurface = Surface('deadSiPMSurface')
deadSiPMSurface.set('detect', 0.)
deadSiPMSurface.set('reflect_specular', 0.60)
deadSiPMSurface.set('reflect_diffuse',0.05)
deadSiPMSurface.set('absorb', 0.35)

#idealSiPM
iSiPMSurface = Surface('iSiPMSurface')
iSiPMSurface.set('detect', 1.)
iSiPMSurface.set('reflect_specular', 0.)
iSiPMSurface.set('reflect_diffuse',0.0)
iSiPMSurface.set('absorb', 0.)

# transmission surface
transSurface = Surface('transSurface')
transSurface.set('detect', 0.)
transSurface.set('reflect_specular', 0.)
transSurface.set('reflect_diffuse',0.0)
transSurface.set('absorb', 0.)
