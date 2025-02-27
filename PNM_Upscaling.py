import pickle
import glob
from PIL import Image
import numpy as np
import openpnm as op
import porespy as pp
import connectNetworksUpscale
import openpnm.models.physics as mods
from Relperm import RelativePermeability
import time
import pandas as pd
import tensorflow as tf

start_time = time.time()

#---------------------------------------------------------
load_files = False
load_networks = False
#---------------------------------------------------------
pickle_name = "pickle_name"
#---------------------------------------------------------
minThroatLength = 1.0974724914186572e-05 
maxThroatLength = 0.0011850426099862402  
#---------------------------------------------------------
highResolution = 1.28*1e-6 #micrometer
#---------------------------------------------------------
factor = 2 #Resolution difference factor between LR and HR image
#---------------------------------------------------------
output_path = "./ouput_path_folder/"
load_path = "./load_path_folder/"
#---------------------------------------------------------
vtkName_before = "name_before.vtk"
vtkName_after = "name_after.vtk"
#---------------------------------------------------------
ml_output_path = './Outputs/'
pickle_path = './ML_output/'
#---------------------------------------------------------
lowpath = './path_to_LR_image_folder/'
#--------------------------------------------------------

# Load trained ANN model
ann_model = tf.keras.models.load_model(pickle_path + 'trained_ann_model.h5')

#---------------------------------------------------------
# Percolation method (Drainage)
method = "Ordinary Percolation" #"Invasion Percolation" and "Mixed Invasion Percolation"

# Residual gas saturation = critical gas saturation
residual = False 
residual_value = 0

# It's a boolean parameter and it calculates the Swc
trapping = False

firstPoint = np.load(load_path+"firstPointOct.npy", allow_pickle=True)
neighbors = np.load(load_path+"neighbors.npy", allow_pickle=True)
neighbors = np.ndarray.tolist(neighbors)
highResImages = np.load(load_path+"highresImages.npy", allow_pickle=True)
highResImages = np.ndarray.tolist(highResImages)
connectedRegions = np.load(load_path+"connectedRegions.npy", allow_pickle=True)
connectedRegions = np.ndarray.tolist(connectedRegions)
clusterFirstPoints = np.load(load_path+"clusterFirstPoints.npy", allow_pickle=True)
clusterCubes = np.load(load_path+"clusterCubes.npy", allow_pickle=True)
clusterCubes = np.ndarray.tolist(clusterCubes)
clusterPhases = np.load(load_path+"clusterPhases.npy", allow_pickle=True)
clusterPhases = np.ndarray.tolist(clusterPhases)
clusterHighResImages = np.load(load_path+"clusterHighResImages.npy", allow_pickle=True)
clusterHighResImages = np.ndarray.tolist(clusterHighResImages) 
#---------------------------------------------------------

image_format = '*.tif'

# Reading image paths
imgs_path = glob.glob(lowpath+image_format)
imgs_path.sort()

I = Image.open(imgs_path[0])
I = I.convert("L")
width, height = np.shape(I)
depth = len(imgs_path)

if load_files:    
    #----------------------------------------------------------------
    network = op.io.Pickle.load_project(pickle_path + 'networks_' + pickle_name + '.pkl')[0]
else:
    # Loading images
    lowResImage = np.zeros(shape=[width,height,depth])
    count = 0
    for i in imgs_path:
        I = Image.open(i)
        I = I.convert("L")
        lowResImage[:,:,count] = I
        count += 1
    
    lowResImage = np.array(lowResImage, dtype=bool)
    lowResImage = ~lowResImage
    
    # Extract the main network from the low lowResolution image
    pn = pp.networks.snow2(lowResImage, voxel_size=highResolution*factor)
    network = None
    network = op.network.GenericNetwork()
    network.update(pn.network)

    # Export VTK
    op.io.VTK.export_data(network=network, filename=vtkName_before)
    
    op.io.Pickle.save_project(network.project, pickle_path + 'networks_' + pickle_name + '.pkl')

if load_networks:
    with open(load_path+'highResNetworks.pkl', 'rb') as file:
        highResNetworks = pickle.load(file)
    
    for counter, net in enumerate(highResNetworks):
        if (net == False):
            continue
        else:
            highResNetworks[counter] = op.io.Pickle.load_project(output_path + 'highResNetworks/highNet' + str(counter) + '.pkl')[0]
            
else:
    # Extract network from the highResolution images and add them to the main network
    highResNetworks = []
    highResExistedNetworks = []
    
    for counter, image in enumerate(clusterHighResImages):
        print(counter)
        try:
            image = ~image
            unresolvedpn = pp.networks.snow2(image, voxel_size=highResolution)
            unresolvedNetwork = None
            unresolvedNetwork = op.network.GenericNetwork()
            unresolvedNetwork.update(unresolvedpn.network)
            unresolvedNetwork['pore.coords'] += [clusterFirstPoints[counter][0]*highResolution,
                                                 clusterFirstPoints[counter][1]*highResolution,
                                                 clusterFirstPoints[counter][2]*highResolution]
            op.io.Pickle.save_project(unresolvedNetwork.project, output_path + 'highResNetworks/highNet' + str(counter) + '.pkl')
            highResNetworks.append(unresolvedNetwork)
            highResExistedNetworks.append(unresolvedNetwork)
        except:
            highResNetworks.append(False)

    with open(output_path+'highResNetworks.pkl', 'wb') as file:
        pickle.dump(highResNetworks, file)

poreNumsBefore = len(network.Ps)
throatNumsBefore = len(network.Ts)

start_time = time.time()
highResNetwork, geometry, phase_inv, phase_def, phys_inv, phys_def,\
microThroats, macroThroats = connectNetworksUpscale.connect(
    network, 
    highResNetworks, 
    neighbors,
    connectedRegions,
    minThroatLength,
    ann_model,     
    maxThroatLength
)

elapsedTime = time.time() - start_time

#Export vtk
op.io.VTK.export_data(network=highResNetwork, filename=vtkName_after)

poreNums = len(highResNetwork.Ps)
throatNums = len(highResNetwork.Ts)

h = highResNetwork.check_network_health()
isolated_pores = len(h['isolated_pores'])
op.topotools.trim(network=highResNetwork, pores=h['trim_pores'])

op.topotools.find_surface_pores(highResNetwork)
op.topotools.label_faces(highResNetwork)

Lx = width*highResolution*factor
Ly = height*highResolution*factor
Lz = depth*highResolution*factor

pnVolume= np.sum(highResNetwork['pore.volume']) + np.sum(highResNetwork['throat.volume'])
porosity=((pnVolume)/(Lx*Ly*Lz))*100

for i in highResNetwork.props():
    if i not in ['throat.conns','pore.coords']:
        geometry.update({i:highResNetwork.pop(i)})

A_x=Lz*Ly
A_y=Lx*Lz
A_z=Lx*Ly

SF=op.algorithms.StokesFlow(network=highResNetwork,phase=phase_inv)
SF.set_value_BC(pores=highResNetwork.pores('back'),values=200000)
SF.set_value_BC(pores=highResNetwork.pores('front'),values=100000)
SF.run()
Q_tot_x=np.absolute(SF.rate(pores=highResNetwork.pores('back')))

SF1=op.algorithms.StokesFlow(network=highResNetwork,phase=phase_inv)
SF1.set_value_BC(pores=highResNetwork.pores('left'),values=200000)
SF1.set_value_BC(pores=highResNetwork.pores('right'),values=100000)
SF1.run()
Q_tot_y=np.absolute(SF1.rate(pores=highResNetwork.pores('left')))

SF2=op.algorithms.StokesFlow(network=highResNetwork,phase=phase_inv)
SF2.set_value_BC(pores=highResNetwork.pores('bottom'),values=200000)
SF2.set_value_BC(pores=highResNetwork.pores('top'),values=100000)
SF2.run()
Q_tot_z=np.absolute(SF2.rate(pores=highResNetwork.pores('bottom')))
    
mu = np.mean(phase_inv['pore.viscosity'])
delta_P = 100000

K1 = float(Q_tot_x*mu*Lx/(delta_P*A_x))/(0.97*10**-15)
K2 = float(Q_tot_y*mu*Ly/(delta_P*A_y))/(0.97*10**-15)
K3 = float(Q_tot_z*mu*Lz/(delta_P*A_z))/(0.97*10**-15)
K_tot = (K1+K2+K3)/3

coordinationNum = highResNetwork.num_neighbors(pores=highResNetwork.Ps, flatten=False)

file1 = open(ml_output_path+'FinalResult', 'w')
L = [f"Micro throats = {microThroats} \n", f"Macro throats = {macroThroats} \n", f"Pore numbers = {poreNums} \n", \
     f"Throat numbers = {throatNums} \n", f"Porosity = {porosity} \n", \
     f"Permeability_X = {K1} \n", f"Permeability_Y = {K2} \n", \
     f"Permeability_Z = {K3} \n", f"Permeability_total = {K_tot} \n"]

file1.writelines(L)

file1.close()

if method == "Ordinary Percolation":
    if residual:
        Np=network.Np
        res_sat = eval(residual_value)
        pNum = int(np.floor(res_sat * Np))
        res_pores = []
        for i in range (pNum):
            res_pores.append(i)
    
    OP=op.algorithms.OrdinaryPercolation(network=network,phase=phase_inv)
    OP.setup(phase=phase_inv, pore_volume='pore.volume', throat_volume='throat.volume')
    OP.set_inlets(pores=network.pores('front'))
    if residual:
        OP.set_residual(pores=res_pores)
    OP.run()
    phase_inv.update(OP.results(Pc=0))
    pcdata = OP.get_intrusion_data()
    
    PC_data = {}
    
    PC_data['PCx'] = np.array(pcdata[0])
    PC_data['SWx'] = 1-np.array(pcdata[1])
    
    PC_data['PCx'] = [float(i) for i in PC_data['PCx']]
    PC_data['SWx'] = [float(i) for i in PC_data['SWx']]
    
    
    OP1=op.algorithms.OrdinaryPercolation(network=network,phase=phase_inv)
    OP1.setup(phase=phase_inv, pore_volume='pore.volume', throat_volume='throat.volume')
    OP1.set_inlets(pores=network.pores('left'))
    if residual:
        OP1.set_residual(pores=res_pores)
    OP1.run()
    phase_inv.update(OP1.results(Pc=0))
    pcdata1 = OP1.get_intrusion_data()
    
    PC_data['PCy'] = np.array(pcdata1[0])
    PC_data['SWy'] = 1-np.array(pcdata1[1])
    
    PC_data['PCy'] = [float(i) for i in PC_data['PCy']]
    PC_data['SWy'] = [float(i) for i in PC_data['SWy']]
    
    OP2=op.algorithms.OrdinaryPercolation(network=network,phase=phase_inv)
    OP2.setup(phase=phase_inv, pore_volume='pore.volume', throat_volume='throat.volume')
    OP2.set_inlets(pores=network.pores('top'))
    if residual:
        OP2.set_residual(pores=res_pores)
    OP2.run()
    phase_inv.update(OP2.results(Pc=0))
    pcdata2 = OP2.get_intrusion_data()
    
    PC_data['PCz'] = np.array(pcdata2[0])
    PC_data['SWz'] = 1-np.array(pcdata2[1])
    
    PC_data['PCz'] = [float(i) for i in PC_data['PCz']]
    PC_data['SWz'] = [float(i) for i in PC_data['SWz']]
    
    phys_inv.add_model(model=mods.multiphase.conduit_conductance,
                       propname='throat.conduit_hydraulic_conductance',
                       throat_conductance='throat.hydraulic_conductance')
    phys_def.add_model(model=mods.multiphase.conduit_conductance,
                       propname='throat.conduit_hydraulic_conductance',
                       throat_conductance='throat.hydraulic_conductance')
    
    A = [A_x,A_y,A_z]
    Length = [Lx,Ly,Lz]
    bounds = [['front', 'back'], ['left', 'right'], ['top', 'bottom']]
    perm_air = {'0': [], '1': [], '2': []}
    perm_water = {'0': [], '1': [], '2': []}
    diff_air = {'0': [], '1': [], '2': []}
    diff_water = {'0': [], '1': [], '2': []}
    sat= []
    tot_vol = np.sum(network["pore.volume"]) + np.sum(network["throat.volume"])
    i = 1
    t = -1
    for Pc in np.unique(OP['pore.invasion_pressure']):
        t += 1
        phase_inv.update(OP.results(Pc=Pc))
        phase_def['pore.occupancy'] = 1 - phase_inv['pore.occupancy']
        phase_def['throat.occupancy'] = 1 - phase_inv['throat.occupancy']
        phys_inv.regenerate_models()
        phys_def.regenerate_models()
        this_sat = 0
        this_sat += np.sum(network["pore.volume"][phase_inv["pore.occupancy"] == 1])
        this_sat += np.sum(network["throat.volume"][phase_inv["throat.occupancy"] == 1])
        sat.append(this_sat)
        i = 0
        for bound_increment in range(len(bounds)):
            BC1_pores = network.pores(labels=bounds[bound_increment][0])
            BC2_pores = network.pores(labels=bounds[bound_increment][1])
            ST_1 = op.algorithms.StokesFlow(network=network)
            ST_1.setup(phase=phase_inv, conductance='throat.conduit_hydraulic_conductance')
            ST_1.set_value_BC(values=0.6, pores=BC1_pores)
            ST_1.set_value_BC(values=0.2, pores=BC2_pores)
            ST_1.run()
            eff_perm = ST_1.calc_effective_permeability(domain_area=A[i], domain_length=Length[i])
            perm_air[str(bound_increment)].append(eff_perm)
            ST_2 = op.algorithms.StokesFlow(network=network)
            ST_2.setup(phase=phase_def, conductance='throat.conduit_hydraulic_conductance')
            ST_2.set_value_BC(values=0.6, pores=BC1_pores)
            ST_2.set_value_BC(values=0.2, pores=BC2_pores)
            ST_2.run()
            eff_perm = ST_2.calc_effective_permeability(domain_area=A[i], domain_length=Length[i])
            perm_water[str(bound_increment)].append(eff_perm)
            network.project.purge_object(ST_1)
            network.project.purge_object(ST_2)
            i = i + 1
    
    sat = np.asarray(sat)
    sat /= tot_vol
    rel_perm_air_x    =  np.asarray(perm_air['0'])
    rel_perm_air_x   =  (rel_perm_air_x/rel_perm_air_x[0]) * 1e-6
    rel_perm_air_y    =  np.asarray(perm_air['1'])
    rel_perm_air_y   =  (rel_perm_air_y/rel_perm_air_y[0]) * 1e-6
    rel_perm_air_z    =  np.asarray(perm_air['2'])
    rel_perm_air_z   =  (rel_perm_air_z/rel_perm_air_z[0]) * 1e-6
    rel_perm_water_x  =  np.asarray(perm_water['0'])
    rel_perm_water_x =  (rel_perm_water_x/rel_perm_water_x[-1]) * 1e-6
    rel_perm_water_y  =  np.asarray(perm_water['1'])
    rel_perm_water_y =  (rel_perm_water_y/rel_perm_water_y[-1]) * 1e-6
    rel_perm_water_z  =  np.asarray(perm_water['2'])
    rel_perm_water_z =  (rel_perm_water_z/rel_perm_water_z[-1]) * 1e-6
    
    Krdata = {}
    Krdata['Swx'] = sat
    Krdata['Swy'] = sat
    Krdata['Swz'] = sat
    Krdata['Krwx'] = rel_perm_air_x
    Krdata['Krnwx'] = rel_perm_water_x
    Krdata['Krwy'] = rel_perm_air_y
    Krdata['Krnwy'] = rel_perm_water_y
    Krdata['Krwz'] = rel_perm_air_z
    Krdata['Krnwz'] = rel_perm_water_z
    
    Krdata['Swx'] = [float(i) for i in Krdata['Swx']]
    Krdata['Swy'] = [float(i) for i in Krdata['Swy']]
    Krdata['Swz'] = [float(i) for i in Krdata['Swz']]
    Krdata['Krwx'] = [float(i) for i in Krdata['Krwx']]
    Krdata['Krnwx'] = [float(i) for i in Krdata['Krnwx']]
    Krdata['Krwy'] = [float(i) for i in Krdata['Krwy']]
    Krdata['Krnwy'] = [float(i) for i in Krdata['Krnwy']]
    Krdata['Krwz'] = [float(i) for i in Krdata['Krwz']]
    Krdata['Krnwz'] = [float(i) for i in Krdata['Krnwz']]

elif (method == "Invasion Percolation"):
    if (residual):
        Np=network.Np
        res_sat = eval(residual_value)
        pNum = int(np.floor(res_sat * Np))
        res_pores = []
        for i in range (pNum):
            res_pores.append(i)
    
    ip = op.algorithms.InvasionPercolation(network=network)
    ip.setup(phase=phase_inv)
    Finlets_init = network.pores('front')
    Finlets=([Finlets_init[x] for x in range(0, len(Finlets_init))])
    ip.set_inlets(pores=Finlets)
    ip.run()
    if (trapping):
        ip.apply_trapping(outlets=network.pores('back'))
    inv_points = np.arange(0, 100, 1)
    phase_inv.update(ip.results())
    data = ip.get_intrusion_data()
    
    rp = RelativePermeability(network=network, invading_phase=phase_inv.name, defending_phase=phase_def.name, invasion_sequence='invasion_sequence',
                              flow_inlets={'x': 'front'}, flow_outlets={'x': 'back'})
    rp.run()
    data1_kr = rp.get_Kr_data()
    Krdata = {}
    Krdata['Swx'] = 1 - np.array(data1_kr['sat']['x'])
    Krdata['Krwx'] = np.array(data1_kr['relperm_wp']['x'])
    Krdata['Krnwx'] = np.array(data1_kr['relperm_nwp']['x'])
    
    ##################################################################
    ip1 = op.algorithms.InvasionPercolation(network=network)
    ip1.setup(phase=phase_inv)
    Finlets_init = network.pores('left')
    Finlets=([Finlets_init[x] for x in range(0, len(Finlets_init))])
    ip1.set_inlets(pores=Finlets)
    ip1.run()
    if (trapping):
        ip1.apply_trapping(outlets=network.pores('right'))
    inv_points = np.arange(0, 100, 1)
    phase_inv.update(ip1.results())
    data1 = ip1.get_intrusion_data()
    
    rp = RelativePermeability(network=network, invading_phase=phase_inv.name, defending_phase=phase_def.name, invasion_sequence='invasion_sequence',
                              flow_inlets={'y': 'left'}, flow_outlets={'y': 'right'})
    rp.run()
    data2_kr = rp.get_Kr_data()
    
    Krdata['Swy'] = 1 - np.array(data2_kr['sat']['y'])
    Krdata['Krwy'] = np.array(data2_kr['relperm_wp']['y'])
    Krdata['Krnwy'] = np.array(data2_kr['relperm_nwp']['y'])

    ###################################################################
    ip2 = op.algorithms.InvasionPercolation(network=network)
    ip2.setup(phase=phase_inv)
    Finlets_init = network.pores('top')
    Finlets=([Finlets_init[x] for x in range(0, len(Finlets_init))])
    ip2.set_inlets(pores=Finlets)
    ip2.run()
    if (trapping):
        ip2.apply_trapping(outlets=network.pores('bottom'))
    inv_points = np.arange(0, 100, 1)
    phase_inv.update(ip2.results())
    data2 = ip2.get_intrusion_data()
    
    rp = RelativePermeability(network=network, invading_phase=phase_inv.name, defending_phase=phase_def.name, invasion_sequence='invasion_sequence',
                              flow_inlets={'z': 'top'}, flow_outlets={'z': 'bottom'})
    rp.run()
    data3_kr = rp.get_Kr_data()
    
    Krdata['Swz'] = 1 - np.array(data3_kr['sat']['z'])
    Krdata['Krwz'] = np.array(data3_kr['relperm_wp']['z'])
    Krdata['Krnwz'] = np.array(data3_kr['relperm_nwp']['z'])
    
    PC_data = {}
    PC_data['PCx'] = data.Pcap
    PC_data['SWx'] = 1-data.S_tot
    PC_data['PCy'] = data1.Pcap
    PC_data['SWy'] = 1-data1.S_tot
    PC_data['PCz'] = data2.Pcap
    PC_data['SWz'] = 1-data2.S_tot
    
    PC_data['PCx'] = [float(i) for i in PC_data['PCx']]
    PC_data['SWx'] = [float(i) for i in PC_data['SWx']]
    
    PC_data['PCy'] = [float(i) for i in PC_data['PCy']]
    PC_data['SWy'] = [float(i) for i in PC_data['SWy']]
    
    PC_data['PCz'] = [float(i) for i in PC_data['PCz']]
    PC_data['SWz'] = [float(i) for i in PC_data['SWz']]
    
    Krdata['Swx'] = [float(i) for i in Krdata['Swx']]
    Krdata['Swy'] = [float(i) for i in Krdata['Swy']]
    Krdata['Swz'] = [float(i) for i in Krdata['Swz']]
    Krdata['Krwx'] = [float(i) for i in Krdata['Krwx']]
    Krdata['Krnwx'] = [float(i) for i in Krdata['Krnwx']]
    Krdata['Krwy'] = [float(i) for i in Krdata['Krwy']]
    Krdata['Krnwy'] = [float(i) for i in Krdata['Krnwy']]
    Krdata['Krwz'] = [float(i) for i in Krdata['Krwz']]
    Krdata['Krnwz'] = [float(i) for i in Krdata['Krnwz']]
    
    ###################################################################
elif (method == "Mixed Invasion Percolation"):
    if (residual):
        Np=network.Np
        res_sat = eval(residual_value)
        pNum = int(np.floor(res_sat * Np))
        res_pores = []
        for i in range (pNum):
            res_pores.append(i)
    
    phys_inv.add_model(propname='pore.entry_pressure',
                       model = mods.capillary_pressure.washburn, diameter="pore.diameter")
    phys_inv.regenerate_models()
    phys_def.add_model(propname='pore.entry_pressure',
                       model = mods.capillary_pressure.washburn, diameter="pore.diameter")
    phys_def.regenerate_models()
    
    ip = op.algorithms.MixedInvasionPercolation(network=network)
    ip.setup(phase=phase_inv, pore_entry_pressure='pore.entry_pressure')
    Finlets_init = network.pores('front')
    Finlets=([Finlets_init[x] for x in range(0, len(Finlets_init))])
    ip.set_inlets(pores=Finlets)
    if (residual):
        ip.set_residual(pores=res_pores)
    ip.run()
    if (trapping):
        ip.set_outlets(network.pores('back'))
        ip.apply_trapping()
    inv_points = np.arange(0, 100, 1)
    phase_inv.update(ip.results(Pc=inv_points.max()))
    data = ip.get_intrusion_data()

    ################################################################################
    ip1 = op.algorithms.MixedInvasionPercolation(network=network)
    ip1.setup(phase=phase_inv, pore_entry_pressure='pore.entry_pressure')
    Finlets_init = network.pores('left')
    Finlets=([Finlets_init[x] for x in range(0, len(Finlets_init))])
    ip1.set_inlets(pores=Finlets)
    if (residual):
        ip1.set_residual(pores=res_pores)
    ip1.run()
    if (trapping):
        ip1.set_outlets(network.pores('right'))
        ip1.apply_trapping()
    inv_points = np.arange(0, 100, 1)
    phase_inv.update(ip1.results(Pc=inv_points.max()))
    data1 = ip1.get_intrusion_data()
    
    ################################################################################
    ip2 = op.algorithms.MixedInvasionPercolation(network=network)
    ip2.setup(phase=phase_inv, pore_entry_pressure='pore.entry_pressure')
    Finlets_init = network.pores('top')
    Finlets=([Finlets_init[x] for x in range(0, len(Finlets_init))])
    ip2.set_inlets(pores=Finlets)
    if (residual):
        ip2.set_residual(pores=res_pores)
    ip2.run()
    if (trapping):
        ip2.set_outlets(network.pores('bottom'))
        ip2.apply_trapping()
    inv_points = np.arange(0, 100, 1)
    phase_inv.update(ip2.results(Pc=inv_points.max()))
    data2 = ip2.get_intrusion_data()
    
    PC_data = {}
    PC_data['PCx'] = data.Pcap
    PC_data['SWx'] = 1-data.S_tot
    PC_data['PCy'] = data1.Pcap
    PC_data['SWy'] = 1-data1.S_tot
    PC_data['PCz'] = data2.Pcap
    PC_data['SWz'] = 1-data2.S_tot
    PC_data['PCx'] = [float(i) for i in PC_data['PCx']]
    PC_data['SWx'] = [float(i) for i in PC_data['SWx']]
    
    PC_data['PCy'] = [float(i) for i in PC_data['PCy']]
    PC_data['SWy'] = [float(i) for i in PC_data['SWy']]
    
    PC_data['PCz'] = [float(i) for i in PC_data['PCz']]
    PC_data['SWz'] = [float(i) for i in PC_data['SWz']]
    ################################################################################
    
    phys_inv.add_model(model=mods.multiphase.conduit_conductance,
                       propname='throat.conduit_hydraulic_conductance',
                       throat_conductance='throat.hydraulic_conductance')
    phys_def.add_model(model=mods.multiphase.conduit_conductance,
                        propname='throat.conduit_hydraulic_conductance',
                        throat_conductance='throat.hydraulic_conductance')
    
    A = [A_x,A_y,A_z]
    Length = [Lx,Ly,Lz]
    bounds = [['front', 'back'], ['left', 'right'], ['top', 'bottom']]
    perm_air = {'0': [], '1': [], '2': []}
    perm_water = {'0': [], '1': [], '2': []}
    satx= []
    saty = []
    satz = []
    
    tot_vol = np.sum(network["pore.volume"]) + np.sum(network["throat.volume"])
    
    i = 0
    Pc = np.unique(ip['throat.invasion_pressure'])
    for j in range(1, len(np.unique(ip['throat.invasion_pressure']))):
        res = ip.results(Pc=Pc[j])
        up_data = {
            'pore.occupancy': np.array(res['pore.occupancy'], dtype=bool),
            'throat.occupancy': np.array(res['throat.occupancy'], dtype=bool)
            }
        phase_inv.update(up_data)
        phase_def['pore.occupancy'] = ~phase_inv['pore.occupancy']
        phase_def['throat.occupancy'] = ~phase_inv['throat.occupancy']
        phys_inv.regenerate_models()
        phys_def.regenerate_models()
        this_sat = 0
        this_sat += np.sum(network["pore.volume"][phase_inv["pore.occupancy"] == 1])
        this_sat += np.sum(network["throat.volume"][phase_inv["throat.occupancy"] == 1])
        satx.append(this_sat)
        BC1_pores = network.pores(labels=bounds[i][0])
        BC2_pores = network.pores(labels=bounds[i][1])
        ST_1 = op.algorithms.StokesFlow(network=network)
        ST_1.setup(phase=phase_inv, conductance='throat.conduit_hydraulic_conductance')
        ST_1.set_value_BC(values=0.6, pores=BC1_pores)
        ST_1.set_value_BC(values=0.2, pores=BC2_pores)
        ST_1.run()
        eff_perm = ST_1.calc_effective_permeability(domain_area=A[i], domain_length=Length[i])
        perm_air[str(i)].append(eff_perm)
        ST_2 = op.algorithms.StokesFlow(network=network)
        ST_2.setup(phase=phase_def, conductance='throat.conduit_hydraulic_conductance')
        ST_2.set_value_BC(values=0.6, pores=BC1_pores)
        ST_2.set_value_BC(values=0.2, pores=BC2_pores)
        ST_2.run()
        eff_perm = ST_2.calc_effective_permeability(domain_area=A[i], domain_length=Length[i])
        perm_water[str(i)].append(eff_perm)
        network.project.purge_object(ST_1)
        network.project.purge_object(ST_2)
    
    i=1
    Pc = np.unique(ip1['throat.invasion_pressure'])
    for j in range(1, len(np.unique(ip1['throat.invasion_pressure']))):
        res = ip1.results(Pc=Pc[j])
        up_data = {
            'pore.occupancy': np.array(res['pore.occupancy'], dtype=bool),
            'throat.occupancy': np.array(res['throat.occupancy'], dtype=bool)
            }
        phase_inv.update(up_data)
        phase_def['pore.occupancy'] = ~phase_inv['pore.occupancy']
        phase_def['throat.occupancy'] = ~phase_inv['throat.occupancy']
        phys_inv.regenerate_models()
        phys_def.regenerate_models()
        this_sat = 0
        this_sat += np.sum(network["pore.volume"][phase_inv["pore.occupancy"] == 1])
        this_sat += np.sum(network["throat.volume"][phase_inv["throat.occupancy"] == 1])
        saty.append(this_sat)
        BC1_pores = network.pores(labels=bounds[i][0])
        BC2_pores = network.pores(labels=bounds[i][1])
        ST_1 = op.algorithms.StokesFlow(network=network)
        ST_1.setup(phase=phase_inv, conductance='throat.conduit_hydraulic_conductance')
        ST_1.set_value_BC(values=0.6, pores=BC1_pores)
        ST_1.set_value_BC(values=0.2, pores=BC2_pores)
        ST_1.run()
        eff_perm = ST_1.calc_effective_permeability(domain_area=A[i], domain_length=Length[i])
        perm_air[str(i)].append(eff_perm)
        ST_2 = op.algorithms.StokesFlow(network=network)
        ST_2.setup(phase=phase_def, conductance='throat.conduit_hydraulic_conductance')
        ST_2.set_value_BC(values=0.6, pores=BC1_pores)
        ST_2.set_value_BC(values=0.2, pores=BC2_pores)
        ST_2.run()
        eff_perm = ST_2.calc_effective_permeability(domain_area=A[i], domain_length=Length[i])
        perm_water[str(i)].append(eff_perm)
        network.project.purge_object(ST_1)
        network.project.purge_object(ST_2)
    
    i = 2
    Pc = np.unique(ip2['throat.invasion_pressure'])
    for j in range(1, len(np.unique(ip2['throat.invasion_pressure']))):
        res = ip2.results(Pc=Pc[j])
        up_data = {
            'pore.occupancy': np.array(res['pore.occupancy'], dtype=bool),
            'throat.occupancy': np.array(res['throat.occupancy'], dtype=bool)
            }
        phase_inv.update(up_data)
        phase_def['pore.occupancy'] = ~phase_inv['pore.occupancy']
        phase_def['throat.occupancy'] = ~phase_inv['throat.occupancy']
        phys_inv.regenerate_models()
        phys_def.regenerate_models()
        this_sat = 0
        this_sat += np.sum(network["pore.volume"][phase_inv["pore.occupancy"] == 1])
        this_sat += np.sum(network["throat.volume"][phase_inv["throat.occupancy"] == 1])
        satz.append(this_sat)
        BC1_pores = network.pores(labels=bounds[i][0])
        BC2_pores = network.pores(labels=bounds[i][1])
        ST_1 = op.algorithms.StokesFlow(network=network)
        ST_1.setup(phase=phase_inv, conductance='throat.conduit_hydraulic_conductance')
        ST_1.set_value_BC(values=0.6, pores=BC1_pores)
        ST_1.set_value_BC(values=0.2, pores=BC2_pores)
        ST_1.run()
        eff_perm = ST_1.calc_effective_permeability(domain_area=A[i], domain_length=Length[i])
        perm_air[str(i)].append(eff_perm)
        ST_2 = op.algorithms.StokesFlow(network=network)
        ST_2.setup(phase=phase_def, conductance='throat.conduit_hydraulic_conductance')
        ST_2.set_value_BC(values=0.6, pores=BC1_pores)
        ST_2.set_value_BC(values=0.2, pores=BC2_pores)
        ST_2.run()
        eff_perm = ST_2.calc_effective_permeability(domain_area=A[i], domain_length=Length[i])
        perm_water[str(i)].append(eff_perm)
        network.project.purge_object(ST_1)
        network.project.purge_object(ST_2)
    
    
    satx = np.asarray(satx)
    satx /= tot_vol
    
    saty = np.asarray(saty)
    saty /= tot_vol
    
    satz = np.asarray(satz)
    satz /= tot_vol
    
    rel_perm_air_x    =  np.asarray(perm_air['0'])
    rel_perm_air_x   =  ((rel_perm_air_x-rel_perm_air_x[0])/(K1-rel_perm_air_x[0]))
    rel_perm_air_y    =  np.asarray(perm_air['1'])
    rel_perm_air_y   =  ((rel_perm_air_y-rel_perm_air_y[0])/(K2-rel_perm_air_y[0]))
    rel_perm_air_z    =  np.asarray(perm_air['2'])
    rel_perm_air_z   =  ((rel_perm_air_z-rel_perm_air_z[0])/(K3-rel_perm_air_z[0]))
    rel_perm_water_x  =  np.asarray(perm_water['0'])
    rel_perm_water_x =  ((rel_perm_water_x-rel_perm_water_x[-1])/(rel_perm_water_x[0]-rel_perm_water_x[-1]))
    rel_perm_water_y  =  np.asarray(perm_water['1'])
    rel_perm_water_y =  ((rel_perm_water_y-rel_perm_water_y[-1])/(rel_perm_water_y[0]-rel_perm_water_y[-1]))
    rel_perm_water_z  =  np.asarray(perm_water['2'])
    rel_perm_water_z =  ((rel_perm_water_z-rel_perm_water_z[-1])/(rel_perm_water_z[0]-rel_perm_water_z[-1]))
    

    Krdata = {}
    Krdata['Swx'] = 1-satx
    Krdata['Swy'] = 1-saty
    Krdata['Swz'] = 1-satz
    Krdata['Krnwx'] = rel_perm_air_x
    Krdata['Krwx'] = rel_perm_water_x
    Krdata['Krnwy'] = rel_perm_air_y
    Krdata['Krwy'] = rel_perm_water_y
    Krdata['Krnwz'] = rel_perm_air_z
    Krdata['Krwz'] = rel_perm_water_z
    
    Krdata['Swx'] = [float(i) for i in Krdata['Swx']]
    Krdata['Swy'] = [float(i) for i in Krdata['Swy']]
    Krdata['Swz'] = [float(i) for i in Krdata['Swz']]
    Krdata['Krwx'] = [float(i) for i in Krdata['Krwx']]
    Krdata['Krnwx'] = [float(i) for i in Krdata['Krnwx']]
    Krdata['Krwy'] = [float(i) for i in Krdata['Krwy']]
    Krdata['Krnwy'] = [float(i) for i in Krdata['Krnwy']]
    Krdata['Krwz'] = [float(i) for i in Krdata['Krwz']]
    Krdata['Krnwz'] = [float(i) for i in Krdata['Krnwz']]
    
df_krdata = pd.DataFrame(Krdata)
df_pcdata = pd.DataFrame(PC_data)
df_pore_radius = pd.DataFrame(highResNetwork['pore.diameter'], columns=['Pore Diameter'])
df_throat_radius = pd.DataFrame(highResNetwork['throat.diameter'], columns=['Throat Diameter'])
df_throat_coordination = pd.DataFrame(coordinationNum, columns=['Coordination Number'])
df_throat_length = pd.DataFrame(highResNetwork['throat.total_length'], columns=['Throat Length'])

with pd.ExcelWriter(ml_output_path + 'output.xlsx', engine='xlsxwriter') as writer:
    df_krdata.to_excel(writer, sheet_name='Krdata', index=False)
    df_pcdata.to_excel(writer, sheet_name='PC_data', index=False)
    df_pore_radius.to_excel(writer, sheet_name='Pore Diameter', index=False)
    df_throat_radius.to_excel(writer, sheet_name='Throat Diameter', index=False)
    df_throat_coordination.to_excel(writer, sheet_name='Coordination Number', index=False)
    df_throat_length.to_excel(writer, sheet_name='Throat Length', index=False)
    
print("Data has been exported to 'output.xlsx'")