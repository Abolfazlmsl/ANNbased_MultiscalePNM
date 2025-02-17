import openpnm as op
import numpy as np
import scipy.spatial as sptl
from solver import computeDt
import openpnm.models.physics as mods

def calculate_series_conductance(conductance_list):
    """
    Calculate the equivalent series conductance for a list of conductances
    """
    if not conductance_list:  
        return 0
    total_resistance = sum(1.0 / conductance for conductance in conductance_list)
    if total_resistance == 0:  
        return float('inf')  
    return 1.0 / total_resistance

def calculate_total_conductance(array_of_lists):
    """
    Calculate the total conductance for an array of lists of series conductances
    """
    total_conductance = 0
    for conductance_list in array_of_lists:
        series_conductance = calculate_series_conductance(conductance_list)
        total_conductance += series_conductance
    return total_conductance

def adjust_current_values(array):
    throats = {}
    
    for idx, lst in enumerate(array):
        for i in range(len(lst) - 1):
            throat = (lst[i], lst[i + 1])
            if throat in throats:
                throats[throat].append((idx, i))
            else:
                throats[throat] = [(idx, i)]
                
    return throats
    
def connect(mainNetwork, networks, neighbors, connectedRegions,
            minLength, ann_model, maxThroatLength):
    minThroats = 0
    macThroats = 0
    diamsThroatsMacro = []
    lengthsThroatsMacro = []
    upscaledPoresDiam = []
    upscaledPoresCoord = []
    upscaledThroatsDism = []
    upscaledThroatsCond = []
    totalNetPores = len(mainNetwork.Ps)

    mainPores = 0
    mergeThroatsMacroMain = []
    for idx, network in enumerate(networks):
        mergeThroatsMacro = []
        mergeThroatsMacroCluster = [] 
        count = 0

        if network is False:
            continue            
            
        mainPores += len(network['pore.all'])    
        op.topotools.find_surface_pores(network)
        surfacePores = np.where(network['pore.surface'])
        poreDistances = sptl.distance_matrix(network['pore.coords'][surfacePores], mainNetwork['pore.coords'])
        poreDistances = np.triu(poreDistances, 1)
        MacroNeighbors = np.where(poreDistances <  maxThroatLength/2)
        for itr, idx in enumerate(MacroNeighbors[0]):
            MacroNeighbors[0][itr] = surfacePores[0][idx]
            
        uniqueMacroNeighbors = np.unique(MacroNeighbors[1])
        
        for ii in range(len(MacroNeighbors[0])):
            pore1coord = network['pore.coords'][MacroNeighbors[0][ii]]
            pore2coord = mainNetwork['pore.coords'][MacroNeighbors[1][ii]]
            pore1diam = network['pore.equivalent_diameter'][MacroNeighbors[0][ii]]
            pore2diam = mainNetwork['pore.equivalent_diameter'][MacroNeighbors[1][ii]]
            
            new = [pore1coord[0], pore1coord[1], pore1coord[2],
                   pore2coord[0], pore2coord[1], pore2coord[2],
                   pore1diam, pore2diam]
            
            prediction = ann_model.predict(new)[0]  
            
            connection_result = prediction[0] 
            predicted_diameter = prediction[1]   
            predicted_length = prediction[2] 
            
            if connection_result:               
                firstPoreIndex = int(count + 1 + totalNetPores)
                secondPoreIndex = int(MacroNeighbors[1][ii])
                          
                firstPoreIndexCluster = int(MacroNeighbors[0][ii])
                previousMacroPoresNum = np.where(uniqueMacroNeighbors == MacroNeighbors[1][ii])[0]
                secondPoreIndexCluster = int(previousMacroPoresNum + 1 + len(network['pore.all']))
                
                diamsThroatsMacro.append(predicted_diameter)
                lengthsThroatsMacro.append(predicted_length)
                mergeThroatsMacroMain.append([firstPoreIndex, secondPoreIndex]) 
                mergeThroatsMacro.append([firstPoreIndex, secondPoreIndex]) 
                mergeThroatsMacroCluster.append([firstPoreIndexCluster, secondPoreIndexCluster]) 

                print("Add a micro-macro throat")
                macThroats += 1
        
        if len(mergeThroatsMacro) == 0:
            networks[idx] = False
            continue
        
        
        network['pore.diameter'] = network['pore.equivalent_diameter']
        network['throat.diameter'] = network['throat.equivalent_diameter']
        phase = op.phases.Air(network=network)
        phase.add_model(propname='throat.hydraulic_size_factors',
                        model=op.models.geometry.hydraulic_size_factors.spheres_and_cylinders)
        phase.add_model(propname='throat.hydraulic_conductance',
                        model=op.models.physics.hydraulic_conductance.hagen_poiseuille)
        
        
        pore_conductance = np.zeros(network.Np)
        for pore in network.Ps:
            throats = network.find_neighbor_throats(pores=pore)
            pore_conductance[pore] = np.sum(phase['throat.hydraulic_conductance'][throats])

        p_best = np.argmax(pore_conductance)
        p_best_diam = network['pore.diameter'][p_best]
        p_best_coord = network['pore.coords'][p_best]
        upscaledPoresDiam.append(p_best_diam)
        upscaledPoresCoord.append(p_best_coord)
        
        mergeThroatsMacro = np.array(mergeThroatsMacro)
        mergeThroatsMacroCluster = np.array(mergeThroatsMacroCluster)

        sortarray, mainIndices = np.unique(mergeThroatsMacro[:,1], return_index=True)
        new_coords = mainNetwork['pore.coords'][mergeThroatsMacro[:,1][np.sort(mainIndices)]]
        new_pores_diam = mainNetwork['pore.equivalent_diameter'][mergeThroatsMacro[:,1][np.sort(mainIndices)]]
        
        #I think network should be mainNetwork
        op.topotools.extend(network=mainNetwork, coords=new_coords)
        op.topotools.extend(network=mainNetwork, conns=mergeThroatsMacroCluster, labels="new_macro")
        
        newthroatsMacro = np.where(network['throat.new_macro'])[0]
        network['pore.equivalent_diameter'][newthroatsMacro] = new_pores_diam
        network['throat.equivalent_diameter'][newthroatsMacro] = diamsThroatsMacro
        network['throat.total_length'][newthroatsMacro] = lengthsThroatsMacro
        network['pore.diameter'] = network['pore.equivalent_diameter']
        network['throat.diameter'] = network['throat.equivalent_diameter']
        network['throat.length'] = network['throat.total_length']

        phase = op.phase.Air(network=network)
        phase.add_model(propname='throat.hydraulic_size_factors',
                              model=op.models.geometry.hydraulic_size_factors.spheres_and_cylinders)
        phase.add_model(propname='throat.hydraulic_conductance',
                              model=op.models.physics.hydraulic_conductance.hagen_poiseuille)
        
        sortarrayCluster, mainIndicesCluster = np.unique(mergeThroatsMacroCluster[:,1], return_index=True)
        for p_macro in mergeThroatsMacroCluster[:,1][np.sort(mainIndicesCluster)]:           
            coordination = op._skgraph.queries.find_coordination(network, p_macro)
            gt_all = []
            dt_all = []
            path_all = []
            pore_all = []
            for c_idx in coordination:
                short_path = op.topotools.find_path(network, [p_macro, p_best]) 
                path_all.append(short_path['edge_paths']) 
                pore_all.append(short_path['node_paths']) 
                gt_current = network['throat.hydraulic_conductance'][short_path['edge_paths']]
                gt_all.append(gt_current)
                dt_current = network['throat.equivalent_diameter'][short_path['edge_paths']]
                dt_all.append(dt_current)
                op.topotools.trim(network, short_path['edge_paths'][0])
            
            #Find the common throats 
            adjusted_throats = adjust_current_values(pore_all)
            
            for throat, occurrences in adjusted_throats.items():
                if len(occurrences) > 1:
                    for lst_idx, pos in occurrences:
                        gt_all[lst_idx][pos] = gt_all[lst_idx][pos] / len(occurrences)

            dt_initial = np.mean(dt_all)
            gt_equivalent = calculate_total_conductance(gt_all)
            l_ctc = np.linalg.norm(network['pore.coords'][p_macro] - network['pore.coords'][p_best])
            dt_equivalent = computeDt(network['pore.equivalent_diameter'][p_best], network['pore.equivalent_diameter'][p_macro],
                                      phase['pore.viscosity'][0], phase['pore.viscosity'][0], phase['pore.viscosity'][0],
                                      l_ctc, gt_equivalent, dt_initial)  
            
            upscaledThroatsDism.append(dt_equivalent)
            upscaledThroatsCond.append(gt_equivalent)
            
        count += 1

    mergeThroatsMacroMain = np.array(mergeThroatsMacroMain)

    op.topotools.extend(network=mainNetwork, coords=upscaledPoresCoord, labels="upscaled_pores")
    op.topotools.extend(network=mainNetwork, conns=mergeThroatsMacroMain, labels="upscaled_throats")

    newthroatsupscaled = mainNetwork['throat.upscaled_pores']    
    newthroatsMacro = mainNetwork['throat.upscaled_throats']
    
    
    mainNetwork['pore.equivalent_diameter'][newthroatsupscaled] = upscaledPoresDiam
    mainNetwork['throat.equivalent_diameter'][newthroatsMacro] = upscaledThroatsDism

    mainNetwork['pore.diameter'] = mainNetwork['pore.equivalent_diameter']
    mainNetwork['throat.diameter'] = mainNetwork['throat.equivalent_diameter']
    mainNetwork['throat.length'] = mainNetwork['throat.total_length']

    mainNetwork.add_model(propname='pore.area',
                      model = op.models.geometry.pore_cross_sectional_area.sphere)
    mainNetwork.add_model(propname='throat.area',
                      model = op.models.geometry.throat_cross_sectional_area.cylinder)

    mainNetwork.add_model(propname='throat.endpoints',
                 model=op.models.geometry.throat_endpoints.spherical_pores,
                 pore_diameter='pore.diameter',
                 throat_diameter='throat.diameter')


    mainNetwork.add_model(propname='throat.conduit_lengths',
                      model = op.models.geometry.throat_length.conduit_lengths)

    mainNetwork.add_model(propname='throat.volume',
                      model = op.models.geometry.throat_volume.cylinder,
                      throat_diameter='throat.diameter',
                      throat_length='throat.length')

    geometry = op.geometry.GenericGeometry(network=mainNetwork,pores=mainNetwork.Ps,throats=mainNetwork.Ts)

    for i in mainNetwork.props():
        if i not in ['throat.conns','pore.coords']:
            geometry.update({i:mainNetwork.pop(i)})

    phase_inv = op.phases.Air(network = mainNetwork)
    phase_def = op.phases.Water(network = mainNetwork)

    phys_inv = op.physics.GenericPhysics(network=mainNetwork, phase=phase_inv, geometry=geometry)
    phys_def = op.physics.GenericPhysics(network=mainNetwork, phase=phase_def, geometry=geometry)

    phys_inv.add_model(propname='throat.entry_pressure',
                       model = mods.capillary_pressure.washburn)
    phys_def.add_model(propname='throat.entry_pressure',
                       model = mods.capillary_pressure.washburn)

    phys_inv.add_model(propname='throat.hydraulic_conductance',
                       model = mods.hydraulic_conductance.hagen_poiseuille)
    phys_def.add_model(propname='throat.hydraulic_conductance',
                       model = mods.hydraulic_conductance.hagen_poiseuille)
    
    phys_inv['throat.hydraulic_conductance'][newthroatsMacro] = upscaledThroatsCond
    phys_def['throat.hydraulic_conductance'][newthroatsMacro] = upscaledThroatsCond
    
    return mainNetwork, geometry, phase_inv, phase_def, phys_inv, phys_def,  minThroats, macThroats
