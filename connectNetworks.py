import openpnm as op
import numpy as np
import scipy.spatial as sptl

def connect(mainNetwork, networks, neighbors, connectedRegions,
            minLength, ann_model, maxThroatLength):
    minThroats = 0
    macThroats = 0
    mergeThroatsMacro = []
    diamsThroatsMacro = []
    lengthsThroatsMacro = []
    randomState = []
    totalNetPores = len(mainNetwork.Ps)

    mergeThroatsMacro = []
    mainPores = 0
    for idx, network in enumerate(networks):
        randomState.append([])
        if network is False:
            continue            
        
        mainPores += len(network['pore.all'])    
        op.topotools.find_surface_pores(network)
        surfacePores = np.where(network['pore.surface'])
        poreDistances = sptl.distance_matrix(network['pore.coords'][surfacePores], mainNetwork['pore.coords'])
        poreDistances = np.triu(poreDistances, 1)
        MacroNeighbors = np.where((poreDistances < maxThroatLength/2) & (poreDistances > 0))
        for itr, idx in enumerate(MacroNeighbors[0]):
            MacroNeighbors[0][itr] = surfacePores[0][idx]
        
        for ii in range(len(MacroNeighbors[0])):
            pore1coord = network['pore.coords'][MacroNeighbors[0][ii]]
            pore2coord = mainNetwork['pore.coords'][MacroNeighbors[1][ii]]
            pore1diam = network['pore.equivalent_diameter'][MacroNeighbors[0][ii]]
            pore2diam = mainNetwork['pore.equivalent_diameter'][MacroNeighbors[1][ii]]
            
            new = np.array([[pore1coord[0], pore1coord[1], pore1coord[2],
                             pore2coord[0], pore2coord[1], pore2coord[2],
                             pore1diam, pore2diam]])
            
      
            prediction = ann_model.predict(new)[0]  
            
            connection_result = prediction[0] 
            predicted_diameter = prediction[1]   
            predicted_length = prediction[2]   
            
            if connection_result:  
                firstPoreIndex = int(MacroNeighbors[0][ii] + mainPores - len(network['pore.all']) + totalNetPores)
                secondPoreIndex = int(MacroNeighbors[1][ii])
                          
                diamsThroatsMacro.append(predicted_diameter)
                lengthsThroatsMacro.append(predicted_length)
                mergeThroatsMacro.append([firstPoreIndex, secondPoreIndex]) 
                
                print("Add a micro-macro throat")
                macThroats += 1

    mergeThroatsMacro = np.array(mergeThroatsMacro)
    print("---------------------------------------------------------------")
    print("mergeThroatsMacro: ", len(mergeThroatsMacro))
    
    existedNetworks = [net for net in networks if net is not False]
    op.topotools.merge_networks(mainNetwork, donor=existedNetworks)
    op.topotools.extend(network=mainNetwork, conns=mergeThroatsMacro, labels="new_throat_macro")
    
    newthroatsMacro = np.where(mainNetwork['throat.new_throat_macro'])[0]
    mainNetwork['throat.equivalent_diameter'][newthroatsMacro] = diamsThroatsMacro
    mainNetwork['throat.total_length'][newthroatsMacro] = lengthsThroatsMacro
        
    return mainNetwork, minThroats, macThroats, randomState
