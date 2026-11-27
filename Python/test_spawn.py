import traceback
from ocean.io.track_loader import load_network
from ocean.transit.transit_agent import TransitAgent
import math

try:
    net, meta = load_network(r'c:\Ocean Library\Matlab code practice\Experiments\Scripts\Simulation\final_20260404_fixed.json')
    
    agent = TransitAgent(net, 'EWL', start_segment=0)
    print('Initial segment index:', agent.current_seg_idx)
    pos = list(agent.get_position())
    
    names = net.get_station_names()
    nodes = sorted(net.get_graph().nodes())
    pasir_ris = None
    tampines = None
    for n in nodes:
        node_name = net.get_graph().nodes[n].get('name')
        if 'Pasir Ris' in node_name:
            pasir_ris = list(net.get_graph().nodes[n].get('pos'))
        if node_name == 'Tampines':
            tampines = list(net.get_graph().nodes[n].get('pos'))

    print(f'Agent position: {pos}')
    print(f'Pasir Ris pos : {pasir_ris}')
    print(f'Tampines pos  : {tampines}')
    
    def dist(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        
    print(f'\nDist to Pasir Ris: {dist(pos, pasir_ris):.2f}')
    print(f'Dist to Tampines : {dist(pos, tampines):.2f}')

except Exception as e:
    traceback.print_exc()
