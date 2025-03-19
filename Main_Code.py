import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import random
import time
import math
import queue
import sys
import copy
import numpy



class Node():
    def __init__(self,i=None):
        # Can't include node objects in here as it would cause parallel recursion issues (recursive pickling). Thus, use id's to reference nodes
        self.connections = []
        self.label = None
        # Give each node an attribute of their index in self.nodes (needed for identification across processes)
        self.id = i

class G():
    def __init__(self,n,p):
        # Instantiating nodes in list comprehension is faster than parallel
        self.nodes = [Node(i) for i in range(n)]
        if p != 0:
            generate_edges(self.nodes,p)
        # If p = 0 then we just create nodes and no edges, so class can still be used by k_nearest_neighbour etc.


def save_results(results,file_name='previous_test_data'):
    f = open(file_name+".txt","w")
    f.write(str(results))
    f.close()


def perform_relay_allcast(n,p,k,l,gtype="erdos",include_extra_node=False):
    # Perform a single instance of relay allcast for given parameters
    # Used for testing purposes
    manager = mp.Manager()
    results = dict()
    results["n"] = n
    # Create G()
    graph = None
    print("Creating graph")
    if gtype == "erdos":
        results["p"] = p
        graph = G(n,p)
    elif gtype == "k_neighbour":
        results["k"] = k
        if include_extra_node == True:
            results["i"] = l
        graph = k_nearest_neighbour_cycle_graph(n,k,l,include_extra_node)
    elif gtype == "world":
        results["p"] = p
        results["k"] = k
        if include_extra_node == True:
            results["i"] = l
        graph = world_graph(n,p,k,l,include_extra_node)
    elif gtype == "line":
        graph = line_graph(n)
        # Store type for line and star to distinguish results as only 2 graphs to store just n
        results["type"] = "line"
    elif gtype == "star":
        graph = star_graph(n)
        results["type"] = "star"
    else:
        return results
    # Do flood fill (and then filter out desired nodes)
    print("Performing flood fill")
    most_occuring_label_and_size, results["Number isolated nodes removed"], results["Number of disconnected graphs removed"], results["Largest size of disconnected graph removed"], results["Average size of disconnected graph removed"] = flood_fill(graph.nodes)
    results["Size of target graph"] = most_occuring_label_and_size[1]
    target_label = most_occuring_label_and_size[0]
    target_graph = []
    print("Assembling target graph")
    for node in graph.nodes:
        if node.label == target_label:
            target_graph.append(node)
    # Call degree_data_collection() to collect data on filtered graph
    print("Computing node degrees")
    results["Max degree"], results["Min degree"], results["Average degree"] = degree_data_collection(target_graph)
    print("Min degree:", results["Min degree"])
    # Perform allcast algorithm requested
    print("Performing allcast")
    results["Time taken"] = random_relay(target_graph)
    # Return results
    print("Complete")
    return results
    

def generate_edges(node_list,p):
    # Use pzer and edge_to_node_pairing here
    G = pzer(node_list,p)
    edge_to_node_pairing(node_list,G)
    # No return needed here


def pzer(node_list,p):
    # COMPLETE! Performance scales with np and n
    # Input list of nodes and p to output list of edge numbers
    # Set initial variables
    n = len(node_list)
    E = (n*(n-1))/2
    G = []
    L = 0
    # Value of B needs to be calculated
    B = approximate_B(n,p,E)
    while L < E:
        # Generate B random skip values in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            s = pool.map(generate_skip,[p for i in range(B)])
            # Add last value of G to first value of s and prefix sum
            if G != []:
                s[0] += G[-1]
            s = prefix_sum(s)
            # Add results to end of G and set L to new value
            G = G + s
            L = G[-1]
    return G

def generate_skip(p):
    # Used by parallel processes to generate and return skip value to s
    t = random.uniform(0, 1)
    k = max(0,math.ceil(math.log(t,1-p))-1)
    # Return k+1
    return k+1

def approximate_B(n,p,E):
    # Optimal solution is a power of 2 equal to a multiple of the number of cores and approximately Ep+lambda*sigma from original source
    # Below calc finds largest power of 2 less than E*p+sigma (we fix lambda = 1)
    # We assume our number of cores is appropriate and we manually assign a power of 2 no.cores on supercomputer
    sigma = math.sqrt(p*(1-p)*E)
    B = 2**(math.floor(math.log2((E*p)+sigma)))
    return B



def prefix_sum(list_values):
    # Need to do in parallel while keeping the order of indexes correct
    # Perform inclusive prefix sum on given list using parallel processing
    # The length of the list passed through must be a power of 2!!
    n = len(list_values)
    # Return if at singularity
    if n == 1:
        return list_values
    y = []
    for k in range(1,n,2):
        y.append(list_values[k]+list_values[k-1])
    y = prefix_sum(y)
    # Recursively called until singularity is returned
    # Now reverse process
    for j in range(len(y)):
        k = (j*2)+1
        list_values[k-1] = y[j]-list_values[k]
        list_values[k] = y[j]
    return list_values  

    

def edge_to_node_pairing(node_list,edge_list):
    # Convert list of [1,26,89,...] edges to actual connections in node attributes
    # Edge list comes ordered already from pzer
    n = len(node_list)
    c, i, g = 0, 0, 0
    while (g < len(edge_list)) and (edge_list[g]<=(n*(n-1))/2):
        # Each node can form n-i-1 edges as we iterate through
        if edge_list[g] > c+n-i-1:
            # If current value of edge_list is larger than cumulative value, increment node
            c += n-i-1
            i += 1
        else:
            # Otherwise add connection between nodes by id
            node_list[i].connections.append(node_list[i+edge_list[g]-c].id)
            node_list[i+edge_list[g]-c].connections.append(node_list[i].id)
            g += 1
    # No values need returning here


############ Original parralelised code which I can archive in favour of the batch and serial approach #############

def degree_data_collection_parallel(node_list):
    # COMPLETE! Operates in parallel. Can do n=1,000,000 in 1m30s
    # Return desired node degree data from given list of nodes
    manager = mp.Manager()
    min_degree = manager.Value('i',len(node_list))
    max_degree = manager.Value('i',0)
    average_degree = manager.Value('d',0.0)
    cores = mp.cpu_count()
    with mp.Pool(processes=cores) as pool:
        # Pass through each node to be processed in parallel. No results need collecting as operating on shared variables
        # Can pass nodes through directly as they don't contain any recursive data that would cause pickling issues
        pool.starmap(check_degree,[(n,min_degree,max_degree,average_degree) for n in node_list])
    # Divide average_degree once done
    average_degree.value = average_degree.value/len(node_list)
    return max_degree.value, min_degree.value, average_degree.value

def check_degree(node,min_degree,max_degree,average_degree):
    # Called by parallel processes to include the degree of given node in results
    # Average degree is done as a sum and divided outside of this function
    degree = len(node.connections)
    if degree > max_degree.value:
        max_degree.value = degree
    if degree < min_degree.value:
        min_degree.value = degree
    average_degree.value += degree



def flood_fill_parallel_method(node_list,n,p):
    # COMPLETE! Can do n=1,000,000 with p=0.00001 in 12mins. Some room for optimisation with processes called or L if needed
    # Can paralellise parts where same label i is being used (i.e. within the loop)
    # Passed n and p through as can implement lemma 3.4 to ensure I can paralellise giant component only to optimise resources
    manager = mp.Manager()
    # Create a shared dictionary for labels so processes can communicate with one another
    label_dict = manager.dict()
    # Create a shared dictionary of connections for each node, that way we don't need to pickle node objects recursively by using node_list as a parameter
    # Connection_dict made in parallel?
    connection_dict = manager.dict()
    for i in range(0,len(node_list)):
        connection_dict[i] = node_list[i].connections
    L = [i for i in range(0,len(node_list))]
    i = 1
    # Can create queue outside of loop as will be emptied each recursion
    q = queue.Queue()
    # Create list for label sizes (index of size corresponds to i-1)
    label_sizes = []
    while len(L)>0:
        current_size = 0
        # Select random l and put in queue
        l = random.choice(L)
        q.put(l)
        # Loop over connections of node with id l until queue empty
        while q.empty()==False:
            # Conditions of lemma 3.4 to confirm we are currently on giant component
            if (n*p > 1) and (n*p < math.log(n)) and (current_size > (math.log(n))/((n*p)-1-math.log(n*p))):
                # We now know we are processing the giant component so can process in parallel
                # Each process will be given a node to work on from the queue
                cores = min(mp.cpu_count(),q.qsize())
                parallel_jobs = []
                for j in range(cores):
                    job = q.get()
                    parallel_jobs.append(job)
                with mp.Pool(processes=cores) as pool:
                    # Start processing in parallel
                    r = pool.starmap(flood_fill_parallel,[(j,i,label_dict,connection_dict) for j in parallel_jobs])
                    # Our results r are a 2D array so we need to work out the union and remove results from L
                    if r != []:
                        remove_L = list(set.union(*map(set,r)))
                        for labelled in remove_L:
                            L.remove(labelled)
                            current_size += 1
            else:
                # Serial processing for smaller components not satisfying lemma 3.4
                node = q.get()
                # If already labelled, pass
                if label_dict.get(node) is not None:
                    pass
                else:
                    # Apply label and remove index from L
                    label_dict[node] = i
                    current_size += 1
                    L.remove(node)
                    # Append connections to queue
                    for next_node in connection_dict[node]:
                        q.put(next_node)
        # End loop
        label_sizes.append(current_size)
        # Increment label
        i+=1

    # As labels were stored in dict so they could be processed in parallel, we now attach labels to node objects
    for j in range(0,len(node_list)):
        node_list[j].label = label_dict[node_list[j].id]
        
    # Harvest and return desired data. Labels applied to objects automatically
    largest_component = max(label_sizes)
    most_occuring_label_and_size = [label_sizes.index(largest_component)+1,largest_component]
    no_disconnected_graphs_removed = 0
    no_isolated_nodes_removed = 0
    largest_size_of_disconnected_graph_removed = 0
    average_size_of_disconnected_graph_removed = 0
    for size in label_sizes:
        if size == 1:
            no_isolated_nodes_removed += 1
        if (size > 1):
            no_disconnected_graphs_removed += 1
            if size < largest_component:
                average_size_of_disconnected_graph_removed += size
                if size > largest_size_of_disconnected_graph_removed:
                    largest_size_of_disconnected_graph_removed = size
    # We need to -1 as largest component was included
    no_disconnected_graphs_removed -= 1
    if no_disconnected_graphs_removed > 0:
        # Check >0 as it avoids zerodivision error
        average_size_of_disconnected_graph_removed = average_size_of_disconnected_graph_removed/(no_disconnected_graphs_removed)
    return most_occuring_label_and_size, no_isolated_nodes_removed, no_disconnected_graphs_removed, largest_size_of_disconnected_graph_removed, average_size_of_disconnected_graph_removed


def flood_fill_parallel(given_node,i,label_dict,connection_dict):
    # Each process is assigned a node object to perform flood fill on
    # This function adds labels assigned to a list A which is returned so it can be combined with other processes to calculate L accurately
    # First we establish a queue for the process to work off of
    A = []
    q = queue.Queue()
    q.put(given_node)
    while q.empty()==False:
        node = q.get()
        # If already labelled in shared memory, pass
        if label_dict.get(node) is not None:
            pass
        else:
            # Apply label to shared dict and add index to A
            label_dict[node] = i
            A.append(node)
            # Append connections to queue
            for next_node in connection_dict[node]:
                q.put(next_node)
    return A


def degree_data_collection(node_list):
    # Serial degree checker
    # Return desired node degree data from given list of nodes
    min_degree = len(node_list)
    max_degree = 0
    average_degree = 0.0
    # Go through each node to be processed sequentially
    for node in node_list:
        degree = len(node.connections)
        if degree > max_degree:
            max_degree = degree
        if degree < min_degree:
            min_degree = degree
        average_degree += degree
    # Divide average_degree once done
    if len(node_list)!=0:
        average_degree = average_degree/len(node_list)
    #print("Min degree:", min_degree)
    return max_degree, min_degree, average_degree



def flood_fill(node_list):
    # Serial flood fill for minimal overhead
    # Should be called in parallel with different graph inputs
    L = [i for i in range(0,len(node_list))]
    i = 1
    # Can create queue outside of loop as will be emptied each recursion
    q = queue.Queue()
    # Create list for label sizes (index of size corresponds to i-1)
    label_sizes = []
    while len(L)>0:
        current_size = 0
        # Select random l and put in queue
        l = random.choice(L)
        q.put(l)
        # Loop over connections of node with id l until queue empty
        while q.empty()==False:   
            # Serial processing
            node_id = q.get()
            # If already labelled, pass
            if node_list[node_id].label != None:
                pass
            else:
                # Apply label and remove index from L
                node_list[node_id].label = i
                current_size += 1
                L.remove(node_id)
                # Append connections to queue
                for next_node in node_list[node_id].connections:
                    q.put(next_node)
        # End loop
        label_sizes.append(current_size)
        # Increment label
        i+=1
        
    # Harvest and return desired data. Labels applied to objects automatically
    largest_component = max(label_sizes)
    most_occuring_label_and_size = [label_sizes.index(largest_component)+1,largest_component]
    no_disconnected_graphs_removed = 0
    no_isolated_nodes_removed = 0
    largest_size_of_disconnected_graph_removed = 0
    average_size_of_disconnected_graph_removed = 0
    for size in label_sizes:
        if size == 1:
            no_isolated_nodes_removed += 1
        if (size > 1):
            no_disconnected_graphs_removed += 1
            if size < largest_component:
                average_size_of_disconnected_graph_removed += size
                if size > largest_size_of_disconnected_graph_removed:
                    largest_size_of_disconnected_graph_removed = size
    # We need to -1 as largest component was included
    no_disconnected_graphs_removed -= 1
    if no_disconnected_graphs_removed > 0:
        # Check >0 as it avoids zerodivision error
        average_size_of_disconnected_graph_removed = average_size_of_disconnected_graph_removed/(no_disconnected_graphs_removed)
    return most_occuring_label_and_size, no_isolated_nodes_removed, no_disconnected_graphs_removed, largest_size_of_disconnected_graph_removed, average_size_of_disconnected_graph_removed



def random_relay(graph_nodes):
    # Serial function to perform allcast of random relay
    # Recommended to run in parallel with different graph parameters
    # That way we can batch process graphs without incurring much overhead
    n = len(graph_nodes)
    # List below is used to signify what node ids have received all packets and thus can have connections TO them removed (not from)
    removed_ids = []
    # And list here is used to signify the ones that are still waiting for all packets
    ids_processing = [node.id for node in graph_nodes]
    # As graph is filtered, id's no longer correspond to index so use hashmap with id:packets
    packet_dict = {node.id:[copy.deepcopy(node.id)] for node in graph_nodes}
    # Need relay_dict to keep track of packets we can select from
    relay_dict = {node.id:1 for node in graph_nodes}
    # We also need a node_id:node dictionary for 1-to-1 lookup when removing edges
    node_dict = {node.id:node for node in graph_nodes}
    # Time results are collected as time step when 1%,2%,3%,...,100% of nodes have all packets
    time_results = []
    # k is percentage of nodes we next record value of (increments every 1%)
    k = 1
    time_taken = 0
    allcast_complete = False
    while allcast_complete == False:
        # Select random packet from node and broadcast to all neighbours
        for node in graph_nodes:
            packet = random.choice(list(packet_dict[node.id][:relay_dict[node.id]]))
            for connection in node.connections:
                # Add packet to dictionary for each connecting node (removing duplicates)
                if packet not in packet_dict[connection]:
                    packet_dict[connection].append(packet)
        # Now update relay ids for nodes still collecting packets and check for edges or nodes we can remove
        to_remove = []
        for ids in ids_processing:
            node = node_dict[ids]
            relay_dict[ids] = len(packet_dict[ids])
            if relay_dict[ids] == n:
                # Now, if an object returns n, it has received all nodes and no longer has to receive packets
                # Thus we eliminate all connections to that node (not connections from as it can still transmit packets)
                # This improves efficiency over time. As we aren't reusing the graph after this, it works
                # We use node_dict here so we can lookup the exact edges we need to remove rather than iterate through everything
                connected_nodes_id = node.connections
                for connected_node in connected_nodes_id:
                    node_dict[connected_node].connections.remove(ids)
                    if (node_dict[connected_node].connections == []) and (connected_node in removed_ids):
                        # If a node has run out of connections and does not require packets, it can be removed from the remainder of the problem as it adds no more value
                        graph_nodes.remove(node_dict[connected_node])
                to_remove.append(ids)
                # Finally, if node that just obtained all packets has no connections itself, remove it
                if node.connections == []:
                    graph_nodes.remove(node)
        # Apply to remove list
        for ids in to_remove:
            # Move node from processing to removed list
            removed_ids.append(ids)
            ids_processing.remove(ids)
        # Increment time step and repeat if end condition is unsatisfied
        time_taken += 1
        if ids_processing == []:
            allcast_complete = True
        # Also check for new time results value
        while len(removed_ids) >= (k/100)*n:
            time_results.append(time_taken)
            k += 1
    return time_results


def mass_relay_allcast(n,p,k,l,iterations,gtype="erdos",include_extra_node=False,save_location='previous_test_data'):
    # Input n, p and number of iterations to perform allcast on those graphs
    # We generate the graphs sequentially as they have embedded paralellism
    # Flood fill, degree checker and random relay are called in parallel for each graph as they are sequential functions
    # In this sense we batch process allcast problems
    # K neighbour graphs are also used here but store k instead of p
    graph_list = []
    for i in range(iterations):
        # Append given type of graph an iterations number of times
        if gtype == "erdos":
            g = G(n,p)
            graph_list.append(g.nodes)
        elif gtype == "k_neighbour":
            g = k_nearest_neighbour_cycle_graph(n,k,l,include_extra_node)
            graph_list.append(g.nodes)
        elif gtype == "world":
            g = world_graph(n,p,k,l,include_extra_node)
            graph_list.append(g.nodes)
        elif gtype == "line":
            g = line_graph(n)
            graph_list.append(g.nodes)
        elif gtype == "star":
            g = star_graph(n)
            graph_list.append(g.nodes)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(perform_relay,[(graph,n,p,k,l,include_extra_node,gtype) for graph in graph_list])
        # Save and return results as list of dictionaries
        save_results(results,save_location)
        print("Complete!")
        print("Results saved to "+save_location+".txt")
        return results

def perform_relay(graph,n,p,k,l,include_extra_node,gtype):
    # Perform relay allcast algorithm on given graph
    # Need to perform flood fill, assemble graph, check degrees and random relay in that order
    results = dict()
    results["n"] = n
    if gtype == "erdos":
        results["p"] = p
    elif gtype == "k_neighbour":
        results["k"] = k
        if include_extra_node == True:
            results["i"] = l
    elif gtype == "world":
        results["p"] = p
        results["k"] = k
        if include_extra_node == True:
            results["i"] = l
    elif gtype == "line":
        results["type"] = "line"
    elif gtype == "star":
        results["type"] = "star"
    # Flood fill
    most_occuring_label_and_size, results["Number isolated nodes removed"], results["Number of disconnected graphs removed"], results["Largest size of disconnected graph removed"], results["Average size of disconnected graph removed"] = flood_fill(graph)
    results["Size of target graph"] = most_occuring_label_and_size[1]
    # Assemble graph
    target_label = most_occuring_label_and_size[0]
    target_graph = []
    for node in graph:
        if node.label == target_label:
            target_graph.append(node)
    # Call degree_data_collection() to collect data on filtered graph
    results["Max degree"], results["Min degree"], results["Average degree"] = degree_data_collection(target_graph)
    # Perform random relay allcast algorithm
    results["Time taken"] = random_relay(target_graph)
    # Return results
    return results


def k_nearest_neighbour_cycle_graph(n,k,l,include_extra_node=False):
    # Create a nearest neighbour graph to analyse diameter vs node degree affecting allcast
    # Use graph class with p=0 so we just get object with node list attribute
    graph = G(n,0)
    for i in range(0,len(graph.nodes)):
        for j in range(1,k+1):
            graph.nodes[i].connections.append(graph.nodes[(i+j)%len(graph.nodes)].id)
            graph.nodes[i].connections.append(graph.nodes[i-j].id)
    # Remove duplicate edges in connections (occurs when k>=n/2)
    for node in graph.nodes:
        node.connections = list(dict.fromkeys(node.connections))
    # Now check if we want l connections to an extra node
    if include_extra_node == True:
        extra_index = len(graph.nodes)
        graph.nodes.append(Node(extra_index))
        for j in range(0,l):
            graph.nodes[j].connections.append(extra_index)
            graph.nodes[extra_index].connections.append(j)
    # Return our nearest neighbour cycle graph
    return graph


def world_graph(n,p,k,l,include_extra_node=False):
    # Create a graph that is the union of an ER and k-neighbour graph
    # Start with ER graph
    graph1 = G(n,p)
    # And create a seperate k-neighbour graph
    graph2 = k_nearest_neighbour_cycle_graph(n,k,l,include_extra_node)
    # Now can simply iterate through each node object in graph2 (as it has same or more nodes due to extra node) and union connections with corresponding node in graph1
    # As graph nodes are initially ordered in order of id, we can just iterate through with ease
    for q in range(0,len(graph2.nodes)):
        if q < len(graph1.nodes):
            graph2.nodes[q].connections = list(set(graph2.nodes[q].connections).union(set(graph1.nodes[q].connections)))
        else:
            break
    # Return modified graph2 object
    return graph2


def line_graph(n):
    # Use graph class with p=0 so we just get object with node list attribute
    graph = G(n,0)
    for i in range(1,len(graph.nodes)):
        # Connect node i to node i-1
        graph.nodes[i].connections.append(graph.nodes[i-1].id)
        graph.nodes[i-1].connections.append(graph.nodes[i].id)
    # Return line graph
    return graph


def star_graph(n):
    # Use graph class with p=0 so we just get object with node list attribute
    graph = G(n,0)
    for i in range(1,len(graph.nodes)):
        # Connect node i to node 0
        graph.nodes[i].connections.append(graph.nodes[0].id)
        graph.nodes[0].connections.append(graph.nodes[i].id)
    # Return star graph
    return graph


def lower_bound_conjecture(n_lim,p_lim,iterations,do_filter=False):
    # Function generates colourmap of conjecture up to n_lim with 0<p<1 at p_step increments
    n_values = [i for i in range(n_lim[0],n_lim[1]+1,n_lim[2])]
    p_values=[]
    i=0
    while p_lim[0]+i*p_lim[2]<=p_lim[1]:
        p_values.append(p_lim[0]+i*p_lim[2])
        i+=1
    results=[]
    for j in range(0,len(n_values)):
        n=n_values[j]
        column=[]
        for p in p_values:
            c=n*p
            if c<=1:
                # 0.5 if conditions not satisfied or min degree is 0
                # Theorem 3.2 requires np>1
                column.append(0.5)
            elif (math.log(n)>c) and (do_filter==True):
                # Extra filter if wanted
                column.append(0.5)
            else:
                # Insert upper bound formula for diameter (c=np)
                diam=(math.log(n)/math.log(n*p))+2*((math.log(n)*((10*c)/((math.sqrt(c)-1)**2)+1))/((n*p)*(c-math.log(2*c))))+1
                min_degree=0
                # Compute maximum minimum degree across given number of iterations
                for i in range(1,iterations+1):
                    graph=G(n,p)
                    most_occuring_label_and_size,_,_,_,_=flood_fill(graph.nodes)
                    target_label = most_occuring_label_and_size[0]
                    target_graph = []
                    for node in graph.nodes:
                        if node.label == target_label:
                            target_graph.append(node)
                    _, mindeg, _ = degree_data_collection(target_graph)
                    if mindeg>min_degree:
                        min_degree=mindeg
                # If satisfies bound append true, else false
                if min_degree == 0:
                    column.append(0.5)
                elif math.ceil((n-1)/min_degree)>=diam:
                    column.append(1)
                else:
                    column.append(0)
        results.append(column)
    # Now can plot results
    plt.figure()
    plt.title("Colourmap of "+ r'$\left\lceil(n-1)/d^{in}_{min}\right\rceil\geq diam(G(n,p))$')
    #plt.pcolor(p_values,n_values,results,cmap="magma")
    binary_map = LinearSegmentedColormap.from_list("binary_cmap", ["black", "yellow"])
    norm = BoundaryNorm([0, 0.5, 1], binary_map.N)
    plt.pcolor(p_values,n_values,results,cmap=binary_map,norm=norm)
    plt.xlabel("p")
    plt.ylabel("n")
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["False", "True"])
    plt.show()



if __name__ == "__main__":
    t = time.time()
    n = 200
    a = 7/9
    p = n**(-a)
    iterations = 100
    k = 1
    i = 1
    print("Cores:",mp.cpu_count())

    # Mass allcast test
##    mass_relay_allcast(n,p,k,i,iterations,"erdos",False,'data_collection_91')

    # Single allcast test
##    print(perform_relay_allcast(n,p,k,i,"k_neighbour",False))

    # Colourmap for lower bound conjecture
    lower_bound_conjecture([90,110,1],[0.03,0.04,0.001],5,False)

    # Line graph test
##    lg = line_graph(n)
##    for node in lg.nodes:
##        print(node.id, node.connections)

    # Star graph test
##    sg = star_graph(n)
##    for node in sg.nodes:
##        print(node.id, node.connections)

    # K nearest neighbour graph test
##    kg = k_nearest_neighbour_cycle_graph(n,k,i,True)
##    for node in kg.nodes:
##        print(node.id, node.connections)

    # World graph test
##    wg = world_graph(n,p,k,i,True)
##    for node in wg.nodes:
##        print(node.id, node.connections)

    # PZER test
##    graph = [Node(i) for i in range(n)]
##    print(pzer(graph,p))

    # Prefix sum test
##    print(prefix_sum([3,2,1,5,7,4,34,6]))

    # Edge-to-node testing
##    edge_to_node_pairing(graph.nodes,[1,7,45,800,1000,1200,499500,1000000])
##    print(graph.nodes[1].connections)
    
    # Flood fill test
##    graph = G(n,p)
##    print("graph done")
##    t = time.time()
##    print(flood_fill(graph.nodes))
##    labels=[]
##    for node in graph.nodes:
##        labels.append(node.label)
##    print(labels)
##    print(time.time()-t)
    
    # Degree test   
##    print(degree_data_collection(graph.nodes))

    print(time.time()-t)
