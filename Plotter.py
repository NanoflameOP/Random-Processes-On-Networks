# Function of this code is to plot our saved data files
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import Normal
import ast
import math
import time


##################################################################################################################
# This code requires manual modification to obtain correct results as some diameter values were computed by hand #
##################################################################################################################
                    


# Define our box plot function
def box_plot(data,desired_parameter,show_bounds=False,min_deg_test_scale=False,n_diam_test_scale=False,er_scale=False,true_er_scale=False):
    # COMPLETE!
    # Pass through data and parameter we will filter out for to plot
    filtered_data = []
    graph_type = []
    for data_set in data:
        local_data = []
        # graph_type list stores the n,p or k values of each box plot for axis values (assumes data files shared same n,p/k pairing for its results)
        if 'type' in data_set[0]:
            if (data_set[0]['type'] == "line") or (data_set[0]['type'] == "star"):
                graph_type.append([data_set[0]['n']])
        elif 'p' in data_set[0]:
            graph_type.append([data_set[0]['n'],data_set[0]['p'],'p'])
        elif 'k' in data_set[0]:
            if 'i' in data_set[0]:
                graph_type.append([data_set[0]['n'],data_set[0]['k'],'k',data_set[0]['i']])
            else:
                graph_type.append([data_set[0]['n'],data_set[0]['k'],'k'])
        for dictionary in data_set:
            if desired_parameter != 'Time taken':
                local_data.append(dictionary[desired_parameter])
            else:
                # We assume time taken is refering to 100% value for boxplots
                local_data.append((dictionary[desired_parameter][-1]))
        filtered_data.append(local_data)

    # Need average of min degrees
    min_degrees = []
    for data_set in data:
        degree = 0
        for dictionary in data_set:
            degree += dictionary['Min degree']
        degree = degree/len(data_set)
        min_degrees.append(degree)
    # Need to calculate diameter
    diameters = []
    n_values = []
    p_values = []
    k_values = []
    for data_set in data:
        n = data_set[0]['n']
        n_values.append(n)
        if 'type' in data_set[0]:
            if (data_set[0]['type'] == "line"):
                diameters.append(n-1)
            elif (data_set[0]['type'] == "star"):
                diameters.append(2)
        elif 'p' in data_set[0]:
            p = data_set[0]['p']
            p_values.append(p)
            # We use theorem 3.1 for basic diameter estimate as other theorem too complex
            diam = math.log(n)/math.log(n*p)
            diameters.append(diam)
        elif 'k' in data_set[0]:
            k = data_set[0]['k']
            k_values.append(k)
            diam = math.ceil(n/(2*k))
            if 'i' in data_set[0]:
                diam += 1
            diameters.append(diam)

    ####### If manual input of diameters needed, put them here (Calculated by hand using theorem 3.3 for accuracy)
    # Fixed n
    #diameters = [3.431,2.479,2.128,1.936,1.810]
    # Fixed p
    #diameters = [3.118,2.479,2.242,2.112]
    # Fixed n and fixed p
    #diameters = [3.431,2.479,2.128,1.936,1.810,3.118,2.479,2.242,2.112]
    # Random ER
    #diameters = [4.590,1.580,2.023,1.317,8.603,1.186,1.229]
    # All ER tests ordered by np/(n(n-1)/2)
    #diameters = [8.603,2.112,3.431,2.242,2.479,2.023,4.590,2.128,1.936,3.118,1.810,1.580,1.229,1.186,1.317]
    # All ER tests ordered by np
    #diameters = [8.603,4.590,3.431,3.118,2.479,2.242,2.128,2.023,2.112,1.936,1.810,1.580,1.317,1.186,1.229]
    # All ER tests ordered by increasing p, then decreasing n if p equal
    #diameters = [8.603,4.590,3.431,2.112,2.242,2.479,3.118,2.023,2.128,1.936,1.810,1.580,1.229,1.317,1.186]

    # If plotting on min degree scale, need to modify data
    # Simply multiplies results by average min degree
    if min_deg_test_scale == True:
        for i in range(len(filtered_data)):
            for j in range(len(filtered_data[i])):
                filtered_data[i][j] = filtered_data[i][j]*(math.sqrt(min_degrees[i]))*math.exp(-(1/(min_degrees[i]))) # Approx. COMPLETE!

    if n_diam_test_scale == True:
        for i in range(len(filtered_data)):
            for j in range(len(filtered_data[i])):
                filtered_data[i][j] = filtered_data[i][j]*(1/n_values[i])*(1/math.sqrt(n_values[i]))*(1/math.log(diameters[i])) # COMPLETE!

    if er_scale == True:
        for i in range(len(filtered_data)):
            for j in range(len(filtered_data[i])):
                filtered_data[i][j] = filtered_data[i][j]*(1/math.log(diameters[i]))*(1/(n_values[i]))*(math.sqrt(min_degrees[i]))*math.exp(-(1/(min_degrees[i]))) # COMBINES BOTH SCALES WITHOUT 1/ROOT(N)

    if true_er_scale == True:
        for i in range(len(filtered_data)):
            for j in range(len(filtered_data[i])):
                filtered_data[i][j] = (filtered_data[i][j])*(1/math.log(diameters[i]))*(1/n_values[i])*(math.sqrt(min_degrees[i]))*math.exp(-(1/(min_degrees[i])))*math.log((n_values[i]-1)/(2*p_values[i]))  # COMBINES ABOVE WITH EXTRA FORMULA 


    # If plotting time taken, plot lower bounds as well
    if (desired_parameter == 'Time taken') and (show_bounds == True):
        for i in range(len(filtered_data)):
            if i == 0:
                #plt.plot([i-0.1,i+0.1],[diameters[i],diameters[i]],color='red',label="Lower bound of proposition 2.1")
                #plt.plot([i-0.1,i+0.1],[math.log(n_values[i])/math.log(n_values[i]*p_values[i]),math.log(n_values[i])/math.log(n_values[i]*p_values[i])],color='blue',label="Lower bound of lower part of theorem 3.2 and proposition 4.1")                
                #plt.plot([i-0.1,i+0.1],[(math.log(n_values[i])/math.log(n_values[i]*p_values[i]))+2*math.log(n_values[i])*((10*((n_values[i]*p_values[i]))/((math.sqrt((n_values[i]*p_values[i]))-1)**2)+1)/((n_values[i]*p_values[i])*((n_values[i]*p_values[i])-math.log(2*(n_values[i]*p_values[i])))))+1,(math.log(n_values[i])/math.log(n_values[i]*p_values[i]))+2*math.log(n_values[i])*((10*((n_values[i]*p_values[i]))/((math.sqrt((n_values[i]*p_values[i]))-1)**2)+1)/((n_values[i]*p_values[i])*((n_values[i]*p_values[i])-math.log(2*(n_values[i]*p_values[i])))))+1],color='brown',label="Lower bound of upper part of theorem 3.2 and proposition 4.1 (c=np)")
                #plt.plot([i-0.1,i+0.1],[math.ceil((n_values[i]-1)/min_degrees[i]),math.ceil((n_values[i]-1)/min_degrees[i])],color='green',label="Lower bound of proposition 2.2")
                plt.plot([i-0.1,i+0.1],[((n_values[i]-1)*math.log(n_values[i]-1))/min_degrees[i],((n_values[i]-1)*math.log(n_values[i]-1))/min_degrees[i]],color='orange',label="Lower bound of conjecture 2.3")
                #plt.plot([i-0.1,i+0.1],[diameters[i]*math.log(n_values[i])/(p_values[i]**2),diameters[i]*math.log(n_values[i])/(p_values[i]**2)],color='purple',label="Upper bound of conjecture 5.1")
                #plt.plot([i-0.1,i+0.1],[(n_values[i]**(3/2))/(2*k_values[i]),(n_values[i]**(3/2))/(2*k_values[i])],color='blue',label="Lower bound conjecture of "+r'$\frac{n^{3/2}}{2k}$')
                #plt.plot([i-0.1,i+0.1],[n_values[i]**2,n_values[i]**2],color='blue',label="Upper bound conjecture of $n^2$")
                #plt.plot([i-0.1,i+0.1],[analytical_bound(n_values[i],p_values[i]),analytical_bound(n_values[i],p_values[i])],color='blue',label="Analytical bound")
                #plt.plot([i-0.1,i+0.1],[((n_values[i]/(2*k_values[i]))**(2)),((n_values[i]/(2*k_values[i]))**(2))],color='blue',label="Lower bound conjecture of "+r'$\left(\frac{n}{2k}\right)^2$')
                plt.plot([i-0.1,i+0.1],[((n_values[i])**(2))/(4*k_values[i]),((n_values[i])**(2))/(4*k_values[i])],color='blue',label="Upper bound conjecture of "+r'$\frac{n^{2}}{4k}$')
            else:
                #plt.plot([i-0.1,i+0.1],[diameters[i],diameters[i]],color='red')
                #plt.plot([i-0.1,i+0.1],[math.log(n_values[i])/math.log(n_values[i]*p_values[i]),math.log(n_values[i])/math.log(n_values[i]*p_values[i])],color='blue')
                #plt.plot([i-0.1,i+0.1],[(math.log(n_values[i])/math.log(n_values[i]*p_values[i]))+2*math.log(n_values[i])*((10*((n_values[i]*p_values[i]))/((math.sqrt((n_values[i]*p_values[i]))-1)**2)+1)/((n_values[i]*p_values[i])*((n_values[i]*p_values[i])-math.log(2*(n_values[i]*p_values[i])))))+1,(math.log(n_values[i])/math.log(n_values[i]*p_values[i]))+2*math.log(n_values[i])*((10*((n_values[i]*p_values[i]))/((math.sqrt((n_values[i]*p_values[i]))-1)**2)+1)/((n_values[i]*p_values[i])*((n_values[i]*p_values[i])-math.log(2*(n_values[i]*p_values[i])))))+1],color='brown')
                #plt.plot([i-0.1,i+0.1],[math.ceil((n_values[i]-1)/min_degrees[i]),math.ceil((n_values[i]-1)/min_degrees[i])],color='green')
                plt.plot([i-0.1,i+0.1],[((n_values[i]-1)*math.log(n_values[i]-1))/min_degrees[i],((n_values[i]-1)*math.log(n_values[i]-1))/min_degrees[i]],color='orange')
                #plt.plot([i-0.1,i+0.1],[diameters[i]*math.log(n_values[i])/(p_values[i]**2),diameters[i]*math.log(n_values[i])/(p_values[i]**2)],color='purple')
                #plt.plot([i-0.1,i+0.1],[(n_values[i]**(3/2))/(2*k_values[i]),(n_values[i]**(3/2))/(2*k_values[i])],color='blue')
                #plt.plot([i-0.1,i+0.1],[n_values[i]**2,n_values[i]**2],color='blue')
                #plt.plot([i-0.1,i+0.1],[analytical_bound(n_values[i],p_values[i]),analytical_bound(n_values[i],p_values[i])],color='blue')
                #plt.plot([i-0.1,i+0.1],[((n_values[i]/(2*k_values[i]))**(2)),((n_values[i]/(2*k_values[i]))**(2))],color='blue')
                plt.plot([i-0.1,i+0.1],[((n_values[i])**(2))/(4*k_values[i]),((n_values[i])**(2))/(4*k_values[i])],color='blue')
                
    # Plot
    print("Diameters: ", diameters)
    print("Min degrees: ", min_degrees)
    print("Mean average of filtered data:")
    for i in range(len(filtered_data)):
        sumation=0
        c=0
        for j in range(len(filtered_data[i])):
            c+=1
            sumation+=filtered_data[i][j]
        print(sumation/c)
    if 'type' in data[0][0]:
        plt.boxplot(filtered_data, positions=[i for i in range(len(filtered_data))], tick_labels=[str("n="+str(graph_type[i][0])) for i in range(len(graph_type))])
    else:
        plt.boxplot(filtered_data, positions=[i for i in range(len(filtered_data))], tick_labels=[str("n="+str(graph_type[i][0])+", "+"\n"+str(graph_type[i][2])+"="+str(graph_type[i][1])+", "+"\n"+"i"+"="+str(graph_type[i][3])) if len(graph_type[i])==4 else str("n="+str(graph_type[i][0])+", "+"\n"+str(graph_type[i][2])+"="+str(graph_type[i][1])) for i in range(len(graph_type))])
    if desired_parameter == 'Time taken':
        plt.title('Boxplots of time taken')
    else:
        plt.title('Boxplots of '+desired_parameter)
    plt.ylabel(desired_parameter)
    if min_deg_test_scale == True:
        plt.ylabel('Time steps taken x $\sqrt{d^{in}_{min}}e^{-1/d^{in}_{min}}$')
    if n_diam_test_scale == True:
        plt.ylabel('Time steps taken x '+ r'$\frac{1}{n^{3/2}\log(Diameter)} $')
    if er_scale == True:
        plt.ylabel('Time steps taken x '+ r'$\frac{\sqrt{d^{in}_{min}}e^{-1/d^{in}_{min}}}{n\log(Diameter)} $')
    if true_er_scale == True:
        plt.ylabel('Time steps taken x '+ r'$\frac{\sqrt{d^{in}_{min}}\log((n-1)/2p)e^{-1/d^{in}_{min}}}{n\log(Diameter)} $')
    if (desired_parameter == 'Time taken') and (show_bounds == True):
        plt.legend()
    plt.ylim(bottom=0)
    plt.show()
    

# Define our line graph function for time taken against percentage
def line_graph_time_taken(data):
    # Filter for time taken from data
    filtered_data = []
    graph_type = []
    for data_set in data:
        local_data = []
        # graph_type list stores the n,p values of each line (assumes data files shared same n,p pairing for its results)
        if 'type' in data_set[0]:
            if (data_set[0]['type'] == "line") or (data_set[0]['type'] == "star"):
                graph_type.append([data_set[0]['n']])
        elif 'p' in data_set[0]:
            graph_type.append([data_set[0]['n'],data_set[0]['p'],'p'])
        elif 'k' in data_set[0]:
            if 'i' in data_set[0]:
                graph_type.append([data_set[0]['n'],data_set[0]['k'],'k',data_set[0]['i']])
            else:
                graph_type.append([data_set[0]['n'],data_set[0]['k'],'k'])
        for dictionary in data_set:
            local_data.append(dictionary['Time taken'])
        filtered_data.append(local_data)

    # Plot
    j = 0
    for data_set in filtered_data:
        # data_set is 2D array of values that need averaging
        plot_data = [0 for i in range(len(data_set[0])+1)]
        for trial in data_set:
            for i in range(len(trial)):
                plot_data[i+1] += trial[i]
        # Divide plot_data to average after
        for i in range(len(plot_data)):
            plot_data[i] = plot_data[i]/len(data_set)
        if len(graph_type[j]) == 1:
            plt.plot(plot_data,np.arange(0,101),label=str("n="+str(graph_type[j][0])))
        elif len(graph_type[j]) == 4:
            plt.plot(plot_data,np.arange(0,101),label=str("n="+str(graph_type[j][0])+", "+"\n"+str(graph_type[j][2])+"="+str(graph_type[j][1])+", "+"\n"+"i"+"="+str(graph_type[j][3])))
        else:
            plt.plot(plot_data,np.arange(0,101),label=str("n="+str(graph_type[j][0])+", "+"\n"+str(graph_type[j][2])+"="+str(graph_type[j][1])))
        j += 1
    plt.title('Plot of time steps taken against percentage of nodes with all packets')
    plt.xlabel('Time steps')
    plt.ylabel('Percentage of nodes with all packets')
    plt.legend()
#    plt.ylim(bottom=0,top=100)
    plt.show()



def mindeg_diam_plot(data):
    # COMPLETE!
    # Pass through data and plot scatter of diam and min degree against time
    filtered_data = []
    graph_type = []
    for data_set in data:
        local_data = []
        # graph_type list stores the n,p or k values of each box plot for axis values (assumes data files shared same n,p/k pairing for its results)
        if 'type' in data_set[0]:
            if (data_set[0]['type'] == "line") or (data_set[0]['type'] == "star"):
                graph_type.append([data_set[0]['n']])
        elif 'p' in data_set[0]:
            graph_type.append([data_set[0]['n'],data_set[0]['p'],'p'])
        elif 'k' in data_set[0]:
            if 'i' in data_set[0]:
                graph_type.append([data_set[0]['n'],data_set[0]['k'],'k',data_set[0]['i']])
            else:
                graph_type.append([data_set[0]['n'],data_set[0]['k'],'k'])
        for dictionary in data_set:
            local_data.append(dictionary['Time taken'][-1])
        filtered_data.append(local_data)

    # Need average of min degrees
    min_degrees = []
    for data_set in data:
        degree = 0
        for dictionary in data_set:
            degree += dictionary['Min degree']
        degree = degree/len(data_set)
        min_degrees.append(degree)
    # Need to calculate diameter
    diameters = []
    n_values = []
    p_values = []
    k_values = []
    for data_set in data:
        n = data_set[0]['n']
        n_values.append(n)
        if 'type' in data_set[0]:
            if (data_set[0]['type'] == "line"):
                diameters.append(n-1)
            elif (data_set[0]['type'] == "star"):
                diameters.append(2)
        elif 'p' in data_set[0]:
            p = data_set[0]['p']
            p_values.append(p)
            # We use theorem 3.1 for basic diameter estimate as other theorem too complex
            diam = math.log(n)/math.log(n*p)
            diameters.append(diam)
        elif 'k' in data_set[0]:
            k = data_set[0]['k']
            k_values.append(k)
            diam = math.ceil(n/(2*k))
            if 'i' in data_set[0]:
                diam += 1
            diameters.append(diam)

    # Need to filter data further down to a single value
    average_filtration = []
    for i in range(len(filtered_data)):
        sumation=0
        c=0
        for j in range(len(filtered_data[i])):
            c+=1
            sumation+=filtered_data[i][j]
        average_filtration.append(sumation/c)

    ####### If manual input of diameters needed, put them here (Calculated by hand using theorem 3.3 for accuracy)
    # Fixed n
    #diameters = [3.431,2.479,2.128,1.936,1.810]
    # Fixed p
    #diameters = [3.118,2.479,2.242,2.112]
    # Random ER
    #diameters = [4.590,1.580,2.023,1.317,8.603,1.186,1.229]
    # All ER tests ordered by np/(n(n-1)/2)
    #diameters = [8.603,2.112,3.431,2.242,2.479,2.023,4.590,2.128,1.936,3.118,1.810,1.580,1.229,1.186,1.317]
    # All ER tests ordered by np
    #diameters = [8.603,4.590,3.431,3.118,2.479,2.242,2.128,2.023,2.112,1.936,1.810,1.580,1.317,1.186,1.229]
    # All ER tests ordered by increasing p, then decreasing n if p equal
    #diameters = [8.603,4.590,3.431,2.112,2.242,2.479,3.118,2.023,2.128,1.936,1.810,1.580,1.229,1.317,1.186]
                
    # Plot
    print("Diameters: ", diameters)
    print("Min degrees: ", min_degrees)
    print("Mean average of filtered data: ", average_filtration)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(diameters,min_degrees,average_filtration,marker='x')
    plt.title('Scatter graph of time steps taken, minimum degree and diameter')
    ax.set_ylabel("Minimum Degree")
    ax.set_xlabel("Diameter")
    ax.set_zlabel("Time Steps Taken")
    plt.show()



def np_diam_plot(data):
    # COMPLETE!
    # Pass through data and plot scatter of n and p against time
    filtered_data = []
    n_vals = []
    p_vals = []
    for data_set in data:
        local_data = []
        n_vals.append(data_set[0]['n'])
        p_vals.append(data_set[0]['p'])
        for dictionary in data_set:
            local_data.append(dictionary['Time taken'][-1])
        filtered_data.append(local_data)

    # Need to filter data further down to a single value
    average_filtration = []
    for i in range(len(filtered_data)):
        sumation=0
        c=0
        for j in range(len(filtered_data[i])):
            c+=1
            sumation+=filtered_data[i][j]
        average_filtration.append(sumation/c)

    # Plot
    print("n values: ", n_vals)
    print("p values: ", p_vals)
    print("Mean average of filtered data: ", average_filtration)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(n_vals,p_vals,average_filtration,marker='x')
    plt.title('Scatter graph of time steps taken, n and p')
    ax.set_ylabel("p")
    ax.set_xlabel("n")
    ax.set_zlabel("Time Steps Taken")
    plt.show()



# Define our box plot function
def log_box_plot(data,desired_parameter,show_bounds=False):
    # COMPLETE! Copied version of box plot but with logarithmic scale and logarithmic bounds
    # Pass through data and parameter we will filter out for to plot
    filtered_data = []
    graph_type = []
    for data_set in data:
        local_data = []
        # graph_type list stores the n,p or k values of each box plot for axis values (assumes data files shared same n,p/k pairing for its results)
        if 'type' in data_set[0]:
            if (data_set[0]['type'] == "line") or (data_set[0]['type'] == "star"):
                graph_type.append([data_set[0]['n']])
        elif 'p' in data_set[0]:
            graph_type.append([data_set[0]['n'],data_set[0]['p'],'p'])
        elif 'k' in data_set[0]:
            if 'i' in data_set[0]:
                graph_type.append([data_set[0]['n'],data_set[0]['k'],'k',data_set[0]['i']])
            else:
                graph_type.append([data_set[0]['n'],data_set[0]['k'],'k'])
        for dictionary in data_set:
            if desired_parameter != 'Time taken':
                local_data.append(dictionary[desired_parameter])
            else:
                # We assume time taken is refering to 100% value for boxplots
                local_data.append(math.log(dictionary[desired_parameter][-1]))
        filtered_data.append(local_data)

    # Need average of min degrees
    min_degrees = []
    for data_set in data:
        degree = 0
        for dictionary in data_set:
            degree += dictionary['Min degree']
        degree = degree/len(data_set)
        min_degrees.append(degree)
    # Need to calculate diameter
    diameters = []
    n_values = []
    p_values = []
    k_values = []
    for data_set in data:
        n = data_set[0]['n']
        n_values.append(n)
        if 'type' in data_set[0]:
            if (data_set[0]['type'] == "line"):
                diameters.append(n-1)
            elif (data_set[0]['type'] == "star"):
                diameters.append(2)
        elif 'p' in data_set[0]:
            p = data_set[0]['p']
            p_values.append(p)
            # We use theorem 3.1 for basic diameter estimate as other theorem too complex
            diam = math.log(n)/math.log(n*p)
            diameters.append(diam)
        elif 'k' in data_set[0]:
            k = data_set[0]['k']
            k_values.append(k)
            diam = math.ceil(n/(2*k))
            if 'i' in data_set[0]:
                diam += 1
            diameters.append(diam)

    ####### If manual input of diameters needed, put them here (Calculated by hand using theorem 3.3 for accuracy)
    # Fixed n
    #diameters = [3.431,2.479,2.128,1.936,1.810]
    # Fixed p
    #diameters = [3.118,2.479,2.242,2.112]
    # Random ER
    #diameters = [4.590,1.580,2.023,1.317,8.603,1.186,1.229]
    # All ER tests ordered by np/(n(n-1)/2)
    #diameters = [8.603,2.112,3.431,2.242,2.479,2.023,4.590,2.128,1.936,3.118,1.810,1.580,1.229,1.186,1.317]
    # All ER tests ordered by np
    diameters = [8.603,4.590,3.431,3.118,2.479,2.242,2.128,2.023,2.112,1.936,1.810,1.580,1.317,1.186,1.229]
    # All ER tests ordered by increasing p, then decreasing n if p equal
    #diameters = [8.603,4.590,3.431,2.112,2.242,2.479,3.118,2.023,2.128,1.936,1.810,1.580,1.229,1.317,1.186]


    # If plotting time taken, plot lower bounds as well
    if (desired_parameter == 'Time taken') and (show_bounds == True):
        for i in range(len(filtered_data)):
            if i == 0:
                #plt.plot([i-0.1,i+0.1],[math.log(diameters[i]),math.log(diameters[i])],color='red',label="Lower bound of proposition 2.1")
                #plt.plot([i-0.1,i+0.1],[math.log(math.log(n_values[i])/math.log(n_values[i]*p_values[i])),math.log(math.log(n_values[i])/math.log(n_values[i]*p_values[i]))],color='blue',label="Lower bound of lower part of theorem 3.2 and proposition 4.1")                
                #plt.plot([i-0.1,i+0.1],[math.log((math.log(n_values[i])/math.log(n_values[i]*p_values[i]))+2*math.log(n_values[i])*((10*((n_values[i]*p_values[i]))/((math.sqrt((n_values[i]*p_values[i]))-1)**2)+1)/((n_values[i]*p_values[i])*((n_values[i]*p_values[i])-math.log(2*(n_values[i]*p_values[i])))))+1),math.log((math.log(n_values[i])/math.log(n_values[i]*p_values[i]))+2*math.log(n_values[i])*((10*((n_values[i]*p_values[i]))/((math.sqrt((n_values[i]*p_values[i]))-1)**2)+1)/((n_values[i]*p_values[i])*((n_values[i]*p_values[i])-math.log(2*(n_values[i]*p_values[i])))))+1)],color='brown',label="Lower bound of upper part of theorem 3.2 and proposition 4.1 (c=np)")
                #plt.plot([i-0.1,i+0.1],[math.log(math.ceil((n_values[i]-1)/min_degrees[i])),math.log(math.ceil((n_values[i]-1)/min_degrees[i]))],color='green',label="Lower bound of proposition 2.2")
                #plt.plot([i-0.1,i+0.1],[math.log(((n_values[i]-1)*math.log(n_values[i]-1))/min_degrees[i]),math.log(((n_values[i]-1)*math.log(n_values[i]-1))/min_degrees[i])],color='orange',label="Lower bound of conjecture 2.3")
                plt.plot([i-0.1,i+0.1],[math.log(diameters[i]*math.log(n_values[i])/(p_values[i]**2)),math.log(diameters[i]*math.log(n_values[i])/(p_values[i]**2))],color='purple',label="Upper bound of conjecture 5.1")
                #plt.plot([i-0.1,i+0.1],[math.log((n_values[i]**(2))/(2*k_values[i])),math.log((n_values[i]**(2))/(2*k_values[i]))],color='blue',label="Upper bound conjecture of "+r'$\frac{n^{2}}{2k}$')
                #plt.plot([i-0.1,i+0.1],[math.log(n_values[i]**2),math.log(n_values[i]**2)],color='blue',label="Upper bound conjecture of $n^2$")
                plt.plot([i-0.1,i+0.1],[math.log(analytical_bound(n_values[i],p_values[i])),math.log(analytical_bound(n_values[i],p_values[i]))],color='blue',label="Analytical bound")
                #plt.plot([i-0.1,i+0.1],[math.log(((n_values[i])**(2))/(4*k_values[i])),math.log(((n_values[i])**(2))/(4*k_values[i]))],color='blue',label="Upper bound conjecture of "+r'$\frac{n^{2}}{4k}$')
                #plt.plot([i-0.1,i+0.1],[math.log(((n_values[i])**(2))/(k_values[i]**2)),math.log(((n_values[i])**(2))/(k_values[i]**2))],color='blue',label="Upper bound conjecture of "+r'$\left(\frac{n}{k}\right)^2$')
            else:
                #plt.plot([i-0.1,i+0.1],[math.log(diameters[i]),math.log(diameters[i])],color='red')
                #plt.plot([i-0.1,i+0.1],[math.log(math.log(n_values[i])/math.log(n_values[i]*p_values[i])),math.log(math.log(n_values[i])/math.log(n_values[i]*p_values[i]))],color='blue')                
                #plt.plot([i-0.1,i+0.1],[math.log((math.log(n_values[i])/math.log(n_values[i]*p_values[i]))+2*math.log(n_values[i])*((10*((n_values[i]*p_values[i]))/((math.sqrt((n_values[i]*p_values[i]))-1)**2)+1)/((n_values[i]*p_values[i])*((n_values[i]*p_values[i])-math.log(2*(n_values[i]*p_values[i])))))+1),math.log((math.log(n_values[i])/math.log(n_values[i]*p_values[i]))+2*math.log(n_values[i])*((10*((n_values[i]*p_values[i]))/((math.sqrt((n_values[i]*p_values[i]))-1)**2)+1)/((n_values[i]*p_values[i])*((n_values[i]*p_values[i])-math.log(2*(n_values[i]*p_values[i])))))+1)],color='brown')
                #plt.plot([i-0.1,i+0.1],[math.log(math.ceil((n_values[i]-1)/min_degrees[i])),math.log(math.ceil((n_values[i]-1)/min_degrees[i]))],color='green')
                #plt.plot([i-0.1,i+0.1],[math.log(((n_values[i]-1)*math.log(n_values[i]-1))/min_degrees[i]),math.log(((n_values[i]-1)*math.log(n_values[i]-1))/min_degrees[i])],color='orange')
                plt.plot([i-0.1,i+0.1],[math.log(diameters[i]*math.log(n_values[i])/(p_values[i]**2)),math.log(diameters[i]*math.log(n_values[i])/(p_values[i]**2))],color='purple')
                #plt.plot([i-0.1,i+0.1],[math.log((n_values[i]**(2))/(2*k_values[i])),math.log((n_values[i]**(2))/(2*k_values[i]))],color='blue')
                #plt.plot([i-0.1,i+0.1],[math.log(n_values[i]**2),math.log(n_values[i]**2)],color='blue')
                plt.plot([i-0.1,i+0.1],[math.log(analytical_bound(n_values[i],p_values[i])),math.log(analytical_bound(n_values[i],p_values[i]))],color='blue')
                #plt.plot([i-0.1,i+0.1],[math.log(((n_values[i])**(2))/(4*k_values[i])),math.log(((n_values[i])**(2))/(4*k_values[i]))],color='blue')
                #plt.plot([i-0.1,i+0.1],[math.log(((n_values[i])**(2))/(k_values[i]**2)),math.log(((n_values[i])**(2))/(k_values[i]**2))],color='blue')
                
    # Plot
    print("Diameters: ", diameters)
    print("Min degrees: ", min_degrees)

    if 'type' in data[0][0]:
        plt.boxplot(filtered_data, positions=[i for i in range(len(filtered_data))], tick_labels=[str("n="+str(graph_type[i][0])) for i in range(len(graph_type))])
    else:
        alpha_list = [0.66,0.71,0.75,0.78,0.8,0.8,0.8,0.8,0.8]
        plt.boxplot(filtered_data, positions=[i for i in range(len(filtered_data))], tick_labels=[str("n="+str(graph_type[i][0])+", "+"\n"+r'$\alpha$'+"="+str(alpha_list[i])) for i in range(len(graph_type))])
        #plt.boxplot(filtered_data, positions=[i for i in range(len(filtered_data))], tick_labels=[str("n="+str(graph_type[i][0])+", "+"\n"+str(graph_type[i][2])+"="+str(graph_type[i][1])+", "+"\n"+"i"+"="+str(graph_type[i][3])) if len(graph_type[i])==4 else str("n="+str(graph_type[i][0])+", "+"\n"+str(graph_type[i][2])+"="+str(graph_type[i][1])) for i in range(len(graph_type))])
    if desired_parameter == 'Time taken':
        plt.title('Boxplots of time taken on a logarithmic scale')
    else:
        plt.title('Boxplots of '+desired_parameter+ ' on a logarithmic scale')
    plt.ylabel(desired_parameter)
    if (desired_parameter == 'Time taken') and (show_bounds == True):
        plt.legend()
    #plt.ylim(bottom=0)
    plt.show()



def analytical_bound(n,p,res=0.01):
    answer = 1
    # Start by computing all Z_i
    Z_list=[round((n-1)*p)]
    Zvar_list=[0]
    while True:
        new_z = round((n-1-sum(Z_list))*(1-(1-p)**Z_list[-1]))
        var_z = ((n-1-sum(Z_list))*(1-(1-p)**Z_list[-1])*((1-p)**Z_list[-1]))
        if new_z>=1:
            Z_list.append(new_z)
            Zvar_list.append(var_z)
        else:
            break
    print(Z_list)
    #print("Variance of Z values: ", Zvar_list)
    
    # Formulate complete list of m until n (afterwards any instance of it will just be n)
    m_list = [1]
    while True:
        # Calculate maximised delta (s)
        s = (n-m_list[-1])
        X = Normal(mu=n-m_list[-1], sigma=math.sqrt((n-m_list[-1])*((n-1)/(n**2))))
        while True:
            prob = 1-X.cdf(s)
            if (prob < 1/(n*p)):
                break
            else:
                s += res
        y = min(m_list[-1],s)
        # Calculate maximised gamma (g)
        g = n*p
        l = y/m_list[-1]
        X = Normal(mu=n*p, sigma=math.sqrt(n*p*l*(1-l)))
        if math.sqrt(n*p*l*(1-l)) == 0:
            pass
        else:
            while True:
                prob = 1-X.cdf(g)
                if (prob < (1/n)):
                    break
                else:
                    g += res
        new_m = min(n,m_list[-1]+g)
        if (new_m<n-res) and (new_m!=m_list[-1]):
            m_list.append(new_m)
        else:
            if (new_m==m_list[-1]):
                print("Breakage early due to low resolution")
            break
    print(len(m_list))
    #print(m_list)
    
    for i in range(0,len(Z_list)-1):
        # Find k_i (and l)
        l = 1
        while True:
            prob_less_than_l = 0
            for j in range(0,l):
                prob_less_than_l += (math.factorial(Z_list[i]-1)/(math.factorial((Z_list[i]-1)-j)*math.factorial(j)))*(p**j)*((1-p)**((Z_list[i]-1)-j))
            # Include -0.0000001 incase Z_list[i+1]=1
            if prob_less_than_l < 1/Z_list[i+1]-0.0000001:
                l += 1
            else:
                l -= 1
                break
        k = l+1
        print("k is: ", k)

        # Formualte complete list of q_i from answer onwards
        q_list = []
        for t in range(answer,len(m_list)):
            m = m_list[t]
            q_list.append(1-(1-(1/m))**k)
        # Finish with n packet prob at end
        q_list.append(1-(1-(1/n))**k)
            
        # Find T_{i+1}-T_i
        tail = 1
        cum_prob = 0
        while True:
            prob = 1
            for j in range(0,tail-1):
                prob *= (1-q_list[min(j,len(q_list)-1)])
            prob *= q_list[min(tail-1,len(q_list)-1)]
            cum_prob += prob
            if (1-cum_prob) < 1/Z_list[i+1]:
                break
            else:
                tail += 1
        # Append tail to answer
        answer += tail
        print("Upper tail of T_"+str(i+2)+"-T_"+str(i+1)+" is: ", tail)
    return answer



    


if __name__=='__main__':
    # Define files we wish to plot data from
    # Fixed n ER
    #files = ['data_collection_41','data_collection_42','data_collection_43','data_collection_44','data_collection_46']
    # Fixed p ER
    #files = ['data_collection_4','data_collection_42','data_collection_45','data_collection_47']
    # Fixed n and fixed p
    #files = ['data_collection_41','data_collection_42','data_collection_43','data_collection_44','data_collection_46','data_collection_4','data_collection_42','data_collection_45','data_collection_47']
    # Diam tests
    #files = ['data_collection_48','data_collection_49','data_collection_50','data_collection_51','data_collection_52','data_collection_53']
    # Min degree tests
    #files = ['data_collection_7','data_collection_8','data_collection_9','data_collection_10','data_collection_11','data_collection_12']
    # Min degree test 2 (world)
    #files = ['data_collection_28','data_collection_29','data_collection_30','data_collection_31']
    # n tests
    #files = ['data_collection_18','data_collection_19','data_collection_20','data_collection_21','data_collection_22']
    # n test 2
    #files = ['data_collection_24','data_collection_25','data_collection_26','data_collection_27']
    # Random ER graph test
    #files = ['data_collection_54','data_collection_55','data_collection_56','data_collection_57','data_collection_58','data_collection_59','data_collection_60']
    # All ER tests ordered by np/(n(n-1)/2)
    #files = ['data_collection_58','data_collection_47','data_collection_41','data_collection_45','data_collection_42','data_collection_56','data_collection_54','data_collection_43','data_collection_44','data_collection_4','data_collection_46','data_collection_55','data_collection_60','data_collection_59','data_collection_57']
    # All ER tests ordered by np
    #files = ['data_collection_58','data_collection_54','data_collection_41','data_collection_4','data_collection_42','data_collection_45','data_collection_43','data_collection_56','data_collection_47','data_collection_44','data_collection_46','data_collection_55','data_collection_57','data_collection_59','data_collection_60']
    # All ER tests ordered by increasing p, then decreasing n if p equal
    #files = ['data_collection_58','data_collection_54','data_collection_41','data_collection_47','data_collection_45','data_collection_42','data_collection_4','data_collection_56','data_collection_43','data_collection_44','data_collection_46','data_collection_55','data_collection_60','data_collection_57','data_collection_59']
    # Line graphs
    #files = ['data_collection_64','data_collection_65','data_collection_66','data_collection_67','data_collection_68']
    # Star graphs
    #files = ['data_collection_69','data_collection_70','data_collection_71','data_collection_72','data_collection_73']
    # K-NN fixed n test
    #files = ['data_collection_81','data_collection_77','data_collection_76','data_collection_75','data_collection_74']
    # K-NN fixed k test
    #files = ['data_collection_48','data_collection_49','data_collection_50','data_collection_51','data_collection_52','data_collection_53']
    # Sparse tests with p=n^-a and a=0.8
    #files = ['data_collection_82','data_collection_83','data_collection_84','data_collection_85','data_collection_86']
    # Sparse tests with p=n^-a and n=1000
    #files = ['data_collection_87','data_collection_88','data_collection_89','data_collection_90','data_collection_82']
    # Sparse tests combined
    files = ['data_collection_87','data_collection_88','data_collection_89','data_collection_90','data_collection_82','data_collection_83','data_collection_84','data_collection_85','data_collection_86']

    # Open each file and combine all results into a single 2d array with dict contents
    # i.e. [[{},{},{}],[{},{},{}],...]
    data = []
    for file in files:
        with open(file+'.txt') as f:
            d = f.read()
            # Eval is needed to convert txt to list data type
            data.append(ast.literal_eval(d))

    # Define the function we wish to perform
    #box_plot(data,'Min degree')
    #box_plot(data,'Max degree')
    #box_plot(data,'Average degree')
    #box_plot(data,'Number isolated nodes removed')
    #box_plot(data,'Number of disconnected graphs removed')
    #box_plot(data,'Average size of disconnected graph removed')
    #box_plot(data,'Size of target graph')
    #box_plot(data,'Time taken',True)
    log_box_plot(data,'Time taken',True)
    #mindeg_diam_plot(data)
    #np_diam_plot(data)
    #line_graph_time_taken(data)
    #print(analytical_bound(10000,10000**(-0.6)))
