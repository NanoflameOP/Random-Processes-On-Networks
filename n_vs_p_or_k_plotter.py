import multiprocessing as mp
import matplotlib.pyplot as plt
import math

def create_plot_p(p_range,n_range,func):
    # Create lists of desired input values for degree and diameter
    n_vals = [i for i in range(n_range[0],n_range[1]+1,n_range[2])]
    p_vals = []
    i=0
    while p_range[0]+i*p_range[2]<=p_range[1]:
        p_vals.append(p_range[0]+i*p_range[2])
        i+=1
    # Now can create output data depending on desired function
    results=[]
    for p in p_vals:
        column = []
        for n in n_vals:
            try:
                # Need np to be large (set minimum to 1000)
                m=n*p
                d=math.log(n)/math.log(n*p)
                if (d>=1) and (m>=1):
                    try:
                        if func == "conjecture 5.2":
                            column.append(((n-1)*math.log(n-1)/m))
                        elif func == "conjecture 5.1":
                            try:
                                column.append(((d/(p**2))*math.log(n)))
                            except ZeroDivisionError:
                                column.append(0)
                        else:
                            column.append(0)
                    except ValueError:
                        # Add 0 if output values simply too large as well
                        column.append(0)
                else:
                    # Add 0 if invalid inputs
                    column.append(0)
            except OverflowError:
                    # Add 0 if input values simply too large as well
                    column.append(0)
        results.append(column)
    # Now plot results
    plt.figure()
    plt.title("Colourmap of "+func)
    plt.pcolor(n_vals,p_vals,results,cmap="GnBu")
    plt.xlabel("n")
    plt.ylabel("p")
    plt.colorbar()
    plt.show()


create_plot_p([0.001,0.01,0.00001],[100000,1000000,100],"conjecture 5.2")
