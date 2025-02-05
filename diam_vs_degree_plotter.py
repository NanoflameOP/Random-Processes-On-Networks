import multiprocessing as mp
import matplotlib.pyplot as plt
import math

def create_plot(diam_range,deg_range,func):
    # Create lists of desired input values for degree and diameter
    deg_vals = [i for i in range(deg_range[0],deg_range[1]+1,deg_range[2])]
    diam_vals = []
    i=0
    while diam_range[0]+i*diam_range[2]<=diam_range[1]:
        diam_vals.append(diam_range[0]+i*diam_range[2])
        i+=1
    # Now can create output data depending on desired function
    results=[]
    for d in diam_vals:
        column = []
        for m in deg_vals:
            try:
                n = int(math.exp(d*math.log(m)))
                p = float(m/n)
                # Check n and p are valid inputs
                if (n>=3) and (0<p) and (p<1):
                    # If valid add result of desired function
                    # Use logarithmic scaling to avoid large values making smaller ones indistinguishable
                    try:
                        if func == "posterior prediction":
                            column.append(math.log((26*math.exp(1/m)*n*math.log(d))/(math.log((n-1)/(2*p))*math.sqrt(m))))
                        elif func == "conjecture 5.2":
                            column.append(math.log(((n-1)*math.log(n-1))/m))
                        elif func == "conjecture 5.1":
                            try:
                                column.append(math.log((d*math.log(n))/(p**2)))
                            except ZeroDivisionError:
                                column.append(0)
                        elif func == "n":
                            column.append(math.log(n))
                        elif func == "1/p":
                            column.append(math.log(1/p))
                        else:
                            column.append(0)
                    except ValueError:
                        # Add 0 if output values simply too large as well
                        column.append(0)
                else:
                    # If not valid inputs, add result of T=0
                    column.append(0)
            except OverflowError:
                    # Add 0 if input values simply too large as well
                    column.append(0)
        results.append(column)
    # Now plot results
    plt.figure()
    plt.title("Colourmap of "+func+ " on a logarithmic scale")
    plt.pcolor(deg_vals,diam_vals,results,cmap="GnBu")
    plt.xlabel("Minimum Degree")
    plt.ylabel("Diameter")
    plt.colorbar()
    plt.show()

#create_plot([1,200,0.1],[1,100000,10],"posterior prediction")
#create_plot([1,200,0.1],[1,100000,10],"conjecture 5.2")
#create_plot([1,200,0.1],[1,100000,10],"conjecture 5.1")
#create_plot([1,600,0.1],[1,100,1],"posterior prediction")
#create_plot([1,1200,0.1],[1,100,1],"conjecture 5.2")
#create_plot([1,600,0.1],[1,100,1],"conjecture 5.1")
#create_plot([1,50,0.1],[1,100,1],"posterior prediction")
#create_plot([1,50,0.1],[1,100,1],"conjecture 5.2")
#create_plot([1,50,0.1],[1,100,1],"conjecture 5.1")
#create_plot([1,100,0.1],[10000,100000,10],"posterior prediction")
#create_plot([1,100,0.1],[10000,100000,10],"conjecture 5.2")
create_plot([1,100,0.1],[10000,100000,10],"conjecture 5.1")
#create_plot([1,100,0.1],[10000,100000,10],"n")
#create_plot([1,100,0.1],[10000,100000,10],"1/p")
#create_plot([1,50,0.1],[100000,1000000,100],"conjecture 5.2")
#create_plot([1,50,0.1],[100000,1000000,100],"conjecture 5.1")
#create_plot([1,50,0.1],[100000,1000000,100],"n")
