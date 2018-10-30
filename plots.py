from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np

def featuretransform(X, degree):
    d= X.shape[1]-1;
    # print (((degree+1)*(degree+2) )/2 - 1 )
    tra_X= np.ones( (X.shape[0], round( (degree+1)*(degree+2) /2 )  ) );

    for i in range(1,degree+1):
        for j in range(i+1):
            # print( 1+( i*(i+1)/2 -1 ) +j , "   ", i,"   ",j );
            tra_X[:, 1+ round( i*(i+1)/2 -1 ) +j ] =  np.multiply( np.power(X[:,2],j), np.power(X[:,1],i-j) ) ;

    return tra_X


def scatter_plot(X, Y,i,title=""):
    # Generating a Gaussion dataset:
    # creating random vectors from the multivariate normal distribution
    # given mean and covariance

    x1_samples = X[ np.where(Y == 1),: ][0]
    x2_samples = X[ np.where(Y == 0),: ][0]

    #print( X.shape," \n   ", Y.shape," \n ",x1_samples.shape," \n   ",x2_samples.shape);

    plt.figure(i)

    plt.scatter(x1_samples[:,0], x1_samples[:, 1], marker='x',
                color='blue', label=' Class 1')
    plt.scatter(x2_samples[:, 0], x2_samples[:, 1], marker='o',
                color='green', label=' Class 0')
    plt.title('DataSet')
    plt.ylabel('X1')
    plt.xlabel('X2')
    plt.legend(loc='upper right')
    plt.draw()



def scatter_plot_com(ax,x,y_1,lx,ly ,i,frac,leg=[],title="",log=False):
    color=['r', 'b', 'g', 'k', 'm']

    # cmap = plt.cm.viridis
    # patches = [mpatches.Patch(color=cmap(v), label=k) for k, v in my_colors.items()]
    #
    # plt.legend(handles=patches)
    ax.scatter(x, y_1,color=color[frac])
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    if(log):
        ax.set_xscale('log');

    if(title!=""):
        ax.set_title(title)
    else:
        ax.set_title('line plot')

    ax.legend( leg, loc='upper right')

def linear_plot_com(ax,x,y_1,lx,ly ,i,frac,leg=[],title="",log=False):
    color=['r', 'b', 'g', 'k', 'm']

    # cmap = plt.cm.viridis
    # patches = [mpatches.Patch(color=cmap(v), label=k) for k, v in my_colors.items()]
    #
    # plt.legend(handles=patches)
    ax.plot(x, y_1,color=color[frac])
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    if(log):
        ax.set_xscale('log');

    if(title!=""):
        ax.set_title(title)
    else:
        ax.set_title('line plot')

    ax.legend( leg, loc='upper right')


def linear_plot(x,y_1,y2,lx,ly ,title,i,log=False ):
    plt.figure(i)
    plt.plot(x, y_1)
    plt.plot(x, y2)
    plt.xlabel(lx)
    plt.ylabel(ly)
    if(log):
        plt.xscale('log');

    plt.title('Simple line plot')
    plt.legend(['Train Error', 'Test Error'], loc='upper right')
    plt.title(title)
    plt.draw()


def plot_descion_linear(x,w,lx,ly ,i,log=False ):
    plt.figure(i)
    print(w)
    intr = np.sum( w[0] / -w[2]);
    slp = np.sum(w[1] /-w[2]) ;
    print(  slp,intr );

    print( [ min(x[:,1]), max(x[:,1]) ] );
    print( [ slp* min(x[:,1]) + intr , slp*max(x[:,1]) + intr ] );

    plt.plot( x[:,1] , intr + slp* x[:,1] , marker='x')
    plt.xlabel(lx)
    plt.ylabel(ly)
    if(log):
        plt.xscale('log');

    plt.title('Descison plot')
    plt.legend(['sample 1'], loc='upper left')

    plt.draw()

## input x is transfromed already
def plot_descion(x,w,degree,lx,ly ,i,log=False,lamda=0 ):

    plt.figure(i)
    u = np.linspace(min(x[:,1])-5, max(x[:,1])+5, 5000);
    v = np.linspace(min(x[:,2])-5, max(x[:,2])+5, 5000);
    n = u.shape[0]**2;
    xx, yy = np.meshgrid(u,v)
    # print(np.ones((n, 1)) )
    # print(xx.ravel())
    # print(yy.ravel())

    z = np.dot( featuretransform( np.c_[ np.ones((n,1)), xx.ravel(), yy.ravel() ] ,degree ), w ) ;
    z=z.T;
    z = z.reshape(xx.shape);
    ##3 important to transpose z before calling contour

    plt.contour(u, v, z, cmap=plt.cm.Paired ,levels=0)
    if(lamda==0):
        plt.title('Descison plot for degree= '+ str(degree))
    else:
        plt.title('Descison plot for degree= '+ str(degree) + " lambda=" +str(lamda) )

    plt.legend(['sample 1'], loc='upper left')

    ## savefig for each fration test and train errors.
    if(lamda==0):
        plt.savefig('decsion_b ' + str(degree) + '.png')
    else:
        plt.savefig('decsion_b ' + str(degree) + "l_"  +str(lamda)+  '.png')


    plt.draw()



#linear_plot([1 ,2 ,3],[1 ,2 ,3]);