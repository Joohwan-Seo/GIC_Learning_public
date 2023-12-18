import numpy as np
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotter():

    name_GIC = './analyzing_data/dataset_GIC_GCEV_wiping.pkl'
    file = open(name_GIC, 'rb')
    proposed = pickle.load(file)
    file.close()

    name_CIC = './analyzing_data/dataset_CIC_CEV_wiping.pkl'
    file = open(name_CIC, 'rb')
    benchmark = pickle.load(file)
    file.close()

    angles = ['-45','-30','0','30','45']

    print(proposed['45'].keys())

    t = proposed['-45']['t']



    for angle in angles:

        gic_xd = np.asarray(proposed[angle]['xd_list'])
        gic_x  = np.asarray(proposed[angle]['x_list'])
        gic_Fe = np.asarray(proposed[angle]['Fe_list'])

        cic_xd = np.asarray(benchmark[angle]['xd_list'])
        cic_x  = np.asarray(benchmark[angle]['x_list'])
        cic_Fe = np.asarray(benchmark[angle]['Fe_list'])

        gic_eg = np.asarray(proposed[angle]['eg_list'])

        force_profile = np.ones((t.shape[0],))

        Rd = rotmat_x(float(angle)/180*np.pi)


        # force_profile[gic_eg[:,0] > 0] = -7.5
        # force_profile[gic_eg[:,0] <= 0] = -15

        xd_mod = gic_xd

        for i in range (t.shape[0]):
            if gic_eg[i,0] > 0 or t[i] < 7.5:
                force_profile[i] = -7.5
            else:
                force_profile[i] = -15.0

            xd_mod[i,:] = gic_xd[i,:] + (Rd @ np.array([0,0,0.0225]).reshape((-1,1))).reshape((-1,))


        plt.figure(1)
        plt.subplot(3,1,1)
        plt.plot(t,gic_x[:,0],'r')
        plt.plot(t,cic_x[:,0],'k--')
        plt.plot(t,cic_xd[:,0],'b:')
        plt.ylabel('x(t), xd(t)',fontsize = 12)
        plt.legend(['GIC+GCEV','CIC+CEV','desired'],fontsize = 12)

        plt.subplot(3,1,2)
        plt.plot(t,gic_x[:,1],'r')
        plt.plot(t,cic_x[:,1],'k--')
        plt.plot(t,cic_xd[:,1],'b:')
        plt.ylabel('y(t), yd(t)',fontsize = 12)

        plt.subplot(3,1,3)
        plt.plot(t,gic_x[:,2],'r')
        plt.plot(t,cic_x[:,2],'k--')
        plt.plot(t,cic_xd[:,2],'b:')
        plt.ylabel('z(t), zd(t)',fontsize = 12)
        plt.xlabel('t (s)',fontsize = 12)

        plt.figure(2)

        plt.plot(t,gic_Fe[:,2],'r')
        plt.plot(t,cic_Fe[:,2],'k--')
        plt.plot(t,force_profile, 'b:')
        plt.ylabel('Force, Desired force')
        plt.xlabel('t (s)')
        plt.legend(['GIC+GCEV','CIC+CEV','desired'],fontsize = 12)

        plt.figure(3)
        ax = plt.axes(projection = '3d')
        ax.plot3D(gic_x[:,0],gic_x[:,1],gic_x[:,2],'r')
        ax.plot3D(cic_x[:,0],cic_x[:,1],cic_x[:,2],'k--')
        ax.plot3D(xd_mod[:,0],xd_mod[:,1],xd_mod[:,2],'b:')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')

        plt.legend(['GIC+GCEV','CIC+CEV','desired'],fontsize = 12)



        plt.show()

def rotmat_x(th):
        R = np.array([[1,0,0],
                      [0,np.cos(th),-np.sin(th)],
                      [0,np.sin(th), np.cos(th)]])

        return R

if __name__ == "__main__":

    plotter()