#---------------------------------------
#Since : Jun/17/2012
#UpdatBe: 2013/05/15
# -*- coding: utf-8 -*-
# Using Growing Neural Gas 1995
#---------------------------------------
from PIL import Image
import numpy as np
import random
import math as mt
import pylab as pl
import networkx as nx
from scipy import ndimage
import sys

class Skeletonization():
    def __init__(self):
        # Parameters
        # max of units
        self.NUM = 20
        # the number of delete processes
        self.END = 200
        # the number of learning
        self.SET = 400

        # tuning process
        self.ADDSET = 1000

        # Learning coefficient
        self.Ew = 0.2
        # Learning coefficient of neighbors
        self.En = 0.2

        # threshold to remove a edge
        self.AMAX = self.END*self.SET/10000

        #rewiring threshold
        self.RETH = 0.2

        # threshold to select near neurons
        self.NEAR = 0.05

        # Valuables
        self.units = np.zeros((self.NUM,2))
        self.sumerror = np.zeros(self.NUM)
        self.g_units = nx.Graph()


    def set_file(self, filename):
        # imput an image
        self.ifilename = filename
        img = Image.open(self.ifilename)
        # convert to gray scale
        img = img.convert("L")
        img = ndimage.rotate(img, -90)
        img = pl.fliplr(img)
        self.imarray = np.asarray(img) > 128

        self.MAXX = self.imarray.shape[0]
        self.MAXY = self.imarray.shape[1]

        # threshold for birth of a neuron
        self.ERROR = mt.hypot(self.MAXX, self.MAXY) * 0.12 * self.SET/self.NUM

    def beta(self, A, ac, end):
        a =  (A * (1.0 - float(ac)/float(end)) )
        return a

    def GNG(self):
        # GNG process

        # initialize
        # set two neuron
        self.units += float("inf")
        self.units[0] = self.MAXX/2.0  - self.MAXX*0.2, self.MAXY/2.0  - self.MAXY*0.2
        self.units[1] = self.MAXX/2.0  + self.MAXX*0.2, self.MAXY/2.0  + self.MAXY*0.2

        self.g_units.add_node(0)
        self.g_units.add_node(1)
        self.g_units.add_edge(0,1,weight=0)

        # choose the cell that is not 0.
        self.on_chara = []
        for i in range(self.MAXX):
            for j in range(self.MAXY):
                if(self.imarray[i][j]):
                    self.on_chara.append([i,j])

        for t in range(self.END):
            for n in range(self.SET):
                num = np.random.randint(len(self.on_chara))
                x = float(self.on_chara[num][0])
                y = float(self.on_chara[num][1])

                temp_pos = self.units.copy()
                temp_pos -= [x,y]
                temp_pos2 = temp_pos * temp_pos
                dists = temp_pos2[:, 0] + temp_pos2[:, 1]
                min_unit_num = dists.argmin()
                sec_min_unit_num = dists.argsort()[1]

                # Learning
                self.units[min_unit_num] += self.beta(self.Ew, t*self.SET + n, self.END*self.SET) * ([x,y] - self.units[min_unit_num])
                self.sumerror[min_unit_num] += np.linalg.norm([x,y] - self.units[min_unit_num])


                # Connect NN and second NN with each other
                flag = 0
                for e in self.g_units.edges():
                    if min_unit_num in e and sec_min_unit_num in e:
                        flag = 1
                        break
                    else:
                        flag = 0

                if flag == 1:
                    self.g_units[min_unit_num][sec_min_unit_num]['weight'] -= 2
                else:
                    self.g_units.add_edge(min_unit_num,sec_min_unit_num,weight=0)

                # Process for neighbors
                for i in list(self.g_units.neighbors(min_unit_num)):
                    self.units[i] += self.En * self.beta(self.Ew, t*self.SET + n, self.END*self.SET) * ([x,y] - self.units[i])

                    self.g_units[min_unit_num][i]['weight'] += 1

                    if self.g_units[min_unit_num][i]['weight'] > self.AMAX:
                        self.g_units.remove_edge(min_unit_num,i)
                        if self.g_units.degree(i) == 0:
                            self.g_units.remove_node(i)
                            self.units[i] = float("inf"), float("inf")
                            self.sumerror[i]=0

            if self.sumerror.max() > self.ERROR:
                max_error_unit_num = self.sumerror.argmax()

                temp_pos = self.units.copy()
                temp_pos -= self.units[max_error_unit_num]
                temp_pos2 = temp_pos * temp_pos
                dists = temp_pos2[:, 0] + temp_pos2[:, 1]

                far_unit_num = max_error_unit_num
                for i in self.g_units.neighbors(max_error_unit_num):
                    if dists[far_unit_num] < dists[i]:
                        far_unit_num = i

                for i in range(self.NUM):
                    if self.units[i][0] == float("inf"):
                        self.units[i] = (self.units[max_error_unit_num] + self.units[far_unit_num])/2.0
                        self.g_units.add_node(i)
                        self.g_units.remove_edge(max_error_unit_num,far_unit_num)
                        self.g_units.add_edge(i,max_error_unit_num,weight=0)
                        self.g_units.add_edge(i,far_unit_num,weight=0)
                        break

            self.sumerror = np.zeros(self.NUM)

    def Rewiring(self):
        #------------------------------------------
        # rewiring

        self.g=nx.Graph()

        infnum=[]
        for i in range(self.NUM):
            if self.units[i][0] == float("inf"):
                infnum.append(i)

        for i in range(len(infnum)):
            self.units=np.delete(self.units, infnum[-i-1], 0)

        units_num =  self.units.shape[0]

        for i in range(units_num):
            self.g.add_node(i)

        for i in range(units_num):
            opponent = []
            distsi = []
            distsj = []

            #Calculate distance from node i
            temp_pos = self.units.copy()
            temp_pos -= self.units[i]
            temp_pos2 = temp_pos * temp_pos
            distsi = temp_pos2[:, 0] + temp_pos2[:, 1]

            for j in range(units_num):
                if j != i:
                    candidatei = []
                    candidatej = []

                    #Calculate distance from node j
                    temp_pos = self.units.copy()
                    temp_pos -= self.units[j]
                    temp_pos2 = temp_pos * temp_pos
                    distsj = temp_pos2[:, 0] + temp_pos2[:, 1]

                    #distance between node i and j
                    dist2 = np.linalg.norm(self.units[i] - self.units[j])**2

                    if dist2 < (mt.hypot(self.MAXX,self.MAXY)*self.RETH)**2:
                        #choose nodes whose distance from i is less than distance between node i and j
                        for k in range(units_num):
                            if k!=i and k!=j and distsi[k] < dist2:
                                candidatei.append(k)
                            if k!=i and k!=j and distsj[k] < dist2:
                                candidatej.append(k)

                        flag = 0
                        if len(candidatei) == 0:
                            # node j is nearest from node i
                            opponent.append(j)
                        else:
                            flag = 1
                            if len(candidatej) != 0:
                                for k in candidatei:
                                    for l in candidatej:
                                        if l == k:
                                            flag = 0
                            else:
                                flag = 0

                            if flag == 1:
                                opponent.append(j)

            for j in opponent:
                self.g.add_edge(i, j)

    def fit(self):
        self.GNG()
        self.Rewiring()


    def output_img(self, imgfm = "png"):
        self.ofilename = self.ifilename.split(".")[0] + "_graph." + imgfm

        pimage = np.zeros((self.MAXY, self.MAXX))

        for i in self.on_chara:
            pimage[i[1], i[0]] = 255

        img=pl.imread(self.ifilename)
        pl.imshow(pimage, cmap=pl.gray())
        nx.draw_networkx_nodes(self.g,self.units,node_size=200,node_color=(0.5,1,1))
        nx.draw_networkx_edges(self.g,self.units,width=10,edge_color='b',alpha=0.5)
        #nx.draw(g,units)
        pl.savefig(self.ofilename)

if __name__ == '__main__':
   filename =  sys.argv[1]
   sk = Skeletonization()
   sk.set_file(filename)
   sk.fit()
   sk.output_img()
