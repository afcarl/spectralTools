from scipy.optimize import curve_fit
from matplotlib.widgets import RadioButtons, Button
import matplotlib.pyplot as plt
from numpy import mean, zeros, matrix, sqrt, array, linspace, power
from TmaxSelector import TmaxSelector
import pickle
from mpCurveFit import mpCurveFit
import pyfits as pf

from lightCurve import lightCurve

class pulseFit:



    def __init__(self):


        self.data = 0
        self.errors = 0
        self.tBins = 0

        self.fig = plt.figure(1) 
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.3)
        self.numPulse = 1
        self.flcFlag = False
        self.timeOffset = 0.0
        self.initialValues=[0,1,1,1]
        self.fixPar = [1,0,0,1,1]

        self.pulseLookup=[f1,f2,f3]

      

    def ReadTTE(self,tteFile,eMin,eMax,tMin,tMax,dt):

        lc = lightCurve('') # Give it a fake parfile
        lc.inFiles = [tteFile]
        lc.ImportData()
        lc.dt=dt
        lc.tMax=tMax
        lc.tMin=tMin
        lc.eBins = [[eMin,eMax]]
        lc.bkgFlag=True
        lc.EnergyBinning()
        lc.TimeBinning()

        self.data = lc.lcBkgSub[0]
        self.errors = array(map(sqrt,lc.lcBkgSub[0]))
        self.tBins = array(lc.tBins)

        self.PlotData()



    def LoadFlux(self,fileName):

        flux  = pickle.load(open(fileName))
        
        self.fluxes = flux['fluxes']
        self.errors = array(flux['errors'])
        self.tBins = flux['tBins']
        
        axcolor = 'lightgoldenrodyellow'


 #       self.radioFig = plt.figure(2)
#
        ax = plt.axes([.01, 0.7, 0.2, 0.32], axisbg=axcolor)

        self.data = self.fluxes['total']

        self.radio = RadioButtons(ax,tuple(self.fluxes.keys()))
        self.radio.on_clicked(self.Selector)
        
        self.PlotData()
        self.FindMax() # Must happen after data is plotted!


    def SetData(self, flux, errors, tBins):
        
        self.data = flux
        self.errors = errors
        self.tBins = tBins


  


    def FindMax(self):

        self.fmax = self.data.max()

        for i in range(len(self.data)):
            if self.data[i]==self.fmax:
                self.tmax=[mean(self.tBins[i])]
                break

        self.initialValues[3]=self.fmax
        self.tMaxSelector.points  =self.tmax
        print self.tmax
        print self.fmax


    def ReadFluxLC(self,fluxLC):


        self.fluxes =  fluxLC.fluxes

        self.errors =  array(fluxLC.fluxErrors)
        self.tBins = fluxLC.tBins

        self.models = fluxLC.modelNames


        self.flcFlag =True


        axcolor = 'lightgoldenrodyellow'


 #       self.radioFig = plt.figure(2)
#
        ax = plt.axes([.01, 0.7, 0.2, 0.32], axisbg=axcolor)

        self.data = self.fluxes['total']

        self.radio = RadioButtons(ax,tuple(self.fluxes.keys()))
        self.radio.on_clicked(self.Selector)
        
        self.PlotData()
        self.FindMax()
        

###### Plotting

    def Selector(self,label):

        self.ax.cla()
        self.data = self.fluxes[label]
        self.tMaxSelector.Kill()
        
        del self.tMaxSelector
        self.PlotData()

        

    def AddPulse(self,event=0):

        if self.numPulse>=3:
            self.numPulse=1
            self.tMaxSelector.SetNumPoints(self.numPulse)

        else:
            

            self.numPulse+=1
            self.tMaxSelector.SetNumPoints(self.numPulse)


   


    def PlotData(self):

    
       # plt.axes(self.fig.get_axes()[0])
        
        lowerT=[]
        upperT=[]

        for x,y in zip (self.tBins, map(mean,self.tBins)  ):

            lowerT.append(abs(x[0]-y))
            upperT.append(abs(x[1]-y))


        

        pl,er,bar, = self.ax.errorbar(array(map(mean,self.tBins))+self.timeOffset,self.data,fmt='o', color='b', yerr=array(self.errors), xerr=[lowerT,upperT])
        self.pl = pl

        self.tMaxSelector = TmaxSelector(pl)
        self.tMaxSelector.SetNumPoints(self.numPulse)

        ax = plt.axes([.05, 0.05, 0.2, 0.32])
        self.addButton = Button(ax,'Add Pulse')

        self.addButton.on_clicked(self.AddPulse)

   #     ax = plt.axes([.06, 0.05, 0.2, 0.32])
   #     self.fitButton = Button(ax,'Fit')

    #    self.fitButton.on_clicked(self.FitPulse)



        self.fig.canvas.draw()
      #  pl.xlabel("T")
      #  pl.ylabel("Flux")
        
    def ResetPlot(self):

      #   self.pl.remove()
        del self.pl


    def DisplayFitPlot(self):
            
        m = linspace(0,self.tBins.flatten().max(),1000)+self.timeOffset

        tmpFitResults = array(self.fitResults)
        tmpFitResults[3]+=self.timeOffset

        def f(t):
            
            tmp=[t]
            #print tmp
            tmp.extend(tmpFitResults.tolist())
         
            return apply(self.pulseLookup[self.numPulse-1],tmp)


        n = array(map(f,m))

   
        lowerT=[]
        upperT=[]

        for x,y in zip (self.tBins, map(mean,self.tBins)  ):

            lowerT.append(abs(x[0]-y))
            upperT.append(abs(x[1]-y))

        lowerT =array(lowerT)+self.timeOffset
        upperT = array(upperT)+self.timeOffset
        self.resultFig = plt.figure(2)


        resultAx = self.resultFig.add_subplot(111)


        resultAx.errorbar(array(map(mean,self.tBins))+self.timeOffset,self.data,fmt='o', color='b',yerr=array(self.errors), xerr=[lowerT,upperT])
        resultAx.plot(m,n,'r')
        self.resultFig.canvas.draw()




    def DisplayTestPlot(self,c,r,d,tmax,fmax):


            
        m = linspace(0,self.tBins.flatten().max(),1000)+self.timeOffset

        tmpFitResults = array([c,r,d,tmax,fmax])
        tmpFitResults[3]+=self.timeOffset

        def f(t):
            
            tmp=[t]
            #print tmp
            tmp.extend(tmpFitResults.tolist())
         
            return apply(self.pulseLookup[self.numPulse-1],tmp)


        n = array(map(f,m))

   
        lowerT=[]
        upperT=[]

        for x,y in zip (self.tBins, map(mean,self.tBins)  ):

            lowerT.append(abs(x[0]-y))
            upperT.append(abs(x[1]-y))

        lowerT =array(lowerT)+self.timeOffset
        upperT = array(upperT)+self.timeOffset
        self.testFig = plt.figure(2)


        resultAx = self.testFig.add_subplot(111)

        resultAx.loglog(m,n,'r')
        resultAx.errorbar(array(map(mean,self.tBins))+self.timeOffset,self.data,fmt='o', color='b',yerr=array(self.errors), xerr=[lowerT,upperT])
        
        self.testFig.canvas.draw()





###### Pulse Fitting


    def FitPulse(self,event=0):

       func = self.pulseLookup[self.numPulse-1]
       
       initialValues=[]
       print 

       self.tmax=self.tMaxSelector.GetData()


       print "Initial guess(es) for Tmax"
       for x in self.tmax:
           print x
       print  "\n___________\n"



       for x in self.tmax:
           #initialValues.extend([.1,-1,-.5,x,1])
           initialValues.extend([self.initialValues[0],self.initialValues[1],self.initialValues[2],x,self.initialValues[3]])

        #popt, pcov = mpCurveFit(func, array(map(mean,self.tBins))+self.timeOffset, self.data.tolist(), sigma=self.errors,p0=initialValues)
 
       limits =[ [[1,0],[0,0]], [[1,0],[0,0]], [[1,0],[0,0]],[[1,0],[0,0]],[[1,0],[0,0]] ]

       fit = mpCurveFit(func, array(map(mean,self.tBins))+self.timeOffset, self.data.tolist(), sigma=self.errors,p0=initialValues,fixed=self.fixPar,limits=limits,maxiter=400) 

       

   

       self.fitResults = fit.params 
       self.fitCov = fit.errors
       self.fitResults[3]-=self.timeOffset
       self.GoodnessOfFit()


    def GoodnessOfFit(self):


        def f(t):
            
            tmp=[t]
            #print tmp
            tmp.extend(self.fitResults.tolist())
         
            return apply(self.pulseLookup[self.numPulse-1],tmp)
        

        n = array(map(f, map(mean,self.tBins)  ))


        tmp = (power((self.data - n),2))/power(self.errors,2)

        chi2 =  tmp.sum()

        dof = len(self.data) - 5*self.numPulse - 1

        print "\nReduced Chi2: "+str(chi2/dof)






    def DisplayFitResults(self):


        fitParams = ['c: ','r: ','d: ','tmax: ', 'fmax: ']

        for i in range(self.numPulse-1):
            fitParams.extend(fitParams)


        fitParams = tuple(fitParams)
        errors = self.fitCov
       # errors = map(sqrt, matrix(self.fitCov).diagonal().tolist()[0] )

        print '\n\n*****************************'
        print 'Fit Results:\n'

        for x,y,z in zip(fitParams,self.fitResults,errors):

            print x+str(y)+' +/- '+str(z)



        print '\n\n******************************\n'


    def SaveFit(self):
        '''
        Save the fit results to a dictionary in the form of dic['<param>'][pulseNumber][val,err]

        '''

        fitParams = ['c','r','d','tmax', 'fmax']

#        errors = map(sqrt, matrix(self.fitCov).diagonal().tolist()[0] )
        errors=self.fitCov
        saveList = []

        for i in range( len(fitParams) ):
            
            tmpRow=[]

            for j in range(self.numPulse):
                
                tmp = [self.fitResults[i+j*len(fitParams)],errors[i+j*len(fitParams)]]
                tmpRow.append(tmp)
            saveList.append(tmpRow)


        saveDic = dict (zip(fitParams ,saveList) )
        
        pickle.dump(saveDic,open('fitSave.p','w'))



        

     #   f=open('pulsefitresults.txt','w')
     
#   for x,y in zip(self.fitResults,errors):
 #           f.write(str(x)+'\t'+str(y))
        
  #      print "\nWrote \'pulsefitresults.txt\'\n\n"


   







 ####################################################       

def KRLPulse(t,c,r,d,tmax,fmax):

    f = (fmax*(((((t+c)/(tmax+c)))**r)/(((d+(r*((((t+c)/(tmax+c)))**(1+r))))/(d+r))**((d+r)/(1+r)))))
    return f



def f1(t,c,r,d,tmax,fmax):
    return KRLPulse(t,c,r,d,tmax,fmax)

def f2(t,c1,r1,d1,tmax1,fmax1,c2,r2,d2,tmax2,fmax2):
    return KRLPulse(t,c1,r1,d1,tmax1,fmax1)+KRLPulse(t,c2,r2,d2,tmax2,fmax2)

def f3(t,c1,r1,d1,tmax1,fmax1,c2,r2,d2,tmax2,fmax2,c3,r3,d3,tmax3,fmax3):
    return KRLPulse(t,c1,r1,d1,tmax1,fmax1)+KRLPulse(t,c2,r2,d2,tmax2,fmax2)+KRLPulse(t,c3,r3,d3,tmax3,fmax3)
    
    



    





        
        