from astropy.table import Table
from numpy import mean, dtype, float64, array, concatenate, asarray
from copy import deepcopy
import gc
from spectralTools.models import modelLookup
keV2erg =1.60217646e-9
import numpy as np

class specReader:
    '''
    SCATReadrer reads in a filename of an scat file made by RMFIT
    and turns it into a python object. It stores the covariance matrix,
    fit params, fluxes from RMFIT, and time bins for EACH model.

    It can read single fits or batch fits.
    
    It interfaces with other progams for plotting and flux calculation

    The add operator is overloaded so that two SCATReaders can be added.
    At the moment the temporal ordering is based on you adding the files
    in the proper order
    

    '''
    def __init__(self, fileName, silent=True):
        

        self.effArea = True
        if fileName == "summed":
            return
        
        self.modelNames = []
        self.models = []

        
        self.silent=silent

        self.scat  = np.load(fileName)
        if not self.silent:
            print "Opening SCAT file: "+fileName

        self.tBins  = self.scat['tBins']
        self.meanTbins = array(map(mean,self.tBins))
        #self.met=

        self.phtFlux = self.scat['phtflux']
        #self.phtFluence = self.scat[2].data['PHTFLNC']
        self.covars = self.scat['covars']
        
        self.dof = self.scat['dof']
        self.cstat = self.scat['cstat']
        

        # I may take this out at some point
        self.batchFit = True
        self.ExtractModels()
        self.FormatCovarMat()
        if not self.silent:
            print "Closing SCAT file\n"
        #self.scat.close()
        del self.scat
        gc.collect()





    def GetParamArray(self, model,param):
        '''
        Returns a paramerter array for that model. This is NOT a
        structured list. The params are the first column and the 
        errors are the last two columns.

        '''
        paramArr = deepcopy(self.models[model]['values'][param])
        paramErrplus = deepcopy(self.models[model]['+'][param])
        paramErrminus = deepcopy(self.models[model]['-'][param])
        tmp = Table(asarray([paramArr,paramErrminus,paramErrplus]).transpose(),names=[param,"+","-"])
        #tmp.dtype = dtype([(float,'value'),(float,'error')])
        return tmp
        

    def GetTime(self):


        tmp = Table(array(zip(self.meanTbins,self.tBins[:,0],self.tBins[:,1])),names=["t","tstart","tstop"] )
        return tmp

    def __repr__(self):

        info = "SCAT Models:\n"
        for x in self.modelNames:
            info = info+x+"\n"
        info = info+"\n\n"

        info = info+"Time Bins:\n"
        for x in self.tBins:
            info = info + str(x[0])+' : '+str(x[1])+'\n'

        info = info+"\n\n"

        




        return info


    def ExtractModels(self):


        self.modelNames = self.scat['modelnames']
    
    

        # Now extract the parameters from the models

        self.paramNames =self.scat['parNames']
        

        self.numEffCor = len(self.scat['eac'][0])
        #self.numParams = np
        self.numParams=self.scat['numbers'][0]

        

        dicString = ['values','-','+']
        
        tmp =[]
        
        
        for p,v,e in zip(self.paramNames,self.scat['params'],self.scat['errs']):
            tmp.append([Table(v,names=p),Table(e,names=p),Table(e,names=p)])
            
        
        tmp = map(lambda x: dict(zip(dicString,x))   , tmp)

        self.models = dict(zip(self.modelNames,tmp))

        self.numModels = len(self.modelNames)

        

  
    def FormatCovarMat(self):


        length = self.numParams
        
        covars = []
        
        for x in self.covars:
            
            covar = []
            
 
            for i in range(length):
                

                tmp = []

                for j in range(length):
                   

                    #tmp.append(x[i*length+j])
                    tmp.append(x[i][j])

                covar.append(tmp)
                    
            covars.append(array(covar))
        #if self.effArea and not self.batchFit:
        #    print "Correcting COVAR matrix"
        #    print self.numEffCor
            
        #    for i in range(len(covars)):
        #        print "test"
                #covars[i] = covars[i][:self.numEffCor,:self.numEffCor]
                #self.numParams = self.numParams - self.numEffCor
        self.covars=covars





    def _TimeIndex(self,t):

        indx = self.meanTbins.searchsorted(t)
        test=[abs(self.meanTbins[indx]-t),abs(self.meanTbins[indx-1]-t)]


        val2 = min(test)



        indx = indx-test.index(val2)
        return indx



    def _CalcModel(self,energy,params,modelName="Band's GRB, Epeak"):
        '''
        Returns the photon flux of the model at a given energy

        '''

        model = modelLookup[modelName]

        return model(energy,*params)

    
           
    def SpecificEnergyFlux(self,time, energy,modelName="Band's GRB, Epeak"):



        tIndx = self._TimeIndex(time)


        params = self.models[modelName]['values'][tIndx][0]

        
        
        phtFlux = self._CalcModel(energy,params,modelName)

        return energy*phtFlux*keV2erg
        


class CreateFitFiles(object):
    '''
    This class reads David Yu's RMFIT output
    files and turns them into numpy array
    files so that they can be read with a form
    of scatreader.
    '''
    
    
    def SetCovarFile(self,covar):
        '''
        Set the file that contains the covariance matrix of the fits
        
        :param covar: The txt file containing the covarariance matrix
        
        '''
        
        self.covarFile = covar
    

    def SetParamsFile(self,param):
        '''
        Set the file that contains the covariance matrix of the fits
        
        :param param: The txt file containing the parameters
        
        '''
        
        self.paramFile = param
        
        
    def SetStatFile(self,stat):
        '''
        Set the file that contains the covariance matrix of the fits
        
        :param stat: The txt file containing the stats
        
        '''
        self.statFile = stat
        
    
    def SetEACFile(self,eac):
        '''
        Set the file that contains the covariance matrix of the fits
        
        :param eac: The txt file containing the covarariance matrix
        
        '''
        self.eacFile = eac
        
        
    def _ReadCovar(self):
        '''
        Formats the covariance matric
        
        '''
        covars = np.genfromtxt(self.covarFile)
        
        
        covars = np.array(map(lambda x: x[4:].reshape((self.nParams+self.numEAC,self.nParams+self.numEAC)),covars))
        tmp = []


        tmpP = np.zeros((len(covars),self.nParams+self.numEAC))
        for i in range(len(self.models)):
            for j in range(len(self.paramErrors[i])):
                for k in range(len(self.paramErrors[i][j])):
                    tmpP[j,i+k+i*1] = self.paramErrors[i][j][k]
        
        
        for i in range(len(covars)):
            for j in range(self.numEAC):
                tmpP[j,self.nParams+j] = self.eacErr[i,j]
        
            
            

        self.covars = []
        for cv,p in zip(covars,tmpP):
            cv=np.matrix(cv)

            cv = array(np.diag(p) * cv * np.diag(p))
            self.covars.append(cv)
        
        
    def _ReadParams(self):
        '''
        Formats the params
        
        '''
        params = np.genfromtxt(self.paramFile,names=True)
        mCount = 0
        ps=[]
        es=[]
        for m in self.models:
            tmpP = []
            tmpE = []
            for p in self.params[mCount]:
                tmpP.append(params[p])
                tmpE.append(params[p+'err'])
            ps.append(array(tmpP).T.tolist())
            es.append(array(tmpE).T.tolist())
            mCount+=1
                
        self.paramValues= np.array(ps)
        self.paramErrors = np.array(es)
        
        self.cstat = params['cstat']
        self.dof   = params['dof']
        self.tBins = np.array(zip(params['tstart'],params['tstop']))
        self.phtflux=params['phflux']
        self.phtfluxErr=params['phfluxerr']
    
    def _ReadEAC(self):
        '''
        Formats the EACs
        
        '''
        
        eacs = np.genfromtxt(self.eacFile)
        eac=[]
        eacErr=[]
        for x in eacs:
            eac.append(x[4::2].tolist())
            eacErr.append(x[4::2].tolist())
        self.eac=np.array(eac)
        self.eacErr=np.array(eacErr) 
        
        self.numEAC = len(self.eac[0])
        
        
    def Process(self):
        '''
        Run all the readers and save the output
        '''
        
        self._ReadEAC()
        self._ReadParams()
        self._ReadCovar()
        
        
        np.savez_compressed(self.filename,
                            tBins=self.tBins,
                            covars=self.covars,
                            eac=self.eac,
                            eacErr=self.eacErr,
                            params=self.paramValues,
                            errs=self.paramErrors,
                            cstat=self.cstat,
                            dof=self.dof,
                            phtflux=self.phtflux,
                            phtfluxErr=self.phtfluxErr,
                            modelnames=self.models,
                            parNames=self.rmfitParams,
                            numbers=array([self.nParams])
                           )
        



class BandReader(CreateFitFiles):
    
    def __init__(self,filename):
        
        self.params = [['norm','epeak','alpha','beta']]
        self.rmfitParams=[["Amplitude","Epeak","alpha","beta"]]
        self.nParams = 4
        self.models=array(["Band's GRB, Epeak"])
        self.filename =filename
        
        
class CompReader(CreateFitFiles):
    
    def __init__(self,filename):
        
        self.params = [['norm','epeak','alpha']]
        self.rmfitParams=[["Amplitude","Epeak","Index",]]
        self.nParams = 3
        self.models=array(["Comptonized, Epeak np"])
        self.filename =filename
class PLReader(CreateFitFiles):
    
    def __init__(self,filename):
        
        self.params = [['norm','alpha']]
        self.rmfitParams=[["Amplitude","Index"]]
        self.nParams = 2
        self.models=array(["Power Law np"])
        self.filename =filename
class PLBBReader(CreateFitFiles):
    
    def __init__(self,filename):
        
        self.params = [['norm','alpha'],["ktnorm","kt"]]
        self.rmfitParams=[["Amplitude","Index"],["Amplitude","kT"]]
        self.nParams = 4
        self.models=array(["Power Law np","Black Body"])
        self.filename =filename
        
class BandBBReader(CreateFitFiles):
    
    def __init__(self,filename):
        
        self.params = [['norm','epeak','alpha','beta'],["ktnorm","kt"]]
        self.rmfitParams=[["Amplitude","Epeak","alpha","beta"],["Amplitude","kT"]]
        self.nParams = 6
        self.models=array(["Band's GRB, Epeak","Black Body"])
        self.filename =filename
