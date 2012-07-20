from mpfit import mpfit
from mpfitexy import mpfitexy
from mpCurveFit import mpCurveFit
from functions import functionLookup
import inspect
from numpy import array, linspace, log10, log, dtype
import matplotlib.pyplot as plt

class funcFitter:

    def __init__(self, interactive = False):

        print "Start Up"
        self.interactive = interactive
        #self.twoD_flag = False
        self.funcTable =  functionLookup
        self.xName="x"
        self.yName="y"
        self.title="fit"
        self.yErr =None
        self.dataColor="b"
        self.fitColor="g"
        self.guessColor="r"
        self.dataMarker="o"
        self.fitLineStyle="-"
        self.errorbarThick=1.
        self.fitLineThick=2.
        self.plotNum = 1000

        if self.interactive:
            self.PrintFuncs()

    def SetPlotNum(self, plotNum):
        self.plotNum = plotNum

    def SelectFunction(self, funcName = ""):
        
        self.fitFunc = self.funcTable[funcName]
        self.funcName = funcName
        args, varargs, varkw, defaults = inspect.getargspec(self.fitFunc)
        self.params = args[1:]
        args = args[2:]
        if self.interactive:
            
            print "\n\nSet initial values:\n"
            vals = [1.]
            fixeds = [0]
            for x in args:
                val = float(raw_input("\t"+x +": "))
                fixed = int(raw_input("\tfixed (1/0): "))
                vals.append(val)
                fixeds.append(fixed)
            self.iVals = vals
            self.fixed = fixeds

    

    def PrintFuncs(self):

        print "Functions to fit:"
        for x in self.funcTable.keys():
            print "\t\""+x+"\""
        



    def SetXData(self, xData):
        self.xData = xData

    def SetYData(self, yData):
        self.yData = yData


    def SetYErr(self, yErr):
        
        self.yErr = yErr

    def SetXName(self,xName):
        self.xName = str(xName)

    def SetYName(self,yName):
        self.yName = str(yName)

    def SetTitle(self,title):
        self.title=str(title)
    
    def ConvertData2Log(self,data,err):

        logData = log10(data)
        logErr = err/(data*log(10))
        
        return [array(logData),array(logErr)]  

    def SetPlotColors(self,dataColor="b",fitColor="g",guessColor="r",dataMarker="o",fitLineStyle="-", fitLineThick=2., errorbarThick=1.):

        print errorbarThick
        print fitLineThick
        self.dataColor=dataColor
        self.fitColor=fitColor
        self.guessColor=guessColor
        self.dataMarker=dataMarker
        self.fitLineStyle=fitLineStyle
        self.fitLineThick=fitLineThick
        self.errorbarThick=errorbarThick




    def Fit(self,showLog=False,showGuess=False):
        
        print "Fitting with "+self.funcName 



        resultFig = plt.figure(self.plotNum)
        resultAx = resultFig.add_subplot(111)
            

        xRange = linspace(self.xData.min(),self.xData.max(),100)
        yGuess = self.fitFunc(xRange,*self.iVals)
          
            
        if showLog:
                
            resultAx.loglog(self.xData,self.yData,color=self.dataColor,marker=self.dataMarker,linestyle='.')
            if showGuess:
                    
                resultAx.loglog(xRange,yGuess,color=self.guessColor,linestyle=self.fitLineStyle,linewidth=self.fitLineThick)
        if showGuess:
            
            resultAx.plot(xRange,yGuess,color=self.guessColor,linestyle=self.fitLineStyle)
                

        resultAx.errorbar(self.xData,self.yData,linestyle='.',marker=self.dataMarker, color=self.dataColor,yerr=self.yErr,elinewidth=self.errorbarThick)
        
                

        fit = mpCurveFit(self.fitFunc, self.xData, self.yData, p0 = self.iVals, sigma = self.yErr, fixed = self.fixed,quiet=1)
        params, errors = [fit.params, fit.errors]
            

        print "\nFit results: "
        try:
            for x,y,z in zip(self.params, params, errors):
                print x+": "+str(y)+" +/- "+str(z)
        except TypeError:
            print "-----------> FIT FAILED!!!!!"
            return
        

        xRange = linspace(self.xData.min(),self.xData.max(),100)
        yResult = self.fitFunc(xRange,*params)
        self.result =  zip(params,errors)
        self.result.append([fit.chi2,fit.dof])
        self.result=array(self.result)
            

         

            

        if showLog:
            resultAx.loglog(xRange,yResult,color=self.fitColor,linestyle=self.fitLineStyle,linewidth=self.fitLineThick)
                
        else:
            resultAx.plot(xRange,yResult,color=self.fitColor,linestyle=self.fitLineStyle,linewidth=self.fitLineThick)
        #resultAx.errorbar(self.xData,self.yData,fmt=self.dataMarker, color=self.dataColor,yerr=self.yErr)
        resultAx.set_xlabel(self.xName)
        resultAx.set_ylabel(self.yName)
        resultAx.set_title(self.title)
           
            
            
            
