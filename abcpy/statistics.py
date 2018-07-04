from abc import ABCMeta, abstractmethod
import numpy as np

class Statistics(metaclass = ABCMeta):
    """This abstract base class defines how to calculate statistics from dataset.

    The base class also implements a polynomial expansion with cross-product
    terms that can be used to get desired polynomial expansion of the calculated statistics.
    
            
    """
    
    @abstractmethod
    def __init__(self, degree = 2, cross = True):
        """Constructor that must be overwritten by the sub-class.

        The constructor of a sub-class must accept arguments for the polynomial 
        expansion after extraction of the summary statistics, one has to define 
        the degree of polynomial expansion and cross, indicating whether cross-prodcut
        terms are included. 

        Parameters
        ----------
        degree: integer, optional
            Of polynomial expansion. The default value is 2 meaning second order polynomial expansion.
        cross: boolean, optional
           Defines whether to include the cross-product terms. The default value is TRUE, meaning the cross product term is included.
        """       

        raise NotImplementedError
        
        
    @abstractmethod
    def statistics(self, data: object) -> object:
        """To be overwritten by any sub-class: should extract statistics from the 
        data set data. It is assumed that data is a  list of n same type 
        elements(eg., The data can be a list containing n timeseries, n graphs or n np.ndarray).
        
        Parameters
        ----------
        data: python list
            Contains n data sets.
        Returns
        -------
        numpy.ndarray
            nxp matrix where for each of the n data points p statistics are calculated.
            
        """
        
        raise NotImplementedError

    def _polynomial_expansion(self, summary_statistics):
        """Helper function that does the polynomial expansion and includes cross-product
        terms of summary_statistics, already calculated summary statistics.

        Parameters
        ----------
        summary_statistics: numpy.ndarray
            nxp matrix where n is number of data points in the datasets data set and p number os 
            summary statistics calculated.
        Returns
        -------
        numpy.ndarray
            nx(p+degree*p+cross*nchoosek(p,2)) matrix where for each of the n pointss with 
            p statistics, degree*p polynomial expansion term and cross*nchoosek(p,2) many
            cross-product terms are calculated.      
               
        """
        
        # Check summary_statistics is a np.ndarry
        if not isinstance(summary_statistics, (np.ndarray)):
            raise TypeError('Summary statisticss is not of allowed types')
        # Include the polynomial expansion
        result = summary_statistics
        for ind in range(2,self.degree+1):
            result = np.column_stack((result,np.power(summary_statistics,ind)))

        # Include the cross-product term
        if self.cross == True and summary_statistics.shape[1]>1:          
            # Convert to a matrix
            for ind1 in range(0,summary_statistics.shape[1]):
                for ind2 in range(ind1+1,summary_statistics.shape[1]):
                    result = np.column_stack((result,summary_statistics[:,ind1]*summary_statistics[:,ind2]))
        return result



class Identity(Statistics):
    """
    This class implements identity statistics returning a nxp matrix when the data set 
    contains n numpy.ndarray of length p.
    """
    def __init__(self, degree = 2, cross = True):
        self.degree = degree
        self.cross = cross

    def statistics(self, data):
        if isinstance(data, list):
            if np.array(data).shape == (len(data),):
                if len(data) == 1:                
                    data = np.array(data).reshape(1,1)
                data = np.array(data).reshape(len(data),1)
            else:
                data = np.concatenate(data).reshape(len(data),-1)
        else:
            raise TypeError('Input data should be of type list, but found type {}'.format(type(data)))
        # Expand the data with polynomial expansion            
        result = self._polynomial_expansion(data)  
        
        return result    
