
class SimpleFilter(object):
    """
    Implementation of a very simple filter object used for globalizing
    constrained optimization problems.
    
    Attributes
    ----------
    points : list of tuples
    """
    def __init__(self):
        self.points = []
        
    def dominates(self, obj, cnstr_norm):
        """
        Checks if the points stored in the filter dominate the new point 
        (obj, cnstr_norm).
        
        If the filter dominates, then the new point is accepted, and any 
        existing points greater than the new point are removed from the filter.
        
        If the new point dominates, then it is rejected.
        
        Parameters
        ----------
        obj : float
        cnstr_norm : float

        Returns
        -------
        bool : True if point is accepted
        """
        new_point = (obj, cnstr_norm)
        
        # check if new point dominates any filter point
        dominated_by = []
        for i in xrange(len(self.points)):
            if new_point >= self.points[i]:
                # new point is not acceptable
                return False
            else:
                dominated_by.append(self.points[i])
                
        # if we got here, point is acceptable
        
        # remove dominating points
        for i in range(len(dominated_by)):
            self.points.remove(dominated_by[i])
            
        # add the new point at the front
        self.points.insert(0, new_point)
        
        return True
                
        