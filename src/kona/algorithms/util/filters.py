import sys

class Filter(object):
    """
    Simple filter used for globalizing FLECS step solutions in RSNK algorithms.

    Attributes
    ----------
    point_list : list of tuple of float
        A list containing (objective, constraint) point pairs.
    """
    def __init__(self):
        self.point_list = []

    def dominates(self, obj, constr):
        """
        Determines whether the given objective and constraint pair is dominated
        by the filter.

        If the pair is not dominated, then it is acceptable and is used to
        modify the filter list.

        Parameters
        ----------
        obj : float
            Objective value.
        constr : float
            Constraint value.

        Returns
        -------
        boolean : True if the pair is dominated by the filter. False otherwise.
        """
        # create a new point pair
        new_point = (obj, constr)

        # check if new point is dominated by points in the filter
        for point in self.point_list:
            if new_point > point:
                return True

        # if we got here, new point is acceptable to the filter
        # remove old points that are dominated by the new one
        for point in self.point_list:
            if point >= new_point:
                del point

        # add the new point to the filter
        self.point_list.append(new_point)

        return False

    def info(self, fout=sys.stdout):
        """
        Prints out the points that define the filter.

        Parameters
        ----------
        fout : file object (optional)
        """
        fout.write('Filter contents:\n')
        for index, point in enumerate(self.point_list):
            fout.write('    %i: (%f, %f)\n'%(index, point[0], point[1]))
