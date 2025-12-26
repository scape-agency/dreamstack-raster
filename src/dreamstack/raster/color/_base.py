
class ColorBase(object):
    """
    A base class holding some common methods and values.
    """

    # Attribute names containing color data on the sub-class. For example,
    # sRGBColor would be ['rgb_r', 'rgb_g', 'rgb_b']
    VALUES = []
    # If this object as converted such that its values passed through an
    # RGB colorspace, this is set to the class for said RGB color space.
    # Allows reversing conversions automatically and accurately.
    _through_rgb_type = None

    def get_value_tuple(self):
        """
        Returns a tuple of the color's values (in order). For example,
        an LabColor object will return (lab_l, lab_a, lab_b), where each
        member of the tuple is the float value for said variable.
        """
        retval = tuple()
        for val in self.VALUES:
            retval += (getattr(self, val),)
        return retval

    def __str__(self) -> str:
        """
        String representation of the color.
        """
        retval = self.__class__.__name__ + " ("
        for val in self.VALUES:
            value = getattr(self, val, None)
            if value is not None:
                retval += "%s:%.4f " % (val, getattr(self, val))
        if hasattr(self, "observer"):
            retval += "observer:" + self.observer
        if hasattr(self, "illuminant"):
            retval += " illuminant:" + self.illuminant
        return retval.strip() + ")"

    def __repr__(self) -> str:
        """
        Evaluable string representation of the object.
        """
        retval = self.__class__.__name__ + "("
        attributes = [(attr, getattr(self, attr)) for attr in self.VALUES]
        values = [x + "=" + repr(y) for x, y in attributes]
        retval += ", ".join(values)
        if hasattr(self, "observer"):
            retval += ", observer='" + self.observer + "'"
        if hasattr(self, "illuminant"):
            retval += ", illuminant='" + self.illuminant + "'"
        return retval + ")"

