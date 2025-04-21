def gpa_to_grade_class(gpa):
    """
    Convert GPA to GradeClass based on the following scale:
        0: 'A' (GPA >= 3.5)
        1: 'B' (3.0 <= GPA < 3.5)
        2: 'C' (2.5 <= GPA < 3.0)
        3: 'D' (2.0 <= GPA < 2.5)
        4: 'F' (GPA < 2.0)

    Parameters:
    -----------
    gpa : float or pd.Series
        A single GPA value or a pandas Series of GPA values.

    Returns:
    --------
    int or pd.Series:
        Corresponding GradeClass value(s).
    """
    if hasattr(gpa, 'apply'):  # If gpa is a pandas Series
        return gpa.apply(gpa_to_grade_class)
    if gpa >= 3.5:
        return 0
    elif gpa >= 3.0:
        return 1
    elif gpa >= 2.5:
        return 2
    elif gpa >= 2.0:
        return 3
    else:
        return 4
