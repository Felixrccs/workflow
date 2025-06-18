from subprocess import run
import re



def call_Ba_NSGA(mace):
    """find convex hull of set of points

    Parameters
    ----------
    MACE: str
        path to mace model
    
    Returns
    -------
        None
    """
    with open('/u/friccius/scripts/python/BA_NSGA_AL.py', 'r') as f:
        text_string = f.read()

    text_string = re.sub(r'calc_sign', mace, text_string)

    exec(text_string)
