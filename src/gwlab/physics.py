from .constants import G, C, M_SUN

def get_chirp_mass(m1, m2):
    m1_kg = m1 * M_SUN
    m2_kg = m2 * M_SUN
    return ((m1_kg * m2_kg) ** (3/5)) / ((m1_kg + m2_kg) ** (1/5))