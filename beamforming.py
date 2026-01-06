def select_beamforming(path_loss):
    if path_loss < 60:
        return "Narrow Beam (High Gain)"
    elif path_loss < 90:
        return "Medium Beam"
    else:
        return "Wide Beam (Robust Mode)"
