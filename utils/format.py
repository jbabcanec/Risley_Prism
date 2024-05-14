def round_dict_values(d, decimals):
    if isinstance(d, dict):
        return {k: round_dict_values(v, decimals) for k, v in d.items()}
    elif isinstance(d, list):
        return [round_dict_values(v, decimals) for v in d]
    elif isinstance(d, float):
        return round(d, decimals)
    elif isinstance(d, int):
        return d
    else:
        return d

def format_dict(d):
    return "\n".join(f"{k}: {v}" for k, v in d.items())