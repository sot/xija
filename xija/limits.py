def get_limit_spec(limit):
    """
    Based on an input *limit* string of the form:
    
    <system>.<type>.<direction>.<qualifier>    
    
    (where the "qualifier" element is optional) return the various pieces 
    into a dictionary for convenience.
    """
    words = limit.split(".")
    limit_spec = {
        "system": words[0],
        "type": words[1],
        "direction": words[2]
    }
    if len(words) == 4:
        limit_spec["qualifier"] = words[3]
    else:
        limit_spec["qualifier"] = None
    return limit_spec


def get_limit_color(limit):
    """
    Based on an input *limit* string of the form:
    
    <system>.<type>.<direction>.<qualifier>    
    
    (where the "qualifier" element is optional) return the color to
    be used for plotting.
    """
    limit_spec = get_limit_spec(limit)
    if limit_spec["qualifier"] is not None:
        if limit_spec["qualifier"].startswith("acis") or \
            limit_spec["qualifier"] == "cold_ecs":
            # ACIS FP limits have their own colors
            color = {
                "acisi": "blue",
                "aciss": "purple",
                "aciss_hot": "red",
                "cold_ecs": "dodgerblue"
            }[limit_spec["qualifier"]]
    elif limit_spec["system"] in ["odb", "safety"]:
        # ODB and safety limits are yellow/red
        color = {
            "caution": "gold",
            "warning": "red"
        }[limit_spec["type"]]
    elif limit_spec["system"] == "planning":
        # Planning limits are green, dodgerblue, or gray 
        color = {
            "caution": "dodgerblue",
            "warning": "green",
            "penalty": "gray"
        }[limit_spec["type"]]
    else:
        raise RuntimeError(f"limit \"{limit}\" is unknown!")
    return color
