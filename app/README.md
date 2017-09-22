# When sending a post request to /predict we're expecting either a form or a json request with an object with these fields:
#
# {
#   "algo": "lr", "ransac", "rfr", "sgd", or "svr"
#   "observations": {
#     "X": Int from 0-9,
#     "Y": Int from 0-9,
#     "month": "jan", "feb", "mar", "apr", "may", "jun", 
#              "jul", "aug", "sep", "oct", "nov", or "dec"
#     "day": "mon", "tue", "wed", "thu", "fri", "sat", or "sun"
#     "FFMC": Float,
#     "DMC": Float,
#     "DC": Float,
#     "ISI": Float,
#     "temp": Float,
#     "RH": Float,
#     "wind": Float,
#     "rain": Float
#   }
# }

# Returns
# {
#   "area": Float
# }
