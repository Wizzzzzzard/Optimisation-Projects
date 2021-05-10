import numpy as np
import math
import time
import copy
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Assumes CareDistances-FULL and CareDurations are in same directory as .py file

CareDist_Matrix = np.loadtxt(open("CareDistances-FULL.csv", "rb"), dtype=str, delimiter=",", skiprows=1)
CareDist_Matrix[0,0]='0'
CareDist_Matrix = CareDist_Matrix.astype(np.int)

CareDist_Matrix[CareDist_Matrix == 0] = 9999999 # Replace all 0 distance values with a large 
                                                # number to prevent routing there due to the 0 distance

CareDist_Matrix[0,0] = 0

Care_Durations = np.loadtxt(open("CareDurations.txt", "rb"), dtype=int, delimiter="\t", skiprows=1)
Care_Durations[:, 1] = Care_Durations[:, 1] * 60

# Inserts duration value for the starting points and the dummy node 0
Care_Durations = np.insert(Care_Durations, 0, np.array((0, 0)), 0)
Care_Durations = np.insert(Care_Durations, 3, np.array((71, 0)), 0)
Care_Durations = np.insert(Care_Durations, 7, np.array((142, 0)), 0)
Care_Durations = np.insert(Care_Durations, 14, np.array((280, 0)), 0)
Care_Durations = np.insert(Care_Durations, 104, np.array((3451, 0)), 0)
Care_Durations = np.insert(Care_Durations, 185, np.array((6846, 0)), 0)
Care_Durations = np.insert(Care_Durations, 220, np.array((7649, 0)), 0)

# This allows the same index to be used for the distance matrix and the durations table

# The centres correspond to the following nodes:

# 71 is Node 3
# 142 is Node 7
# 280 is Node 14
# 3451 is Node 104
# 6846 is Node 185
# 7649 is Node 220

def get_depo_list():
    """Return the starting points of routes
    """
    return [3,7,14,104,185,220] # starting points of routes

A = CareDist_Matrix
depots = get_depo_list()
start = random.choice(depots)
wait = Care_Durations


routes = []
routes_cost = []
path = [start]
cost = 0
N = A.shape[0]
mask = np.ones(N, dtype=bool)  # boolean values indicating which 
                               # locations have not been served
mask[start] = False
staff_count = 1
max_cost_per_staff = 7*60*60   # 7 hours in seconds

def get_unserved_clients():
    """Return a list of clients who are currently marked as unserved, an empty list indicated all served
    """
    unserved = np.where(mask == True)[0]
    return unserved

# Random algorithm

while True in mask:
  for i in range(N-1):
    last = path[-1]
    lastw = path[0]
    next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
    next_loc = np.arange(N)[mask][next_ind] # convert to original location
    path.append(next_loc)
    mask[next_loc] = False
    cost += sum(A[last, next_loc] + wait[lastw]) # adds distance between nodes and
                                                 # wait time for each node to cost
    get_unserved_clients()
    if cost >= max_cost_per_staff:
      staff_count += 1
      routes_cost.append(cost)
      routes.append(path)
      cost = 0
      if get_unserved_clients().size == 0:
        print("routes are complete")
      else:
        start = random.choice(depots)
        path = [start]

# Converts index values used for Nodes back to actual Node IDs

nodes = np.delete(CareDist_Matrix[0], 0)

d = dict(enumerate(nodes.flatten(), 1))
d[0]= 0
routes = [[d[(k)] for k in lst] for lst in routes]

# Writes the routes and number of staff required to route.txt

for i in range(len(routes)):
  print("route", i, routes[i], routes_cost[i], file=open("random algorithm route.txt", "a"))
print("\nstaff required:", staff_count, file=open("route.txt", "a"))

def distance_between(source, target):
    """Returns the distance between any two target nodes or -1 if no distance defined"""
    distance = A[int(source),int(target)]
    return distance

def find_closest_depo(source):
    """Returns the closest depo to this target node,
    Includes the depo and travel cost from the source
    Returns None if no valid journey is found
    """
    depos = get_depo_list()
    travelCost = sys.maxsize - 1
    closest_depo = None
    for d in depos:
        journey = distance_between(source, d)
        if journey < 0:
            print("Route between {} and {} is undefined".format(source, d))
        elif journey < travelCost:
            closest_depo = int(d)
            travelCost = journey

    if closest_depo is None:
        return random.choice(depos)
    else:
        return {
            closest_depo
        }

# Minimum distance depot

A = CareDist_Matrix
start = 7
wait = Care_Durations

depots = [3,7,14,104,185,220] # starting points of routes
routes = []
routes_cost = []
path = [start]
cost = 0
N = A.shape[0]
mask = np.ones(N, dtype=bool)  # boolean values indicating which 
                               # locations have not been served
mask[start] = False
staff_count = 1
max_cost_per_staff = 7*60*60   # 7 hours in seconds

while True in mask:
  for i in range(N-1):
    last = path[-1]
    lastw = path[0]
    try:
      next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
    except:
      next_ind = 7 # when a non-integer value is passed depot 7 is selected
    next_loc = np.arange(N)[mask][next_ind] # convert to original location
    path.append(next_loc)
    mask[next_loc] = False
    try:
      cost += sum(A[last, next_loc] + wait[lastw]) # adds distance between nodes and
                                                   # wait time for each node to cost
    except:
      cost = cost # when error just maintains current cost value
    get_unserved_clients()
    if cost >= max_cost_per_staff:
      staff_count += 1
      routes_cost.append(cost)
      routes.append(path)
      cost = 0
      if get_unserved_clients().size == 0:
        print("routes are complete")
        print(routes)
      else:
        depos = get_depo_list()
        travelCost = sys.maxsize - 1
        closest_depo = None
        for d in depos:
          journey = distance_between(next_ind, d)
          if journey < 0:
            print("Route between {} and {} is undefined".format(source, d))
          elif journey < travelCost:
            closest_depo = int(d)
            travelCost = journey
          if closest_depo is None:
            closest_depo = random.choice(depos)
          else:
            start = int(closest_depo)
            path = [int(start)]

nodes = np.delete(CareDist_Matrix[0], 0)

d = dict(enumerate(nodes.flatten(), 1))
d[0]= 0

routes = [[d[(k)] for k in lst] for lst in routes]
print(routes)

for i in range(len(routes)):
  print("route", i, routes[i], routes_cost[i], file=open("minimum distance route.txt", "a"))
print("\nstaff required:", staff_count, file=open("route.txt", "a"))

for i in range(len(routes)):
  print("route", i, routes[i], routes_cost[i], file=open("user-readable route.txt", "a"))
print("\nstaff required:", staff_count, file=open("route.txt", "a"))

print("my_list = ", file=open("machine_readable_route.py", "a"))

for i in range(len(routes)): 
  print (routes[i], file=open("machine_readable_route.py", "a"))