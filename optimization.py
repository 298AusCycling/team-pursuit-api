# %%
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import math 
from scipy.stats import truncnorm
import itertools

# %%
### Black box model that is a stand in for our actual function 
# comment out time.sleep(0.25) to make it instantaneous
# or, change to the number of seconds we want this to take...

# question: do we also need to include acceleration_length (number of half laps)?
counter = 0 
def black_box(schedule, peel, initial_order):
    #time.sleep(0.25) # fxn takes 0.25 seconds (plus time to generate random number...) 
    time_of_race = random.random()*4
    global counter 
    counter = counter + 1
    return time_of_race

# %%
### this is helpful in the genetic algorithm (allows for the function to be minimum of this, but not maximum) 

def sample_truncated_normal(center, min_, max_, std=1):
    a = (min_ - 1 - center) / std
    b = (max_ - 1 - center) / std
    return round(truncnorm.rvs(a, b, loc=center, scale=std))


# uncomment below to see how it works (the max would be considered too big): 
#list_stn = []
#for i in range(1000): 
#    list_stn.append(sample_truncated_normal(center=6, min_=4, max_=8))

#bins = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
#plt.hist(list_stn, bins=bins, edgecolor='black', align='mid')
#plt.xticks([3, 4, 5, 6, 7, 8, 9])
#plt.title('Histogram')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.show()

# %%
# this takes a list, and replaces the closest element to "peel" with "peel" 

def replace_with_peel(peel, relevant_list):
    closest_index = min(range(len(relevant_list)), key=lambda i: abs(relevant_list[i] - peel))
    relevant_list[closest_index] = peel
    #print("did it")
    #print(relevant_list)
    return relevant_list

# %%
# function that creates your list of 10 (11 if you count the parent) jittered options 

def create_jittered_kids(parent, acceleration_length, num_changes, num_children, peel, parent_list):
    parent_list.append(parent) #first append the parent to parent_list 
    warm_start = parent #for help with naming 
    #print("initial:", warm_start)
    all_children = []
    #all_children.append(warm_start) ### CHANGE HERE
    all_children.append(replace_with_peel(peel, warm_start[:]))
    #num_children=10000
    j = 1
    while j <= num_children: 
        #print("option number:", j)
        jittered_list = []
        for i in range(num_changes):
            if i == 0: 
                prev_jitter = 3
                while prev_jitter <= acceleration_length:
                    prev_jitter = sample_truncated_normal(center=warm_start[i], min_=warm_start[i]-(warm_start[i+1]-warm_start[i]), max_=warm_start[i+1])
                    #print("tried this: ", jitter)
                #print("first_jitter:", prev_jitter) 
                jittered_list.append(prev_jitter)
                ### CODE for what to do for the first one 
            elif i != 0 and i != num_changes-1:
                #print("initial:", warm_start[i])
                jitter = -1
                while jitter <= prev_jitter:
                    jitter = sample_truncated_normal(center=warm_start[i], min_=warm_start[i-1], max_=warm_start[i+1])
                #print("jitter:", jitter)
                jittered_list.append(jitter)
                prev_jitter = jitter 
                # CODE FOR FIXING 
            else: 
                # I don't think we need all this, but whatever 
                last_jitter = 50
                while last_jitter > 32 or last_jitter <= prev_jitter:
                    last_jitter = sample_truncated_normal(center=warm_start[i], min_=warm_start[i-1], max_=warm_start[i]+warm_start[i]-warm_start[i-1])
                    #last_jitter = sample_truncated_normal(center=36, min_=32, max_=40)
                #print("last_jitter:",last_jitter) 
                jittered_list.append(last_jitter)
        if jittered_list not in all_children: 
            all_children.append(replace_with_peel(peel, jittered_list[:])) #CHANGE HERE, ADDED THIS
            j = j+1 # and this! 
            #if peel in jittered_list: ### IF WE ITERATE THROUGH PEELS, REMOVE THIS IF STATEMENT (OR MAKE IT ARBITRARY) 
                #all_children.append(jittered_list)
                #j = j+1
                


    return all_children, parent_list
    

# %%
# returns teh top 4 from a list (can be changed to not be 4) 

def best_from_list(children, tested_list,peel,initial_order):
    my_dict = {}
    for child in children: 
        if child not in tested_list: # ADD A FOR LOOP HERE, FOR PEEL IN CHILDREN: 
            the_time = black_box(child, peel, initial_order)
            tested_list.append(child)
            if len(my_dict) < 4 or the_time < max(my_dict.values()):
                my_dict[tuple(child)] = the_time
                if len(my_dict) > 4:
                    # Remove the key with the largest value
                    del my_dict[max(my_dict, key=my_dict.get)]
    return my_dict, tested_list

# %%
# make a function that, given a peel (half lap number), initial_order, and num_changes, outputs the fastest (according to genetic algorithm) 
# maybe also have inputs for number of jittered children from seed, number of seeds chosen at each round, and number of rounds 

def genetic_algorithm(peel, initial_order, acceleration_length, num_changes, num_children = 10, num_seeds = 4, num_rounds = 5):
    warm_start = np.linspace(acceleration_length+1, 31.9, num=num_changes, dtype=int).tolist()
    parent_list = []
    fxn_output = create_jittered_kids(warm_start, acceleration_length,num_changes,num_children,peel,parent_list)
    all_children_from_fxn = fxn_output[0]
    parent_list = fxn_output[1]
    

    # now, we have a function that takes an input, and gives 10 jittered lists. (along with keeping up the parent_list
    # next, we need to take in that list of lists, and output the 4 best, along with keeping a running tab on what we have already tried

    tested_list = []
    fxn_output2 = best_from_list(all_children_from_fxn, tested_list,peel,initial_order)
    dict_of_top_4 = fxn_output2[0]
    tested_list = fxn_output2[1]

    # THAT WAS INITIAL ROUND, WE DON'T REALLY COUNT IT 

    # now, we have a dict, where the keys are the top 4 (we need to re-listify them) 
    list_of_active_parents = [list(key) for key in dict_of_top_4.keys()]
    
    for i in range(num_rounds):
        for a_list in list_of_active_parents:
            if a_list not in parent_list: 
                output = create_jittered_kids(a_list, acceleration_length,num_changes,num_children,peel,parent_list)
                all_kids = output[0]
                parent_list = output[1]
                for a_kid in all_kids: 
                    if a_kid not in tested_list: # ADD A FOR LOOP HERE, FOR PEEL IN A_KID: 
                        time_for_this_kid = black_box(a_kid, peel, initial_order)
                        tested_list.append(a_kid)
                        if time_for_this_kid < max(dict_of_top_4.values()):
                            dict_of_top_4[tuple(a_kid)] = time_for_this_kid
                            del dict_of_top_4[max(dict_of_top_4, key=dict_of_top_4.get)]
        list_of_active_parents = [list(key) for key in dict_of_top_4.keys()]
        

    # AT THIS POINT, we should have dict_of_top_4 that has been updated quite well...
    # IMPROVEMENT: KEEP A THRESHOLD FOR IMPROVEMENT THAT ONCE WE MEET IT, WE ARE GOOD
    # OR JUST KEEP GOING AFTER NUM_ROUNDS IF WE ARE IMPROVING 

    time_of_race = min(dict_of_top_4.values())
    schedule_of_switches = min(dict_of_top_4, key=dict_of_top_4.get)

    # Or, we could just have it output all 4 

    
        # for list in list_of_active_parents 
        # if not in parent_list:
            # create_jittered_kids
            # update parent list
            # for loop of children from jittering
                # best_from_list added to dict_of_top_4 if they are good enough 
                # also update list_of_active_parents ??? NO!
        # after we have gone through entire list_of_active_parents, create list_of_active_parents_2
        #list_of_active_parents = list_of_active_parents2
            # or, do we break if these are the same? I vote no, we want to keep jittering and trying again 

    # ENSURE WE DO THIS EACH TIME WE ARE CALLING A FUNCTION!!!
    # maintain a master list so we know what we have already tried 
    # maintain a parent list so we know what we have already jittered 
   
    # also could do more than 1... 

    sorted_dict = dict(sorted(dict_of_top_4.items(), key=lambda item: item[1]))

    
    return time_of_race, schedule_of_switches, sorted_dict
    

# # %%
# # testing it out just one time 

start1=time.time()
genetic_algorithm(peel=25, initial_order=[1,2,3,4], acceleration_length=3, num_changes=14, num_children = 10, num_seeds = 4, num_rounds = 5)
end1=time.time()
# print(f"Running took {end1 - start1:.4f} seconds")
# We need to run this for all 20+ peels, 24 initial orders, 2 acceleration lengths, and 10 ish peels

# %%
# print(counter)

# %%
# Looping through all of the orders 

counter = 0 # reset the globabl variable 
all_orders = list(itertools.permutations([1, 2, 3, 4]))
list_of_initial_orders =  [list(p) for p in itertools.permutations([0,1,2,3])]

start = time.time()
huge_dict = {}
for acceleration_length in range(3,4):
    for peel in range(10,33):
    #for peel in range(15,29):
        for initial_order in all_orders: 
            for num_changes in (3,4,5,6,7,8,9,10,11,12):
            #for num_changes in (14,15):
                initial_dict = genetic_algorithm(peel=peel, initial_order=initial_order, acceleration_length=acceleration_length, num_changes=num_changes, num_children = 50, num_seeds = 4, num_rounds = 20)[2]
                huge_dict.update(initial_dict)
end=time.time()
print(f"Loop took {end - start:.4f} seconds")

# sorted_final_dict = dict(sorted(huge_dict.items(), key=lambda item: item[1]))
# print(list(sorted_final_dict.items())[:5])

# # What if, instead of looping through peels, for each iteration we tried every conceivable peel? That might take longer, but it might take shorter...
# # It will decrease the time spent on create_jittered_kids, but it will increase the calls to black_box

# # fix create_jittered_kids() to not have to loop through as much (nearest to peel just gets sent to peel) 
# # Next steps probalby is to use some sort of time.time() to figure out what parts of this code are taking the longest to run 

# # %%
# # 4 riders, acceleration phase length of 3 or 4 half laps, peel between half laps 15 and 29, between 7 and 15 changes, black_box takes 0 extra seconds: 
# # 389 seconds 

# ### Post peel update, the above takes: 
# # 365 seconds... saved some time! :) 
# # this is the pre-loaded output btw

# # 4 riders, acceleration phase length of 3 or 4 half laps, peel between half laps 10 and 32, between 7 and 15 changes, black box takes 0 extra seconds: 
# # 666.3641 seconds 

# # woah, this time took 2145 seconds (35 minutes) 
# # calls black box 228633 times 
# # (6.5 hours if each call takes 1 minute! that's not bad!) 
# # 7 hours total... run it over night...
# # we could think about decreasing/increasing the num_children, num_seeds, num_rounds 

# # %%
# print(counter)



# %%
