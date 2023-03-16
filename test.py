from src.bias.Distance import Distance
import numpy as np

male_0_ref = 75/100
male_1_ref = 25/100

female_0_ref = 75/100
female_1_ref = 25/100

male_0_freq = 80/100
male_1_freq = 20/100

female_0_freq = 70/100
female_1_freq = 30/100

ref = [np.array([male_0_ref, male_1_ref]), np.array([female_0_ref, female_1_ref])]
obs = [np.array([male_0_freq, male_1_freq]), np.array([female_0_freq, female_1_freq])]

d = Distance()

distances = d.compute_distance_from_reference(obs, ref)

print(distances)

answer_male = [max( abs(male_0_ref - male_0_freq), abs(male_1_ref - male_1_freq) )]
answer_female = [max( abs(female_0_ref - female_0_freq), abs(female_1_ref - female_1_freq) )]

assert distances == [answer_male, answer_female]


freq_distance = d.compute_distance_between_frequencies(obs)
print(freq_distance)
answer = max( abs(female_0_freq - male_0_freq), abs(female_1_freq - male_1_freq) )
print(answer)
assert freq_distance == answer
