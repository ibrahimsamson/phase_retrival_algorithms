from bit_plane_slicing import bit_plane_slicing
from genetic_GS_algorithm import gerchberg_saxton_genetic_algorithm

#  Perform a 3
retrived_img = gerchberg_saxton_genetic_algorithm(measured_intensity, target_image, 
                                                  n_generations=100, population_size=100, 
                                                  mutation_rate=0.01)

sliced_img = bit_plane_slicing(retrived_img, 2)