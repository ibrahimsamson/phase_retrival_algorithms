# phase_retrival_algorithms
Using Gerchberg Saxton Method of phase retrival in DOE 

GERCHBERG SAXTON PHASE RETRIEVAL ALGORITHM
To use this algorithm, simply call the gerchberg_saxton function with the 
measured intensity distribution, target image, and desired number of iterations:


updated_image = gerchberg_saxton(measured_intensity, target_image, n_iterations)

Note that the measured_intensity and target_image should be 2D arrays of the same shape, 
and n_iterations is an integer specifying the number of iterations to perform.


# GERCHBERG SAXTON GENETIC PHASE RETRIEVAL ALGORITHM

gerchberg_saxton_genetic_algorithm function takes in a measured intensity distribution and a target image, as well as the desired parameters for the genetic algorithm such as the number of generations, population size, and mutation rate. It then uses the Gerchberg-Saxton algorithm to iteratively update the phase of the measured intensity distribution until the resulting intensity distribution matches the target image as closely as possible.

In each iteration, the algorithm selects a subset of the previous phase estimate and uses a genetic algorithm to generate a new set of candidate phase estimates. The genetic algorithm uses uniform crossover and mutation to generate offspring from the parent chromosomes, and then selects the best chromosomes to form the next generation. The process repeats for a fixed number of generations until the final phase estimate is obtained.

The resulting phase estimate can then be combined with the original measured intensity distribution to obtain the final reconstructed image. The gerchberg_saxton_genetic_algorithm function returns the final updated image.



To use this algorithm, simply call the gerchberg_saxton_genetic_algorithm
function with the measured intensity distribution, target image, and desired 
genetic algorithm parameters:

measured_intensity and target_image should be 2D arrays of the same shape.

- n_generations is an integer specifying the number of generations to run the genetic algorithm for.
- population_size is an integer specifying the size of the population.
- mutation_rate is a float specifying the probability of a chromosome mutating during reproduction.

The gerchberg_saxton_genetic_algorithm function returns the final updated image.

# Bit Plane Slicing
In this code, we use the OpenCV library to load a grayscale image and extract the specified bit plane. The np.bitwise_and function is used to perform a bitwise AND operation between the image and a binary number that has a 1 in the bit position corresponding to the desired bit plane (e.g., 2^2 for bit plane 2). We then scale the resulting bit plane values to the range 0-255 and convert them to the uint8 data type, which is the data type expected by OpenCV for image display. Finally, we display the bit plane image using the cv2.imshow function.

# Example usage: display bit plane 2 of a grayscale image
bit_plane_slicing("image.png", 2)