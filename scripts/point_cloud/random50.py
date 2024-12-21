import random

# Function to generate a random point
def generate_random_point(index):
    x, y, z = [round(random.uniform(-1, 1), 6) for _ in range(3)]  # Random X, Y, Z
    r, g, b = [random.randint(0, 255) for _ in range(3)]  # Random R, G, B
    return f"{index} {x} {y} {z} {r} {g} {b} 0 0 0"

# Generate 50 random points
num_points = 50
random_points = [generate_random_point(i + 1) for i in range(num_points)]

# Write points to a string in the desired format
points_text = "\n".join(random_points)

# Save to file
file_path = "points3D.txt"
with open(file_path, "w") as file:
    file.write(points_text)