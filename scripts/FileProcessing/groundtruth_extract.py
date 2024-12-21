# Define file paths
groundtruth_path = '/home/wang/catkin_ws/src/3dgs-dataset/tum-fg2-desk-orb/groundtruth.txt'
images_path = '/home/wang/catkin_ws/src/3dgs-dataset/tum-fg2-desk-orb/sparse-keyframe/0/images.txt'
images_new_path = 'images.txt'

# Function to round timestamps to 4 decimal places
def round_timestamp(timestamp):
    return round(float(timestamp), 4)

# Convert TWC to TCW
def convert_TWC_to_TCW(tx, ty, tz, qx, qy, qz, qw):
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # Translation vector and rotation matrix for TWC
    TWC_translation = np.array([tx, ty, tz])
    TWC_rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # Invert TWC to get TCW
    TCW_rotation = TWC_rotation.T
    TCW_translation = -np.dot(TCW_rotation, TWC_translation)

    # Convert back to quaternion
    TCW_quaternion = R.from_matrix(TCW_rotation).as_quat()
    return TCW_translation.tolist() + TCW_quaternion.tolist()

# Parse groundtruth.txt
groundtruth = {}
with open(groundtruth_path, 'r') as gt_file:
    for line in gt_file:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split()
        timestamp = round_timestamp(parts[0])
        tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:])
        groundtruth[timestamp] = convert_TWC_to_TCW(tx, ty, tz, qx, qy, qz, qw)

# Sort groundtruth timestamps
sorted_timestamps = sorted(groundtruth.keys())

def find_closest_timestamp(target, sorted_timestamps):
    """Find the closest timestamp in sorted_timestamps to the target."""
    from bisect import bisect_left
    pos = bisect_left(sorted_timestamps, target)
    if pos == 0:
        return sorted_timestamps[0]
    if pos == len(sorted_timestamps):
        return sorted_timestamps[-1]
    before = sorted_timestamps[pos - 1]
    after = sorted_timestamps[pos]
    return before if abs(before - target) <= abs(after - target) else after

# Parse images.txt and replace values
new_lines = []
with open(images_path, 'r') as img_file:
    lines = img_file.readlines()
    i = 0
    while i < len(lines):
        line1 = lines[i].strip()
        line2 = lines[i + 1].strip() if i + 1 < len(lines) else ""
        
        # Parse the first line for timestamp and other info
        if line1.startswith('#') or not line1:
            new_lines.append(line1)
            i += 1
            continue

        parts = line1.split()
        image_name = parts[-1]
        timestamp_str = image_name.replace('.png', '')
        timestamp = round_timestamp(timestamp_str)

        # Find the closest timestamp in groundtruth
        closest_timestamp = find_closest_timestamp(timestamp, sorted_timestamps)
        
        # Replace values if groundtruth exists
        if closest_timestamp in groundtruth:
            tx, ty, tz, qx, qy, qz, qw = groundtruth[closest_timestamp]
            parts[1:8] = [qw, qx, qy, qz, tx, ty, tz]
        
        # Append modified line1 and original line2 to new lines
        new_lines.append(" ".join(map(str, parts)))
        new_lines.append(line2)
        i += 2

# Save to images_new.txt
with open(images_new_path, 'w') as new_file:
    new_file.write("\n".join(new_lines))
