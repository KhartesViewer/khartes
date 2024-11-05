import sys
import os

def convert_obj_to_zyx(obj_path):
    output_path = obj_path.rsplit('.', 1)[0] + '_zyx.txt'
    
    # Collect all vertices first
    vertices = []
    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):  # vertex lines only
                _, x, y, z, _, _, _ = line.split()
                vertices.append((float(z), float(y), float(x)))
    
    # Sort by z-value and write to file
    vertices.sort(key=lambda v: v[0])
    with open(output_path, 'w') as out_file:
        for z, y, x in vertices:
            out_file.write(f"{round(z)}, {round(y)}, {round(x)}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python khartes_obj_to_umbilicus_points.py path/to/file.obj")
        sys.exit(1)
        
    obj_path = sys.argv[1]
    convert_obj_to_zyx(obj_path)
