import os
import numpy as np
import trimesh
import re
import math


class SimpleShapeGenerator:
    def __init__(self):
        # Ensure output directory exists
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        # Map shape keywords to their creation methods
        self.shapes = {
            "cube": self.create_cube,
            "sphere": self.create_sphere,
            "cylinder": self.create_cylinder,
            "cone": self.create_cone,
            "torus": self.create_torus,
            "car": self.create_car,
            "bottle": self.create_bottle,
            "chair": self.create_chair,
        }

    def create_cube(self, size=1.0):
        """Return a cube mesh of given size."""
        return trimesh.creation.box(extents=[size, size, size])

    def create_sphere(self, radius=1.0):
        """Return a sphere mesh of given radius."""
        return trimesh.creation.icosphere(radius=radius)

    def create_cylinder(self, radius=0.5, height=2.0):
        """Return a cylinder mesh of given radius and height."""
        return trimesh.creation.cylinder(radius=radius, height=height)

    def create_cone(self, radius=1.0, height=2.0):
        """Return a cone mesh of given base radius and height."""
        return trimesh.creation.cone(radius=radius, height=height)

    def create_torus(self, radius=1.0, tube_radius=0.2):
        """Return a torus mesh (donut shape) with given major and minor radii."""
        major_radius = radius
        minor_radius = tube_radius
        resolution = 32
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, 2 * np.pi, resolution)
        u_grid, v_grid = np.meshgrid(u, v)
        x = (major_radius + minor_radius * np.cos(v_grid)) * np.cos(u_grid)
        y = (major_radius + minor_radius * np.cos(v_grid)) * np.sin(u_grid)
        z = minor_radius * np.sin(v_grid)
        vertices = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                a = i * resolution + j
                b = i * resolution + (j + 1)
                c = (i + 1) * resolution + j
                d = (i + 1) * resolution + (j + 1)
                faces.append([a, b, c])
                faces.append([b, d, c])
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def create_car(self):
        """Return a simple car mesh made from boxes and cylinders."""
        # Main body
        body = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
        body.apply_translation([0, 0, 0.5])
        # Cabin
        cabin = trimesh.creation.box(extents=[1.0, 0.8, 0.4])
        cabin.apply_translation([0, 0, 0.95])
        # Wheels (rotated cylinders)
        wheel_radius = 0.2
        wheel_width = 0.1
        wheel_sections = 16
        wheel_rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        wheel1 = trimesh.creation.cylinder(
            radius=wheel_radius, height=wheel_width, sections=wheel_sections
        )
        wheel1.apply_transform(wheel_rot)
        wheel1.apply_translation([0.5, 0.55, 0.2])
        wheel2 = trimesh.creation.cylinder(
            radius=wheel_radius, height=wheel_width, sections=wheel_sections
        )
        wheel2.apply_transform(wheel_rot)
        wheel2.apply_translation([-0.5, 0.55, 0.2])
        wheel3 = trimesh.creation.cylinder(
            radius=wheel_radius, height=wheel_width, sections=wheel_sections
        )
        wheel3.apply_transform(wheel_rot)
        wheel3.apply_translation([0.5, -0.55, 0.2])
        wheel4 = trimesh.creation.cylinder(
            radius=wheel_radius, height=wheel_width, sections=wheel_sections
        )
        wheel4.apply_transform(wheel_rot)
        wheel4.apply_translation([-0.5, -0.55, 0.2])
        # Combine all parts
        return trimesh.util.concatenate([body, cabin, wheel1, wheel2, wheel3, wheel4])

    def create_bottle(self):
        """Return a simple bottle mesh made from cylinders."""
        body = trimesh.creation.cylinder(radius=0.5, height=2.0)
        body.apply_translation([0, 0, 1.0])
        neck = trimesh.creation.cylinder(radius=0.2, height=0.5)
        neck.apply_translation([0, 0, 2.25])
        cap = trimesh.creation.cylinder(radius=0.25, height=0.2)
        cap.apply_translation([0, 0, 2.6])
        bottom = trimesh.creation.cylinder(radius=0.5, height=0.1)
        bottom.apply_translation([0, 0, 0.05])
        return trimesh.util.concatenate([body, neck, cap, bottom])

    def create_chair(self):
        """Return a simple chair mesh made from boxes and cylinders."""
        seat = trimesh.creation.box(extents=[1.0, 1.0, 0.1])
        seat.apply_translation([0, 0, 0.5])
        backrest = trimesh.creation.box(extents=[1.0, 0.1, 1.0])
        backrest.apply_translation([0, -0.45, 1.0])
        leg_radius = 0.05
        leg_height = 0.5
        leg1 = trimesh.creation.cylinder(radius=leg_radius, height=leg_height)
        leg1.apply_translation([0.4, 0.4, 0.25])
        leg2 = trimesh.creation.cylinder(radius=leg_radius, height=leg_height)
        leg2.apply_translation([-0.4, 0.4, 0.25])
        leg3 = trimesh.creation.cylinder(radius=leg_radius, height=leg_height)
        leg3.apply_translation([0.4, -0.4, 0.25])
        leg4 = trimesh.creation.cylinder(radius=leg_radius, height=leg_height)
        leg4.apply_translation([-0.4, -0.4, 0.25])
        return trimesh.util.concatenate([seat, backrest, leg1, leg2, leg3, leg4])

    def parse_prompt(self, prompt):
        """Return the shape type found in the prompt, or 'cube' if none found."""
        clean_prompt = prompt.lower().strip()
        for shape_name in self.shapes.keys():
            if shape_name in clean_prompt:
                return shape_name
        return "cube"

    def generate_from_text(self, prompt, output_format="obj"):
        """Generate a 3D model from text and save as OBJ or STL."""
        shape_type = self.parse_prompt(prompt)
        clean_name = re.sub(r"[^a-z0-9]", "_", prompt.lower())[:20]
        output_path = os.path.join(self.output_dir, f"{clean_name}.{output_format}")
        print(f"Creating a {shape_type} shape from '{prompt}'")
        mesh = self.shapes[shape_type]()
        mesh.export(output_path)
        print(f"3D model saved to {output_path}")
        return output_path


def generate_3d_from_text(prompt):
    """Generate and display a 3D model from a text description."""
    print("This tool creates basic 3D shapes based on text descriptions.")
    print("Available shapes: cube, sphere, cylinder, cone, torus, car, bottle, chair")
    generator = SimpleShapeGenerator()
    try:
        output_path = generator.generate_from_text(prompt)
        print(f"\nDone! Model saved as: {output_path}")
        print("You can view this file in any 3D viewer or CAD software.")
        # Try to display the model
        try:
            mesh = trimesh.load(output_path)
            mesh.show()
        except Exception as e:
            print(f"Could not display mesh: {e}")
            print("Please use an external 3D viewer to view the model.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please try a different shape or check if trimesh is properly installed.")
