import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os


def segment_object(image):
    # Ensure image is in RGB format for processing
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image.copy()

    # Initialize mask for GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define a rectangle covering most of the image (leaving a margin)
    margin = int(min(image.shape[0], image.shape[1]) * 0.1)  # 10% margin
    rect = (margin, margin, image.shape[1] - 2 * margin, image.shape[0] - 2 * margin)

    # Models required by GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Run GrabCut segmentation
    cv2.grabCut(img_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Extract probable foreground
    foreground_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)

    # Refine mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=1)

    return foreground_mask


def create_depth_map(image, mask):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Normalize grayscale image
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Edge detection (Canny)
    edges = cv2.Canny(gray, 100, 200)

    # Gradient magnitude (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 1, cv2.NORM_MINMAX)

    # Distance transform (distance from background)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)

    # Inverted brightness (brighter = closer)
    inverted_brightness = 1.0 - (gray / 255.0)

    # Weighted combination of cues
    depth_map = 0.5 * inverted_brightness + 0.2 * gradient_mag + 0.3 * dist_transform

    # Smooth the depth map
    depth_map = gaussian_filter(depth_map, sigma=2)

    # Apply mask to keep only foreground
    depth_map = depth_map * mask

    # Bilateral filter to preserve edges
    depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), 9, 75, 75)

    # Normalize final depth map
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    return depth_map


def create_3d_mesh(image, mask, depth_map, output_file, visualize=True):
    height, width = depth_map.shape

    # Create grid of coordinates
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Select only foreground points
    mask_indices = np.where(mask > 0)
    x_masked = x_coords[mask_indices]
    y_masked = y_coords[mask_indices]
    z_masked = depth_map[mask_indices] * 30  # Depth scaling for visibility

    # Get colors from the original image
    if len(image.shape) == 3:
        colors = image[mask_indices] / 255.0  # Normalize to [0, 1]
        if image.shape[2] == 3:
            colors = colors[:, [2, 1, 0]]  # Convert BGR to RGB
    else:
        colors = np.ones((len(x_masked), 3)) * 0.7  # Default gray

    # Stack coordinates into point cloud
    points = np.vstack((x_masked, y_masked, z_masked)).T

    # Center and normalize point cloud
    center = np.mean(points, axis=0)
    points = points - center
    scale = np.max(np.abs(points))
    points = points / scale

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate and orient normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, -1.0])
    )

    # Create mesh using Poisson reconstruction
    print("Creating mesh...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.5, linear_fit=True
    )

    # Remove low-density vertices to clean up mesh
    vertices_to_remove = densities < np.quantile(densities, 0.15)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Clean mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Transfer colors to mesh vertices
    if len(np.asarray(pcd.colors)) > 0:
        try:
            vertices = np.asarray(mesh.vertices)
            pcd_points = np.asarray(pcd.points)
            pcd_colors = np.asarray(pcd.colors)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            vertex_colors = []
            for vertex in vertices:
                _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
                if len(idx) > 0:
                    vertex_colors.append(pcd_colors[idx[0]])
                else:
                    vertex_colors.append([0.7, 0.7, 0.7])  # Default gray
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))
        except Exception as e:
            print(f"Color transfer failed: {e}")
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
    else:
        mesh.paint_uniform_color([0.7, 0.7, 0.7])

    # Smooth mesh for better appearance
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.compute_vertex_normals()

    # Save mesh to file
    print(f"Saving mesh to {output_file}")
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_extension = output_file.split(".")[-1].lower()
    if file_extension == "obj":
        o3d.io.write_triangle_mesh(output_file, mesh, write_vertex_normals=True)
    elif file_extension == "stl":
        o3d.io.write_triangle_mesh(output_file, mesh, write_ascii=True)
    elif file_extension == "ply":
        o3d.io.write_triangle_mesh(output_file, mesh, write_vertex_normals=True)
    else:
        output_file = output_file + ".obj"
        o3d.io.write_triangle_mesh(output_file, mesh, write_vertex_normals=True)
        print(f"Unrecognized extension, saved as {output_file}")

    # Visualize mesh if requested
    if visualize:
        print("Visualizing 3D mesh...")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.mesh_show_wireframe = False
        opt.light_on = True
        opt.background_color = np.array([0.8, 0.8, 0.8])
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        vis.run()
        vis.destroy_window()

    return mesh


def generate_3d_from_image(image_path, output_path="output/model.obj", visualize=True):
    """
    Main workflow: Loads an image, segments the object, creates a depth map, and generates a 3D mesh.
    """
    print(f"Processing image: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Resize if image is too large
    max_dim = 800
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        print(f"Resized image to {image.shape[1]}x{image.shape[0]}")

    # Segment object
    print("Segmenting object...")
    mask = segment_object(image)

    # Create depth map
    print("Creating depth map...")
    depth_map = create_depth_map(image, mask)

    # Generate and save 3D mesh
    mesh = create_3d_mesh(image, mask, depth_map, output_path, visualize)

    print(f"Completed 3D model generation: {output_path}")
    return mesh
