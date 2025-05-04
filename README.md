# 🧠 3D Image & Text to Model Generator

A lightweight prototype toolset to generate 3D models from **2D images** or **natural language text prompts** using classical computer vision and geometric shape generation.

---

## ✨ Features

- 🖼️ Convert images to 3D meshes using depth estimation & segmentation
- 📝 Generate 3D geometric primitives from text prompts
- 💾 Export models to `.obj`, `.stl`, or `.ply` formats
- 👁️ View meshes interactively using built-in or external 3D viewers

---

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/sayedhanzala/3d-image-generator.git
   cd 3d-model-generator
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. _(Optional: for built-in viewer support)_
   ```bash
   pip install pyglet
   ```

---

## 🚀 Usage

### 1️⃣ Image to 3D Model

Convert a 2D image (e.g., photo of an object) into a 3D mesh.

```bash
python main.py --input images/chair.jpg --output outputs/chair.obj
```

**Options:**

- `--output` — Choose format: `.obj`, `.stl`, `.ply`

---

### 2️⃣ Text to 3D Model

Generate simple 3D shapes from text prompts.

```bash
python main.py --text "a car" --output outputs/car.obj
```

**Supported Shapes:**
`cube`, `sphere`, `cylinder`, `cone`, `torus`, `car`, `bottle`, `chair`

The model will be saved to the `outputs/` directory.

---

## 📁 Project Structure

```text
.
├── image_to_3d.py       # Image-to-mesh script
├── text_to_3d.py        # Text-to-shape script
├── image2model.py       # Alternate image-to-3D version
├── main.py              # Unified CLI
├── outputs/             # Output directory for generated models
```

---

## ⚙️ Dependencies

- `numpy`
- `trimesh`
- `open3d`
- `opencv-python`
- `scipy`
- `matplotlib`
- _(optional)_ `pyglet` — for 3D viewer

---

## 🛠️ Troubleshooting

- **3D Viewer not working?**  
  Try installing:

  ```bash
  pip install pyglet
  ```

  Or open the generated mesh in [MeshLab](https://www.meshlab.net/) or [Blender](https://www.blender.org/).

- **Open3D visualization issues on headless servers?**  
  Use the `--no-viz` flag to disable GUI-based previews.

---

## 📬 License & Contributions

Open-source under MIT License. Contributions, feature requests, and improvements are welcome!
