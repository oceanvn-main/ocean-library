#!/usr/bin/env python3
"""
viewer_3d.py

Interactive 3D Viewer for Points, Lines, and Polygons.
Szeliski Computer Vision Exercise 2.3.
- Import a set of point and line commands (primitives).
- Interactively modify object or camera transforms.
- Perspective and Orthographic projection models.
- Hidden surface removal using Painter's Algorithm.
- Mouse drag to rotate the camera.

Author: Antigravity AI
Date: 2026-06-10
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np

# Styling constants consistent with 2D transform editor
COLOR_BG = "#111113"
COLOR_PANEL = "#17191d"
COLOR_CANVAS = "#101216"
COLOR_GRID = "#25272d"
COLOR_TEXT = "#e0e0e0"
COLOR_ACCENT = "#3498db"
COLOR_SECONDARY = "#e74c3c"
COLOR_MUTED = "#55585d"


class PrimitiveScene:
    def __init__(self):
        self.points = []       # list of np.array([x, y, z, 1.0])
        self.lines = []        # list of (idx1, idx2)
        self.polygons = []     # list of list of idxs
        self.poly_colors = []  # list of colors for polygons

    def clear(self):
        self.points = []
        self.lines = []
        self.polygons = []
        self.poly_colors = []

    def load_default_house(self):
        self.clear()
        
        # 3D wireframe house coordinates centered around origin
        # 10 vertices
        vertices = [
            [-1.5, -1.5, -1.0], # 0: floor corners
            [ 1.5, -1.5, -1.0], # 1
            [ 1.5,  1.5, -1.0], # 2
            [-1.5,  1.5, -1.0], # 3
            [-1.5, -1.5,  0.5], # 4: wall tops
            [ 1.5, -1.5,  0.5], # 5
            [ 1.5,  1.5,  0.5], # 6
            [-1.5,  1.5,  0.5], # 7
            [ 0.0, -1.5,  1.5], # 8: roof ridge front
            [ 0.0,  1.5,  1.5]  # 9: roof ridge back
        ]
        for v in vertices:
            self.points.append(np.array([v[0], v[1], v[2], 1.0]))

        # House Lines
        lines = [
            (0, 1), (1, 2), (2, 3), (3, 0), # floor
            (4, 5), (5, 6), (6, 7), (7, 4), # wall tops
            (0, 4), (1, 5), (2, 6), (3, 7), # pillars
            (4, 8), (5, 8), (6, 9), (7, 9), (8, 9) # roof
        ]
        self.lines = lines

        # House Polygons (for hidden surface removal)
        # Floor, Walls, Roof
        self.polygons = [
            [0, 1, 2, 3],       # 0: floor
            [0, 1, 5, 4],       # 1: front wall
            [1, 2, 6, 5],       # 2: right wall
            [2, 3, 7, 6],       # 3: back wall
            [3, 0, 4, 7],       # 4: left wall
            [4, 5, 8],          # 5: roof front triangle
            [6, 7, 9],          # 6: roof back triangle
            [4, 7, 9, 8],       # 7: roof left slope
            [5, 6, 9, 8]        # 8: roof right slope
        ]
        
        # Color coding
        self.poly_colors = [
            "#2c3e50", # floor
            "#34495e", # front wall
            "#7f8c8d", # right wall
            "#95a5a6", # back wall
            "#bdc3c7", # left wall
            "#e67e22", # roof front
            "#d35400", # roof back
            "#e74c3c", # roof left
            "#c0392b"  # roof right
        ]


class Viewer3DApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3D Primitives Viewer — Szeliski Ex 2.3")
        self.geometry("1150x750")
        self.configure(bg=COLOR_BG)

        self.scene = PrimitiveScene()
        self.scene.load_default_house()

        # Camera & Object State variables (Euler in degrees)
        self.cam_rx = tk.DoubleVar(value=15.0)
        self.cam_ry = tk.DoubleVar(value=-25.0)
        self.cam_rz = tk.DoubleVar(value=0.0)
        self.cam_tx = tk.DoubleVar(value=0.0)
        self.cam_ty = tk.DoubleVar(value=1.0)
        self.cam_tz = tk.DoubleVar(value=-8.0)
        
        self.obj_rx = tk.DoubleVar(value=0.0)
        self.obj_ry = tk.DoubleVar(value=0.0)
        self.obj_rz = tk.DoubleVar(value=0.0)
        self.obj_tx = tk.DoubleVar(value=0.0)
        self.obj_ty = tk.DoubleVar(value=0.0)
        self.obj_tz = tk.DoubleVar(value=0.0)
        self.obj_scale = tk.DoubleVar(value=1.0)

        self.projection_type = tk.StringVar(value="Perspective")
        self.focal_length = tk.DoubleVar(value=400.0) # focal length for perspective
        self.ortho_scale = tk.DoubleVar(value=80.0)   # scale for orthographic

        # Mouse dragging state for camera orbit rotation
        self.last_mouse_x = None
        self.last_mouse_y = None

        self.setup_ui()
        self.redraw()

    def setup_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # 1. Left Sidebar
        sidebar = tk.Frame(self, bg=COLOR_PANEL, width=320)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        sidebar.grid_propagate(False)

        # Scrollable container inside sidebar
        canvas_sidebar = tk.Canvas(sidebar, bg=COLOR_PANEL, highlightthickness=0)
        scrollbar = tk.Scrollbar(sidebar, orient="vertical", command=canvas_sidebar.yview)
        scroll_frame = tk.Frame(canvas_sidebar, bg=COLOR_PANEL)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas_sidebar.configure(scrollregion=canvas_sidebar.bbox("all"))
        )
        canvas_sidebar.create_window((0, 0), window=scroll_frame, anchor="nw", width=300)
        canvas_sidebar.configure(yscrollcommand=scrollbar.set)

        canvas_sidebar.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Tab A: Projection Model ---
        tk.Label(scroll_frame, text="Projection Type", font=("Helvetica", 11, "bold"), bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(anchor="w", padx=10, pady=(10, 5))
        p_frame = tk.Frame(scroll_frame, bg=COLOR_PANEL)
        p_frame.pack(fill="x", padx=10)
        tk.Radiobutton(p_frame, text="Perspective", variable=self.projection_type, value="Perspective", bg=COLOR_PANEL, fg=COLOR_TEXT, selectcolor=COLOR_PANEL, command=self.redraw).pack(side="left", padx=5)
        tk.Radiobutton(p_frame, text="Orthographic", variable=self.projection_type, value="Orthographic", bg=COLOR_PANEL, fg=COLOR_TEXT, selectcolor=COLOR_PANEL, command=self.redraw).pack(side="left", padx=5)

        # Sliders: Focal Length & Ortho Scale
        self.create_slider(scroll_frame, "Focal Length (f)", self.focal_length, 100.0, 1500.0, 10.0)
        self.create_slider(scroll_frame, "Ortho Scale (s)", self.ortho_scale, 10.0, 300.0, 1.0)

        tk.Frame(scroll_frame, height=1, bg=COLOR_GRID).pack(fill="x", padx=10, pady=10)

        # --- Tab B: Camera Transform ---
        tk.Label(scroll_frame, text="Camera Transform (Extrinsics)", font=("Helvetica", 11, "bold"), bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(anchor="w", padx=10, pady=(0, 5))
        self.create_slider(scroll_frame, "Cam RX (Pitch)", self.cam_rx, -180.0, 180.0, 1.0)
        self.create_slider(scroll_frame, "Cam RY (Yaw)", self.cam_ry, -180.0, 180.0, 1.0)
        self.create_slider(scroll_frame, "Cam RZ (Roll)", self.cam_rz, -180.0, 180.0, 1.0)
        self.create_slider(scroll_frame, "Cam TX", self.cam_tx, -10.0, 10.0, 0.1)
        self.create_slider(scroll_frame, "Cam TY", self.cam_ty, -10.0, 10.0, 0.1)
        self.create_slider(scroll_frame, "Cam TZ", self.cam_tz, -30.0, -1.0, 0.1)

        tk.Frame(scroll_frame, height=1, bg=COLOR_GRID).pack(fill="x", padx=10, pady=10)

        # --- Tab C: Object Transform ---
        tk.Label(scroll_frame, text="Object Transform (Model)", font=("Helvetica", 11, "bold"), bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(anchor="w", padx=10, pady=(0, 5))
        self.create_slider(scroll_frame, "Obj RX", self.obj_rx, -180.0, 180.0, 1.0)
        self.create_slider(scroll_frame, "Obj RY", self.obj_ry, -180.0, 180.0, 1.0)
        self.create_slider(scroll_frame, "Obj RZ", self.obj_rz, -180.0, 180.0, 1.0)
        self.create_slider(scroll_frame, "Obj TX", self.obj_tx, -10.0, 10.0, 0.1)
        self.create_slider(scroll_frame, "Obj TY", self.obj_ty, -10.0, 10.0, 0.1)
        self.create_slider(scroll_frame, "Obj TZ", self.obj_tz, -10.0, 10.0, 0.1)
        self.create_slider(scroll_frame, "Obj Scale", self.obj_scale, 0.1, 5.0, 0.05)

        tk.Frame(scroll_frame, height=1, bg=COLOR_GRID).pack(fill="x", padx=10, pady=10)

        # --- Tab D: Primitive Importer panel ---
        tk.Label(scroll_frame, text="Import Primitives", font=("Helvetica", 11, "bold"), bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(anchor="w", padx=10, pady=(0, 5))
        
        # Text panel
        self.import_text = tk.Text(scroll_frame, height=6, bg=COLOR_CANVAS, fg=COLOR_TEXT, insertbackground="white", font=("Courier", 9))
        self.import_text.pack(fill="x", padx=10, pady=5)
        # Prepopulate with a simple shape example
        default_cmd = (
            "# Commands format:\n"
            "# P x y z\n"
            "# L idx1 idx2\n"
            "# POLY idx1 idx2 idx3 ...\n"
        )
        self.import_text.insert("1.0", default_cmd)

        btn_row = tk.Frame(scroll_frame, bg=COLOR_PANEL)
        btn_row.pack(fill="x", padx=10, pady=5)
        tk.Button(btn_row, text="Import Scene", command=self.import_commands, bg=COLOR_ACCENT, fg="white", font=("Helvetica", 10, "bold"), relief="flat").pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(btn_row, text="Load Preset", command=self.load_preset_dialog, bg=COLOR_MUTED, fg="white", font=("Helvetica", 10, "bold"), relief="flat").pack(side="left", fill="x", expand=True, padx=2)

        # 2. Canvas
        self.canvas = tk.Canvas(self, bg=COLOR_CANVAS, highlightthickness=1, highlightbackground=COLOR_GRID)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Bindings for mouse camera rotation orbit
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)

    def create_slider(self, parent, label_text, var, from_val, to_val, resolution):
        frame = tk.Frame(parent, bg=COLOR_PANEL)
        frame.pack(fill="x", padx=10, pady=2)
        
        lbl_text = tk.Label(frame, text=label_text, font=("Helvetica", 9), bg=COLOR_PANEL, fg=COLOR_TEXT)
        lbl_text.pack(side="left")
        
        val_lbl = tk.Label(frame, text=f"{var.get():.2f}", font=("Helvetica", 9), bg=COLOR_PANEL, fg=COLOR_MUTED, width=6, anchor="e")
        val_lbl.pack(side="right")
        
        def update_val(val):
            v = float(val)
            var.set(v)
            val_lbl.config(text=f"{v:.2f}")
            self.redraw()

        slider = tk.Scale(
            parent, from_=from_val, to=to_val, resolution=resolution,
            orient="horizontal", showvalue=False, variable=var,
            bg=COLOR_PANEL, fg=COLOR_GRID, highlightthickness=0,
            activebackground=COLOR_ACCENT, troughcolor=COLOR_GRID,
            command=update_val
        )
        slider.pack(fill="x", padx=15, pady=(0, 5))

    def on_mouse_down(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_mouse_drag(self, event):
        if self.last_mouse_x is not None and self.last_mouse_y is not None:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            
            # Orbit rotation mapping:
            # Horizontal drag rotates around Y axis (Yaw)
            # Vertical drag rotates around X axis (Pitch)
            new_ry = self.cam_ry.get() - dx * 0.5
            new_rx = self.cam_rx.get() - dy * 0.5
            
            # Clamp RX (pitch) to avoid inversion
            new_rx = max(-90.0, min(90.0, new_rx))
            
            # wrap RY in [-180, 180]
            if new_ry > 180: new_ry -= 360
            if new_ry < -180: new_ry += 360

            self.cam_ry.set(new_ry)
            self.cam_rx.set(new_rx)

            # Update sliders (which will trigger redraw)
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            self.redraw()

    def get_rotation_matrix(self, rx, ry, rz):
        # rx, ry, rz in degrees
        phi = np.radians(rx)   # Roll
        theta = np.radians(ry) # Pitch
        psi = np.radians(rz)   # Yaw

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        # ZYX convention matching MATLAB Rotation3D.eulerToMatrix
        return Rz @ Ry @ Rx

    def redraw(self):
        self.canvas.delete("all")
        self.canvas.update()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        cx, cy = w / 2, h / 2

        if not self.scene.points:
            return

        # 1. Compute Object Pose Model Matrix
        R_obj = self.get_rotation_matrix(self.obj_rx.get(), self.obj_ry.get(), self.obj_rz.get())
        S_obj = self.obj_scale.get()
        T_obj = np.array([self.obj_tx.get(), self.obj_ty.get(), self.obj_tz.get()])
        
        M_obj = np.eye(4)
        M_obj[:3, :3] = R_obj * S_obj
        M_obj[:3, 3] = T_obj

        # 2. Compute Camera View Matrix (Inverse of camera pose transform)
        R_cam = self.get_rotation_matrix(self.cam_rx.get(), self.cam_ry.get(), self.cam_rz.get())
        T_cam = np.array([self.cam_tx.get(), self.cam_ty.get(), self.cam_tz.get()])
        
        # M_view = [ R_cam.T , -R_cam.T * T_cam ]
        M_view = np.eye(4)
        M_view[:3, :3] = R_cam.T
        M_view[:3, 3] = -R_cam.T @ T_cam

        # 3. Transform points to Camera Space
        pts_cam = []
        for p in self.scene.points:
            p_transformed = M_view @ M_obj @ p
            pts_cam.append(p_transformed)

        # 4. Project transformed points to Screen space
        proj_type = self.projection_type.get()
        pts_screen = []
        in_front = [] # Boolean mask for perspective clipping

        for p in pts_cam:
            x, y, z = p[0], p[1], p[2]
            if proj_type == "Perspective":
                # Perspective projection: x_screen = f * x / z + cx
                # Clip if behind near plane (z <= 0.1)
                if z > 0.1:
                    f = self.focal_length.get()
                    u = f * x / z + cx
                    v = -f * y / z + cy  # Invert Y coordinate
                    pts_screen.append((u, v))
                    in_front.append(True)
                else:
                    pts_screen.append((0, 0))
                    in_front.append(False)
            else:
                # Orthographic projection: x_screen = s * x + cx
                s = self.ortho_scale.get()
                u = s * x + cx
                v = -s * y + cy
                pts_screen.append((u, v))
                in_front.append(True)

        # --- RENDER STEP 1: Polygons with Hidden Surface Removal (Painter's Algorithm) ---
        # Sort polygons by their average depth (average Z in camera space)
        poly_depths = []
        for idx, poly in enumerate(self.scene.polygons):
            # Check if all vertices are in front of camera
            all_in_front = all(in_front[v_idx] for v_idx in poly)
            if not all_in_front:
                continue

            # Average Z depth in camera space
            z_depth = np.mean([pts_cam[v_idx][2] for v_idx in poly])
            poly_depths.append((z_depth, idx))

        # Sort descending (draw farthest first so closer polygons overlap them)
        poly_depths.sort(key=lambda x: x[0], reverse=True)

        # Draw sorted polygons
        for z_val, idx in poly_depths:
            poly = self.scene.polygons[idx]
            color = self.scene.poly_colors[idx] if idx < len(self.scene.poly_colors) else "#34495e"
            
            pts_draw = []
            for v_idx in poly:
                pts_draw.extend(pts_screen[v_idx])
            
            # Fill polygon
            self.canvas.create_polygon(
                pts_draw, fill=color, outline=COLOR_GRID, width=0.5, tags="poly"
            )

        # --- RENDER STEP 2: Line Primitives ---
        for line in self.scene.lines:
            idx1, idx2 = line
            if in_front[idx1] and in_front[idx2]:
                x1, y1 = pts_screen[idx1]
                x2, y2 = pts_screen[idx2]
                self.canvas.create_line(
                    x1, y1, x2, y2, fill=COLOR_ACCENT, width=1.5, tags="line"
                )

        # --- RENDER STEP 3: Point Primitives ---
        for i, pt in enumerate(pts_screen):
            if in_front[i]:
                x1, y1 = pt
                r = 3
                self.canvas.create_oval(
                    x1-r, y1-r, x1+r, y1+r, fill=COLOR_SECONDARY, 
                    outline="white", width=0.5, tags="point"
                )

    def import_commands(self):
        text = self.import_text.get("1.0", tk.END)
        lines = text.split("\n")

        new_pts = []
        new_lines = []
        new_polys = []

        line_num = 0
        try:
            for l in lines:
                line_num += 1
                l_strip = l.strip()
                if not l_strip or l_strip.startswith("#"):
                    continue

                tokens = l_strip.split()
                cmd = tokens[0].upper()

                if cmd == "P":
                    if len(tokens) < 4:
                        raise ValueError(f"Point command requires x, y, z values.")
                    x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
                    new_pts.append(np.array([x, y, z, 1.0]))

                elif cmd == "L":
                    if len(tokens) < 3:
                        raise ValueError(f"Line command requires two vertex indices.")
                    idx1, idx2 = int(tokens[1]), int(tokens[2])
                    new_lines.append((idx1, idx2))

                elif cmd == "POLY":
                    if len(tokens) < 4:
                        raise ValueError(f"Polygon command requires at least three vertex indices.")
                    idxs = [int(t) for t in tokens[1:]]
                    new_polys.append(idxs)

                else:
                    raise ValueError(f"Unknown command: '{cmd}'. Available: P, L, POLY")
            
            # Verification: index bounds checking
            n_pts = len(new_pts)
            for l_idx, line in enumerate(new_lines):
                if line[0] >= n_pts or line[1] >= n_pts or line[0] < 0 or line[1] < 0:
                    raise IndexError(f"Line {l_idx} refers to out-of-bounds indices. Total vertices: {n_pts}")
            for p_idx, poly in enumerate(new_polys):
                for v in poly:
                    if v >= n_pts or v < 0:
                        raise IndexError(f"Polygon {p_idx} refers to out-of-bounds index {v}. Total vertices: {n_pts}")

            # Apply scene import
            self.scene.clear()
            self.scene.points = new_pts
            self.scene.lines = new_lines
            self.scene.polygons = new_polys
            
            # Simple color cycle generator
            colors = ["#2c3e50", "#c0392b", "#27ae60", "#2980b9", "#f39c12", "#8e44ad", "#16a085", "#d35400"]
            self.scene.poly_colors = [colors[i % len(colors)] for i in range(len(new_polys))]

            self.redraw()
            messagebox.showinfo("Success", f"Scene imported successfully:\n- {len(new_pts)} vertices\n- {len(new_lines)} lines\n- {len(new_polys)} polygons")

        except Exception as e:
            messagebox.showerror("Import Error", f"Failed parsing command on line {line_num}:\n{str(e)}")

    def load_preset_dialog(self):
        preset_window = tk.Toplevel(self)
        preset_window.title("Select Preset")
        preset_window.geometry("250x150")
        preset_window.configure(bg=COLOR_PANEL)
        preset_window.grab_set()

        tk.Label(preset_window, text="Choose a 3D Preset Scene:", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Helvetica", 10, "bold")).pack(pady=15)
        
        def select(preset_name):
            if preset_name == "House":
                self.scene.load_default_house()
                # Update text box with house commands
                cmds = "# 3D Wireframe House Preset\n"
                for i, p in enumerate(self.scene.points):
                    cmds += f"P {p[0]:.2f} {p[1]:.2f} {p[2]:.2f}\n"
                for l in self.scene.lines:
                    cmds += f"L {l[0]} {l[1]}\n"
                for p in self.scene.polygons:
                    p_str = " ".join(str(idx) for idx in p)
                    cmds += f"POLY {p_str}\n"
                self.import_text.delete("1.0", tk.END)
                self.import_text.insert("1.0", cmds)
            
            elif preset_name == "Cube":
                self.scene.clear()
                # 8 corners
                vertices = [
                    [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]
                ]
                for v in vertices:
                    self.scene.points.append(np.array([v[0], v[1], v[2], 1.0]))
                self.scene.lines = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]
                self.scene.polygons = [
                    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]
                ]
                self.scene.poly_colors = ["#c0392b", "#27ae60", "#2980b9", "#f1c40f", "#8e44ad", "#e67e22"]
                
                cmds = "# 3D Cube Preset\n"
                for i, p in enumerate(self.scene.points):
                    cmds += f"P {p[0]:.2f} {p[1]:.2f} {p[2]:.2f}\n"
                for l in self.scene.lines:
                    cmds += f"L {l[0]} {l[1]}\n"
                for p in self.scene.polygons:
                    p_str = " ".join(str(idx) for idx in p)
                    cmds += f"POLY {p_str}\n"
                self.import_text.delete("1.0", tk.END)
                self.import_text.insert("1.0", cmds)

            elif preset_name == "Pyramid":
                self.scene.clear()
                # 5 corners (4 base + 1 apex)
                vertices = [
                    [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                    [0.0, 0.0, 1.0] # apex
                ]
                for v in vertices:
                    self.scene.points.append(np.array([v[0], v[1], v[2], 1.0]))
                self.scene.lines = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (0, 4), (1, 4), (2, 4), (3, 4)
                ]
                self.scene.polygons = [
                    [0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]
                ]
                self.scene.poly_colors = ["#2c3e50", "#2ecc71", "#3498db", "#e74c3c", "#f1c40f"]
                
                cmds = "# 3D Pyramid Preset\n"
                for i, p in enumerate(self.scene.points):
                    cmds += f"P {p[0]:.2f} {p[1]:.2f} {p[2]:.2f}\n"
                for l in self.scene.lines:
                    cmds += f"L {l[0]} {l[1]}\n"
                for p in self.scene.polygons:
                    p_str = " ".join(str(idx) for idx in p)
                    cmds += f"POLY {p_str}\n"
                self.import_text.delete("1.0", tk.END)
                self.import_text.insert("1.0", cmds)
                
            self.redraw()
            preset_window.destroy()

        tk.Button(preset_window, text="House", command=lambda: select("House"), bg=COLOR_ACCENT, fg="white", font=("Helvetica", 9), width=8).pack(side="left", padx=10)
        tk.Button(preset_window, text="Cube", command=lambda: select("Cube"), bg=COLOR_ACCENT, fg="white", font=("Helvetica", 9), width=8).pack(side="left", padx=10)
        tk.Button(preset_window, text="Pyramid", command=lambda: select("Pyramid"), bg=COLOR_ACCENT, fg="white", font=("Helvetica", 9), width=8).pack(side="left", padx=10)


if __name__ == "__main__":
    app = Viewer3DApp()
    app.mainloop()
