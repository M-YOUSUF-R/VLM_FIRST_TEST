import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import os

class ZoomableImageCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.orig_image = None
        self.zoom_level = 1.0  
        self.image_id = None
        self.tk_image = None

        # Bind zoom events
        self.bind("<MouseWheel>", self.handle_zoom)
        self.bind("<Button-4>", self.handle_zoom)
        self.bind("<Button-5>", self.handle_zoom)

    def load_new_image(self, image_path):
        """Loads a new image file and resets the zoom factor."""
        try:
            self.orig_image = Image.open(image_path)
        except Exception as e:
            # Fallback placeholder if image fails to load
            self.orig_image = Image.new("RGB", (400, 400), color="darkgray")
        
        self.zoom_level = 1.0
        self.show_image()

    def show_image(self):
        if not self.orig_image:
            return

        if self.zoom_level <= 0.05:
            self.zoom_level = 0.05 

        new_width = int(self.orig_image.width * self.zoom_level)
        new_height = int(self.orig_image.height * self.zoom_level)

        resized_img = self.orig_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_img)

        # Delete previous drawings to keep canvas lightweight
        self.delete("all")
        self.image_id = self.create_image(0, 0, anchor="nw", image=self.tk_image)
        
        self.config(scrollregion=(0, 0, new_width, new_height))

    def handle_zoom(self, event):
        if event.num == 4 or event.delta > 0:
            self.zoom_level *= 1.1  
        elif event.num == 5 or event.delta < 0:
            self.zoom_level /= 1.1  
        self.show_image()


class DatasetViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Viewer (Spacebar to Next)")
        self.root.geometry("1100x650")

        # Define dataset directories
        self.img_dir = "dataset_folder/data/images"
        self.txt_dir = "dataset_folder/data/text"

        # Load file lists safely
        try:
            self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        except FileNotFoundError:
            self.image_files = []
            
        self.current_index = 0

        # UI Layout setup
        self.setup_ui()
        
        # Load the initial file pair
        self.load_dataset_item()

        # Bind the Spacebar event to the entire root window
        self.root.bind("<space>", self.next_item)

    def setup_ui(self):
        # Adjustable side-by-side pane separation
        paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Left Column: Image Area
        image_frame = tk.Frame(paned_window)
        self.canvas_viewer = ZoomableImageCanvas(image_frame, bg="#2e2e2e")
        self.canvas_viewer.pack(fill=tk.BOTH, expand=True)
        paned_window.add(image_frame, stretch="always")

        # Right Column: Text Area
        text_frame = tk.Frame(paned_window)
        text_scroll = tk.Scrollbar(text_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        bangla_font = font.Font(family="Arial", size=14)
        self.text_area = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=text_scroll.set, font=bangla_font, padx=12, pady=12)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.text_area.yview)
        paned_window.add(text_frame, stretch="always")

        # Top Information Bar
        self.info_label = tk.Label(self.root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 10))
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)

    def load_dataset_item(self):
        """Updates both fields with data corresponding to current_index."""
        if not self.image_files:
            self.text_area.insert(tk.END, "No image files found in target folder.")
            return

        # Fetch current image file name
        img_name = self.image_files[self.current_index]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Determine cross-matching text filename (stripping extension safely)
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(self.txt_dir, f"{base_name}.txt")

        # 1. Update Canvas Image Component
        self.canvas_viewer.load_new_image(img_path)

        # 2. Update Text Box Component
        self.text_area.config(state=tk.NORMAL) # Temporarily unlock text field to insert content
        self.text_area.delete("1.0", tk.END)   # Clear old text cleanly

        try:
            with open(txt_path, "r", encoding="utf-8") as file:
                bangla_content = file.read()
                self.text_area.insert(tk.END, bangla_content)
        except FileNotFoundError:
            self.text_area.insert(tk.END, f"Error: Matching file '{base_name}.txt' not found inside text folder.")
        except Exception as e:
            self.text_area.insert(tk.END, f"Error loading text: {str(e)}")

        self.text_area.config(state=tk.DISABLED) # Relock text field

        # 3. Update Status Indicator 
        self.info_label.config(text=f" File {self.current_index + 1} of {len(self.image_files)} | Displaying: {img_name} [Press Space for Next]")

    def next_item(self, event=None):
        """Increments index and handles wrap-around loops back to file 0."""
        if not self.image_files:
            return
            
        self.current_index += 1
        if self.current_index >= len(self.image_files):
            self.current_index = 0 # Loop back to beginning
            
        self.load_dataset_item()


if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetViewerApp(root)
    root.mainloop()
