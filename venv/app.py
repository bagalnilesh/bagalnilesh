import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import time


class AgingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Future Face Conversion")
        self.root.geometry("900x700")
        self.root.config(bg="#F4F6F8")  # Light background color

        self.image_dir = None  # Store the selected directory
        self.selected_images = []  # List to store selected image paths
        self.processed_images = []  # List to store processed image statuses
        self.processing_times = []  # List to store processing times
        self.age_predictions = []  # List to store age predictions (as example)
        self.processing_status = []  # Success or failure status for each image

        # Create a frame with a border around the window
        self.main_frame = tk.Frame(self.root, bd=10, relief="solid", bg="#F4F6F8")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Set up the UI elements
        self.create_widgets()

    def create_widgets(self):
        # Main Heading Label
        header = tk.Label(self.main_frame, text="Future Face Synthesis Using Generative AI", font=("Helvetica", 24, "bold"), bg="white", fg="blue")
        header.pack(fill="x", pady=20)

        # Frame for Buttons
        button_frame = tk.Frame(self.main_frame, bg="#F4F6F8")
        button_frame.pack(pady=20)

        # Upload Directory Button
        self.upload_button = ttk.Button(button_frame, text="Upload Directory", command=self.upload_directory, width=20)
        self.upload_button.grid(row=0, column=0, padx=10)

        # Process Button
        self.process_button = ttk.Button(button_frame, text="Process All Images", command=self.process_images, width=20)
        self.process_button.grid(row=0, column=1, padx=10)

        # Display selected images in a frame
        self.images_frame = tk.Frame(self.main_frame, bg="#F4F6F8")
        self.images_frame.pack(pady=10)

        # Directory Info Label
        self.directory_label = tk.Label(self.main_frame, text="No directory selected", font=("Helvetica", 12), bg="#F4F6F8")
        self.directory_label.pack(pady=10)

        # Number of images to process label
        self.num_images_label = tk.Label(self.main_frame, text="Total images to process: 0", font=("Helvetica", 12), bg="#F4F6F8")
        self.num_images_label.pack(pady=5)

    def upload_directory(self):
        # Open file dialog to select a directory containing images
        dir_path = filedialog.askdirectory(title="Select Directory Containing Images")
        if dir_path:
            self.image_dir = dir_path
            self.directory_label.config(text=f"Selected Directory: {dir_path}")
            self.display_images_from_directory()

    def display_images_from_directory(self):
        # Clear any previously displayed images
        for widget in self.images_frame.winfo_children():
            widget.destroy()

        # Get all images from the selected directory
        supported_formats = ('.jpg', '.jpeg', '.png')
        self.selected_images = [
            os.path.join(self.image_dir, file)
            for file in os.listdir(self.image_dir)
            if file.lower().endswith(supported_formats)
        ]

        if not self.selected_images:
            messagebox.showinfo("No Images", "No image files found in the selected directory.")
            return

        # Update the number of images label
        self.num_images_label.config(text=f"Total images to process: {len(self.selected_images)}")

        # Display each selected image as a thumbnail
        for image_path in self.selected_images:
            image = Image.open(image_path)
            image.thumbnail((150, 150))
            image_tk = ImageTk.PhotoImage(image)
            label = tk.Label(self.images_frame, image=image_tk, bg="#F4F6F8")
            label.image = image_tk  # Keep reference
            label.pack(side="left", padx=10, pady=10)

    def process_images(self):
        if not self.image_dir:
            messagebox.showerror("Error", "Please upload a directory first.")
            return

        try:
            for image_path in self.selected_images:
                start_time = time.time()

                # Run the aging process (simulation)
                self.run_inference(image_path)

                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

                # Simulate age prediction
                self.age_predictions.append(np.random.randint(40, 80))

                # Simulate success/failure
                self.processing_status.append("Success" if np.random.random() > 0.1 else "Failure")

            messagebox.showinfo("Success", "Processing completed for all images!")
            self.generate_graph()

        except Exception as e:
            messagebox.showerror("Error", f"Error during processing: {e}")

    def run_inference(self, image_path):
        # Simulated inference. Replace with actual model call if needed.
        output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        # Example: subprocess.run(['python', 'infer.py', '--image_dir', image_path])
        pass

    def generate_graph(self):
        image_indices = list(range(1, len(self.selected_images) + 1))
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Image Index')
        ax1.set_ylabel('Processing Time (seconds)', color='tab:blue')
        ax1.bar(image_indices, self.processing_times, color='tab:blue', alpha=0.7, label='Processing Time')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Age Prediction', color='tab:orange')
        ax2.plot(image_indices, self.age_predictions, color='tab:orange', marker='o', linestyle='-', label='Age Prediction')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('Success/Failure', color='tab:green')
        ax3.bar(image_indices, [1 if status == "Success" else 0 for status in self.processing_status], color='tab:green', alpha=0.3, label='Success/Failure')
        ax3.tick_params(axis='y', labelcolor='tab:green')

        plt.title('Image Processing Metrics (Time, Age Prediction, Success/Failure)')
        ax1.grid(True, linestyle='--', alpha=0.6)

        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = AgingApp(root)
    root.mainloop()