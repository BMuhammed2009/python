import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk, Image as PilImage  # Import PilImage for handling images
import threading
import time
import os

mask_threshold = 800  # Set a threshold for motion detection sensitivity

camera_id = 1  # Camera ID
# komputerin oz kamerasini istifade edende camera_id = 0 olmalidir
# qiraqda usb kamera qosduqda camera_id = 1 olmalidir

icon_path = "./insect.png"  # Icon path

class InsectDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Load and set the window icon
        img = PilImage.open(icon_path)
        img = ImageTk.PhotoImage(img)
        self.window.iconphoto(False, img)

        self.save_folder = "save"  # Folder to save detected images
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.search_img_folder = "search_img"  # Folder containing searching images
        self.search_images = sorted([f for f in os.listdir(self.search_img_folder) if f.endswith('.png')])
        self.searching = False

        # Create a main frame for layout
        self.main_frame = Frame(window)
        self.main_frame.pack(fill=BOTH, expand=True)

        # Frame for video feed
        self.video_frame = Frame(self.main_frame)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.cap = cv2.VideoCapture(camera_id)
        self.canvas = Canvas(self.video_frame, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Frame for buttons and searching images
        self.button_frame = Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, pady=10, sticky="nw")

        # Rounded Start/Stop buttons
        self.btn_start = Button(self.button_frame, text="Start", width=10, height=2, command=self.start_detection)
        self.btn_start.grid(row=0, column=0, padx=10, pady=10)

        self.btn_stop = Button(self.button_frame, text="Stop", width=10, height=2, state=DISABLED, command=self.stop_detection)
        self.btn_stop.grid(row=0, column=1, padx=10, pady=10)

        # Frame for searching images horizontally aligned with the buttons
        self.searching_label = Label(self.button_frame, text="Searching Images", fg="red", font=("Helvetica", 12))
        self.searching_label.grid(row=0, column=2, padx=10)

        self.searching_image_label = Label(self.button_frame)
        self.searching_image_label.grid(row=0, column=3, padx=10)

        # Frame for image gallery
        self.gallery_frame = Frame(self.main_frame)
        self.gallery_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")

        self.canvas_gallery = Canvas(self.gallery_frame, bg='grey')
        self.scrollbar = Scrollbar(self.gallery_frame, orient="vertical", command=self.canvas_gallery.yview)
        self.canvas_gallery.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=RIGHT, fill=Y)

        self.canvas_gallery.pack(side=LEFT, fill=BOTH, expand=True)

        self.frame_images = Frame(self.canvas_gallery, bg='grey')
        self.canvas_frame = self.canvas_gallery.create_window((0, 0), window=self.frame_images, anchor="nw")

        self.canvas_gallery.bind("<Configure>", self.on_canvas_configure)
        self.frame_images.bind("<Configure>", lambda e: self.canvas_gallery.configure(scrollregion=self.canvas_gallery.bbox("all")))

        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.detect = False
        self.detection_thread = None

        self.update_video()
        self.window.mainloop()

    def on_canvas_configure(self, event):
        self.canvas_gallery.itemconfig(self.canvas_frame, width=event.width)

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.window.after(10, self.update_video)

    def start_detection(self):
        self.detect = True
        self.searching = True
        self.btn_start.config(state=DISABLED)
        self.btn_stop.config(state=NORMAL)
        self.detection_thread = threading.Thread(target=self.detect_motion)
        self.detection_thread.start()

        # Start showing searching images
        self.searching_thread = threading.Thread(target=self.show_searching_images)
        self.searching_thread.start()

    def stop_detection(self):
        self.detect = False
        self.searching = False
        self.btn_start.config(state=NORMAL)
        self.btn_stop.config(state=DISABLED)
        self.detection_thread.join()
        self.searching_thread.join()  # Wait for the searching thread to finish
        self.refresh_gallery()

        # Clear the searching image
        self.searching_image_label.config(image='')

    def detect_motion(self):
        while self.detect:
            ret, frame = self.cap.read()
            if ret:
                motion_mask = self.background_subtractor.apply(frame)
                if cv2.countNonZero(motion_mask) > mask_threshold:
                    timestamp = int(time.time())
                    img_name = f"{self.save_folder}/detected_insect_{timestamp}.jpg"
                    cv2.imwrite(img_name, frame)

    def refresh_gallery(self):
        for widget in self.frame_images.winfo_children():
            widget.destroy()
        images = sorted(os.listdir(self.save_folder))
        for index, img_file in enumerate(images):
            if img_file.endswith('.jpg'):
                row = index // 3
                col = index % 3
                self.update_gallery(f"{self.save_folder}/{img_file}", row, col)

    def update_gallery(self, img_path, row, col):
        img = Image.open(img_path)
        img.thumbnail((100, 100))
        img = ImageTk.PhotoImage(img)
        panel = Label(self.frame_images, image=img)
        panel.image = img
        panel.grid(row=row, column=col, padx=10, pady=10)
        panel.bind('<Button-1>', lambda e, path=img_path: self.open_full_image(path))

    def open_full_image(self, path):
        top = Toplevel(self.window)
        img = Image.open(path)
        photo = ImageTk.PhotoImage(img)
        img_label = Label(top, image=photo)
        img_label.image = photo
        img_label.pack()

        btn_delete = Button(top, text="Delete Image", command=lambda: self.delete_image(path, top))
        btn_delete.pack()

    def delete_image(self, path, top):
        os.remove(path)
        top.destroy()
        self.refresh_gallery()

    def show_searching_images(self):
        while self.searching:
            for img_file in self.search_images:
                if not self.searching:
                    break
                img_path = os.path.join(self.search_img_folder, img_file)
                searching_img = PilImage.open(img_path)
                searching_img = searching_img.resize((50, 50))  # Size to fit next to the label
                searching_img = ImageTk.PhotoImage(searching_img)
                self.searching_image_label.config(image=searching_img)
                self.searching_image_label.image = searching_img
                self.window.update_idletasks()
                for _ in range(10):  # Check 10 times per second if the stop button has been pressed
                    if not self.searching:
                        break
                    time.sleep(0.01)

if __name__ == "__main__":
    root = Tk()
    InsectDetectorApp(root, "Insect Detector App")
