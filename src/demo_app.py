#%%
# Imports
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image
import requests
import io
from recommender import Recommender
import pandas as pd
from utils.paths import ADDITIONAL_DIR, MOVIE_LENS_DIR, BASE_DIR
from config_secrets import OMDb_API_KEY
import threading
import math



ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")



class LoginFrame(ctk.CTkFrame):
    def __init__(self, master, switch_to_main, switch_to_register):
        super().__init__(master)
        self.switch_to_main = switch_to_main
        self.switch_to_register = switch_to_register
        ctk.CTkLabel(self, text="Login", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=20)
        self.entry = ctk.CTkEntry(self, placeholder_text="user ID", width=200)
        self.entry.pack(pady=10)
        ctk.CTkButton(self, text="Continue", command=self.login).pack(pady=10)
        ctk.CTkButton(self, text="New User", command=self.switch_to_register, fg_color="gray").pack(pady=5)

    def login(self):
        userId = self.entry.get().strip()
        if not userId:
            messagebox.showerror("Error", "Please enter a valid ID.")
        else:
            self.switch_to_main(userId)



class RegisterFrame(ctk.CTkFrame):
    def __init__(self, master, go_back_callback):
        super().__init__(master)
        self.moviesDf = master.recommender.data.moviesDf
        self.goBackCallback = go_back_callback
        self.searchJob = None
        self.selectedLovedMovies = []
        self.selectedDislikedMovies = []
        self.mode = "loved"  # default mode
        self.userId = Recommender.data.ratingsDf['userId'].max() + 1 if not Recommender.data.ratingsDf.empty else 1

        ctk.CTkLabel(self, text=f"New User - Your ID is {self.userId}", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))

        # Switch button for loved/disliked mode
        self.toggle_button = ctk.CTkButton(self, text="Loved ❤️", command=self.toggle_mode, fg_color="purple")
        self.toggle_button.pack(pady=5)

        # Search bar for movie titles
        self.searchEntry = ctk.CTkEntry(self, placeholder_text="Search movie")
        self.searchEntry.pack(pady=5)
        self.searchEntry.bind("<KeyRelease>", self.update_results)

        # Frame to display search results
        self.resultsFrame = ctk.CTkScrollableFrame(self, height=150)
        self.resultsFrame.pack(padx=10, pady=(20, 0), fill="both", expand=False)

        # Labels to show selected movies
        self.lovedMoviesLabel = ctk.CTkLabel(self, text="")
        self.lovedMoviesLabel.pack(pady=5)
        self.dislikedMoviesLabel = ctk.CTkLabel(self, text="")
        self.dislikedMoviesLabel.pack(pady=5)

        # Navigation buttons
        ctk.CTkButton(self, text="Register", command=self.register_user, fg_color='green').pack(pady=(0, 10))
        ctk.CTkButton(self, text="Back to login", command=self.goBackCallback, fg_color="gray").pack(pady=(0, 20))


    def toggle_mode(self):
        self.mode = "disliked" if self.mode == "loved" else "loved"
        mode_text = "Disliked 💔" if self.mode == "disliked" else "Loved ❤️"
        self.toggle_button.configure(text=f"{mode_text}")


    def update_results(self, event=None):
        if self.searchJob:
            self.after_cancel(self.searchJob)
        self.searchJob = self.after(350, self.perform_search)


    def perform_search(self):
        query = self.searchEntry.get().strip().lower()
        for widget in self.resultsFrame.winfo_children():
            widget.destroy()
        if not query:
            return
        matches = self.moviesDf[self.moviesDf['title'].str.lower().str.contains(query, na=False)]
        for title in matches['title'].tolist():
            ctk.CTkButton(
                self.resultsFrame,
                text=title,
                command=lambda t=title: self.select_movie(t)
            ).pack(pady=2, padx=5, fill="x")


    def select_movie(self, title):
        movieId = self.moviesDf[self.moviesDf['title'] == title]['movieId'].values[0]
        if self.mode == "loved":
            if (movieId, title) not in self.selectedLovedMovies:
                self.selectedLovedMovies.append((movieId, title))
            else:
                self.selectedLovedMovies.remove((movieId, title))
        else:
            if (movieId, title) not in self.selectedDislikedMovies:
                self.selectedDislikedMovies.append((movieId, title))
            else:
                self.selectedDislikedMovies.remove((movieId, title))
        self.update_labels()


    def update_labels(self):
        lovedTitles = [x[1] for x in self.selectedLovedMovies]
        dislikedTitles = [x[1] for x in self.selectedDislikedMovies]
        self.lovedMoviesLabel.configure(text=f"Loved movies: {lovedTitles}") if lovedTitles else self.lovedMoviesLabel.configure(text="")
        self.dislikedMoviesLabel.configure(text=f"Disliked movies: {dislikedTitles}") if dislikedTitles else self.dislikedMoviesLabel.configure(text="")


    def register_user(self):
        if len(self.selectedLovedMovies) + len(self.selectedDislikedMovies) < 4:
            messagebox.showerror("Error", "Please select at least 4 movies.")
        else:
            lovedMoviesDf = pd.DataFrame([x[0] for x in self.selectedLovedMovies], columns=['movieId'])
            lovedMoviesDf['rating'] = 5
            dislikedMoviesDf = pd.DataFrame([x[0] for x in self.selectedDislikedMovies], columns=['movieId'])
            dislikedMoviesDf['rating'] = 0.5
            userRatingsDf = pd.concat([lovedMoviesDf, dislikedMoviesDf], ignore_index=True)
            userRatingsDf['userId'] = self.userId
            userRatingsDf = userRatingsDf[['userId', 'movieId', 'rating']]
            Recommender.data.ratingsDf = pd.concat([Recommender.data.ratingsDf, userRatingsDf], ignore_index=True)
            userRatingsDf.to_csv(ADDITIONAL_DIR / "myRatings.csv", mode='a', header=False, index=False)
            self.master.show_main(self.userId)



class MainFrame(ctk.CTkFrame):
    def __init__(self, master, userId, logoutCallback):
        super().__init__(master)
        ctk.CTkLabel(self, text=f"User {userId}", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=30)
        
        ctk.CTkLabel(self, text="Choose a recommendation method:", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        ctk.CTkButton(self, text="Collaborative", command=lambda: master.show_recommendations(userId, 'Collaborative')).pack(pady=20)
        ctk.CTkButton(self, text="Content-Based", command=lambda: master.show_recommendations(userId, 'Content-Based')).pack(pady=20)
        ctk.CTkButton(self, text="Hybrid", command=lambda: master.show_recommendations(userId, 'Hybrid')).pack(pady=20)
        ctk.CTkButton(self, text="RL", command=lambda: master.show_recommendations(userId, 'RL')).pack(pady=20)
        
        ctk.CTkButton(self, text="Change User", command=logoutCallback, fg_color="gray").pack(pady=20)



class RecommendationsFrame(ctk.CTkFrame):
    def __init__(self, master, userId, method):
        super().__init__(master)
        self.master = master
        self.userId = int(userId)
        self.method = method
        self.images = {}

        # Loading label
        self.loading_label = ctk.CTkLabel(self, text="Loading recommendations...", font=ctk.CTkFont(size=16))
        self.loading_label.pack(pady=50)

        # Loading data in a separate thread
        self.after(100, lambda: threading.Thread(target=self.load_data, daemon=True).start())


    def load_data(self):
        # Recommendations in a separate thread
        self.master.recommender.reset(userId=self.userId, method=self.method)
        recommendations = self.master.recommender.get_recommendations()
        self.recommendations = recommendations.merge(self.master.linksDf, on='movieId', how='left')

        self.fetch_all_posters()


    def fetch_all_posters(self):
        def fetch_images():
            for idx, row in self.recommendations.iterrows():
                imdbId = row['imdbId']
                posterUrl = self.get_poster_url(imdbId)
                image = self.load_image_from_url(posterUrl)
                self.images[idx] = image
            # Back to main thread after fetching images
            self.master.after(0, self.display_ui)

        threading.Thread(target=fetch_images, daemon=True).start()


    def display_ui(self):
        self.loading_label.destroy()

        ctk.CTkLabel(self, text=f"Recommendations for User {self.userId} using {self.method} method",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=20)

        self.scrollFrame = ctk.CTkScrollableFrame(self)
        self.scrollFrame.pack(expand=True, fill="both", padx=10, pady=10)

        self.after(200, self.display_all_movies)

        ctk.CTkButton(self, text="Back", command=lambda: self.master.show_main(self.userId)).pack(pady=20)


    def display_all_movies(self):
        frameWidth = self.scrollFrame.winfo_width()
        self.numColumns = max(frameWidth // 180, 2)

        for index, row in self.recommendations.iterrows():
            self.display_movie(row, index)


    def display_movie(self, movieRow, index):
        title = movieRow['title'] + f" ({int(movieRow['year']) if not math.isnan(movieRow['year']) else ''})"
        image = self.images.get(index)  # preloaded image

        imgLabel = ctk.CTkLabel(self.scrollFrame, image=image, text="")
        imgLabel.image = image

        textLabel = ctk.CTkLabel(self.scrollFrame, text=title, font=ctk.CTkFont(size=12), wraplength=150, justify="center")

        rowPos = index // self.numColumns
        colPos = index % self.numColumns

        imgLabel.grid(row=rowPos * 2, column=colPos, padx=10, pady=(10, 0))
        textLabel.grid(row=rowPos * 2 + 1, column=colPos, padx=10, pady=(0, 15))


    def load_image_from_url(self, url):
        try:
            if url:
                response = requests.get(url, timeout=5)
                image = Image.open(io.BytesIO(response.content))
            else:
                raise ValueError("Empty URL")
        except:
            image = Image.open(BASE_DIR / "../images/film.png")

        return ctk.CTkImage(light_image=image, dark_image=image, size=(150, 220))


    def get_poster_url(self, imdbId):
        if pd.isna(imdbId):
            return None
        imdbIdStr = f"tt{int(imdbId):07d}"
        print(f"Fetching poster for IMDb ID: {imdbIdStr}")
        url = f"http://www.omdbapi.com/?i={imdbIdStr}&apiKey={OMDb_API_KEY}"
        try:
            response = requests.get(url)
            data = response.json()
            return data.get('Poster') if data.get('Poster') and data.get('Poster') != "N/A" else None
        except:
            return None



class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Movie Recommender App")
        self.geometry("770x600")
        self.minsize(600, 600)
        self.resizable(True, True)
        self.show_login()
        self.recommender = Recommender()
        self.linksDf =  pd.read_csv(MOVIE_LENS_DIR / "links.csv")

    def show_login(self):
        self.clear_widgets()
        self.loginFrame = LoginFrame(self, self.show_main, self.show_register)
        self.loginFrame.pack(expand=True, fill="both", padx=20, pady=20)

    def show_register(self):
        self.clear_widgets()
        self.registerFrame = RegisterFrame(self, self.show_login)
        self.registerFrame.pack(expand=True, fill="both", padx=20, pady=20)

    def show_main(self, userId):
        self.clear_widgets()
        self.mainFrame = MainFrame(self, userId, self.show_login)
        self.mainFrame.pack(expand=True, fill="both", padx=20, pady=20)

    def show_recommendations(self, userId, method):
        self.clear_widgets()
        self.recommendationsFrame = RecommendationsFrame(self, userId, method)
        self.recommendationsFrame.pack(expand=True, fill="both", padx=20, pady=20)

    def clear_widgets(self):
        for widget in self.winfo_children():
            widget.destroy()


 
if __name__ == "__main__":
    app = App()
    app.mainloop()
