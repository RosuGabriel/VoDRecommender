#%%
# Imports
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
from recommender import Recommender
import pandas as pd
from utils.paths import ADDITIONAL_DIR



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
        matches = moviesDf[moviesDf['title'].str.lower().str.contains(query, na=False)]
        for title in matches['title'].tolist():
            ctk.CTkButton(
                self.resultsFrame,
                text=title,
                command=lambda t=title: self.select_movie(t)
            ).pack(pady=2, padx=5, fill="x")


    def select_movie(self, title):
        movieId = moviesDf[moviesDf['title'] == title]['movieId'].values[0]
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
            self.switch_to_main(self.userId)



class MainFrame(ctk.CTkFrame):
    def __init__(self, master, userId, logoutCallback):
        super().__init__(master)
        ctk.CTkLabel(self, text=f"User {userId}", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=30)
        
        ctk.CTkLabel(self, text="Choose a recommendation method:", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        ctk.CTkButton(self, text="Collaborative", command=lambda: master.show_recommendations(userId, 'collaborative')).pack(pady=20)
        ctk.CTkButton(self, text="Content Based", command=lambda: master.show_recommendations(userId, 'content_based')).pack(pady=20)
        ctk.CTkButton(self, text="Hybrid", command=lambda: master.show_recommendations(userId, 'hybrid')).pack(pady=20)
        ctk.CTkButton(self, text="RL", command=lambda: master.show_recommendations(userId, 'rl')).pack(pady=20)
        
        ctk.CTkButton(self, text="Change User", command=logoutCallback, fg_color="gray").pack(pady=20)


class RecommendationsFrame(ctk.CTkFrame):
    def __init__(self, master, userId, method):
        super().__init__(master)
        userId = int(userId)
        master.recommender.reset(userId=userId, method=method)
        self.recommendations = master.recommender.get_recommendations()
        
        ctk.CTkLabel(self, text=f"Recommendations for User {userId} using {method} method", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=20)
        
        if not self.recommendations.empty:
            for index, row in self.recommendations.iterrows():
                ctk.CTkLabel(self, text=row['title']).pack(pady=5)
        else:
            ctk.CTkLabel(self, text="No recommendations found.").pack(pady=5)
        
        ctk.CTkButton(self, text="Back", command=lambda: master.show_main(userId)).pack(pady=20)



class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Movie Recommender App")
        self.geometry("750x600")
        self.minsize(600, 600)
        self.resizable(True, True)
        self.show_login()
        self.recommender = Recommender()

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
    moviesDf = Recommender.data.moviesDf
    app = App()
    app.mainloop()
