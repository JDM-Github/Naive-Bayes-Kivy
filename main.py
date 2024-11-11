import math
from kivy.config import Config
Config.set('graphics', 'width', 1100)
Config.set('graphics', 'height', 600)
Config.write()

from kivy.core.window import Window
Window.size = (1100, 600)

import threading
from kivy.app import App
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivymd.uix.filemanager import MDFileManager
from kivy.animation import Animation
from kivy.properties import NumericProperty, StringProperty

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class Model:

    def __init__(self):
        df = pd.read_csv('MovieReviewTrainingDatabase.csv')
        df['review'] = df['review'].fillna('Unknown')
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(df['review'])
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(df['sentiment'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)

class MainWidget(Widget):

    positive_len = NumericProperty(0)
    negative_len = NumericProperty(0)
    neutral_len = NumericProperty(0)

    positive_percent = NumericProperty(0)
    negative_percent = NumericProperty(0)
    neutral_percent = NumericProperty(0)

    max_percent_label = StringProperty("POSITIVE")
    max_percent = NumericProperty(0)

    full_comments_file = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.naiveBayes = None
        self.load_model_in_thread()
    
    def load_model_in_thread(self):
        def load_model():
            self.naiveBayes = Model()
            print("Model loaded successfully")

        model_thread = threading.Thread(target=load_model)
        model_thread.start()
    
    def on_image_click(self):
        self.open_file_manager()

    def open_file_manager(self):
        self.file_manager = MDFileManager(
            exit_manager=self.exit_file_manager,
            select_path=self.select_file,
        )
        self.file_manager.show('/')

    def select_file(self, path):
        if path.endswith('.csv'):
            self.full_comments_file = path
            print(f"File selected: {path}")
        else:
            self.show_error_popup("Invalid file type", "Please select a CSV file.")
        self.file_manager.close()

    def exit_file_manager(self, *args):
        self.file_manager.close()

    def open_csv(self):
        if not self.naiveBayes or not self.full_comments_file:
            self.show_error_popup("Invalid Submit", "NaiveBayes model or file is missing.")
            print("INVALID SUBMIT")
            return

        model = self.naiveBayes.model
        vectorizer = self.naiveBayes.vectorizer
        encoder = self.naiveBayes.encoder


        try:
            full_df = pd.read_csv(self.full_comments_file)
        except Exception as e:
            self.show_error_popup("File Read Error", f"Error reading the file: {e}")
            return
        full_df.columns = full_df.columns.str.lower()
        target = None
        if 'comment' in full_df.columns:
            target = full_df['comment']
        
        elif 'review' in full_df.columns:
            target = full_df['review']
        
        elif 'text' in full_df.columns:
            target = full_df['text']

        if target is None:
            self.show_error_popup("Missing Column", "The file must contain a 'comment, review, text' column.")
            return

        target = target.fillna('Unknown')
        predicted_sentiments = []

        for comment in target:
            transformed_comment = vectorizer.transform([comment])
            prediction_probabilities = model.predict_proba(transformed_comment)
            neutral_prob = prediction_probabilities[0][0] < 0.6 and prediction_probabilities[0][1] < 0.6
            
            if neutral_prob:
                sentiment = 'Neutral'
            else:
                prediction = model.predict(transformed_comment)
                sentiment = encoder.inverse_transform(prediction)[0]
            
            predicted_sentiments.append(sentiment)
        
        full_df['Predicted_Sentiment'] = predicted_sentiments
        positive_comments = full_df[full_df['Predicted_Sentiment'] == 'Positive']
        negative_comments = full_df[full_df['Predicted_Sentiment'] == 'Negative']
        neutral_comments = full_df[full_df['Predicted_Sentiment'] == 'Neutral']

        self.positive_len = len(positive_comments)
        self.negative_len = len(negative_comments)
        self.neutral_len  = len(neutral_comments)

        total_comments = len(full_df)
        self.positive_percent = (self.positive_len / total_comments) * 100
        self.neutral_percent = (self.neutral_len / total_comments) * 100
        self.negative_percent = (self.negative_len / total_comments) * 100

        percentages = {
            "POSITIVE": self.positive_percent,
            "NEUTRAL": self.neutral_percent,
            "NEGATIVE": self.negative_percent
        }

        self.max_percent_label = max(percentages, key=percentages.get)
        self.max_percent = int(percentages[self.max_percent_label])
        self.draw_pie_chart([self.negative_percent, self.neutral_percent, self.positive_percent])
    
    def draw_pie_chart(self, data):
        total = sum(data)
        colors = [
            (237 / 255, 106 / 255, 110 / 255),
            (240 / 255, 137 / 255, 44 / 255), 
            (134 / 255, 207 / 255, 111 / 255)]
    
        engagement = self.ids.engagement_pie
        engagement.canvas.clear()

        self.animate_pie(engagement, colors, 0, 0, data, total)
    
    def animate_pie(self, engagement, colors, index, start_angle, data, total):
        if (index > 2):
            with engagement.canvas:
                Color(9 / 255, 25 / 255, 47 / 255)
                pie = Ellipse(
                    pos=(
                        (engagement.x + engagement.parent.width / 2 - 60 + engagement.width / 4) + engagement.width / 4,
                        (engagement.y - 40 + engagement.height / 4) + engagement.height / 4),
                    size=(1, 1)
                )
                Animation(size=(engagement.width / 2, engagement.height / 2),
                    pos=((engagement.x + engagement.parent.width / 2 - 60 + engagement.width / 4),
                        (engagement.y - 40 + engagement.height / 4)), duration=0.5).start(pie)
            return

        angle = 360 * (data[index] / total)
        with engagement.canvas:
            Color(*colors[index])
            pie_segment = Ellipse(
                pos=(engagement.x + engagement.parent.width / 2 - 60, engagement.y - 40),
                size=engagement.size,
                angle_start=start_angle,
                angle_end=start_angle
            )
        animation = Animation(angle_end=start_angle + angle, duration=0.5)
        start_angle += angle

        animation.bind(on_complete=lambda *_: self.animate_pie(engagement, colors, index+1, start_angle, data, total))
        animation.start(pie_segment)

    def show_error_popup(self, title, message):
        content = Label(text=message, size_hint=(1, 0.8), font_size="28sp", bold=True)
        close_button = Button(text="Close", size_hint=(1, 0.2), background_color=(9/255, 25/255, 47/255), bold=True)
        close_button.bind(on_release=self.close_popup)
        
        self.popup = Popup(content=content, size_hint=(0.8, 0.3))
        self.popup.content.add_widget(close_button)
        self.popup.open()

    def close_popup(self, instance):
        self.popup.dismiss()


class MainApp(MDApp):

    def build(self):
        Window.bind(on_drop_file=self._on_file_drop)
        self.main_widget = MainWidget()
        return self.main_widget

    def _on_file_drop(self, window, file_path, *args):
        decoded_path = file_path.decode('utf-8')
        if decoded_path.endswith('.csv'):
            self.main_widget.full_comments_file = decoded_path
        else:
            self.main_widget.show_error_popup("Invalid file type", "Please drop a .csv file.")
        return

    

if __name__ == "__main__":
    MainApp().run()