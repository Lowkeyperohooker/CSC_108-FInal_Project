from rec_sys_data import Data
import sys


def main_menu():
    print("\n\nWelcome to our Movie Recommendation System")
    print()
    print("1. Show movies available")
    print("2. Get recommendations")
    print("3. Exit")
    return input()


# Usage
data = Data()
recommender = data.get_data()

while True:
    match main_menu():
        case '1':
            avail_movies = recommender.show_movies()
            for movie in avail_movies:
                print(movie)
            print()
            print("Copy and paste in 'Get recommendations' for more fun.")

        case '2':
            recommendations = None
            try:
                recommendations = recommender.get_recommendations(input('Paste here> '))
                print("Recommendations\n")
                for rec in recommendations:
                    print(rec)
                print()
            except KeyError as e:
                print(f"Error: {e}")

        case '3':
            sys.exit()

        case _:
            print("Invalid option")
