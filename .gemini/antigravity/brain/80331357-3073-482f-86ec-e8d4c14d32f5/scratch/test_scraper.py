from app_store_scraper import AppStore
import pandas as pd

def test():
    print("Testing App Store Scraper...")
    try:
        app = AppStore(country='id', app_name='pubg-mobile', app_id=1330123889)
        app.review(how_many=10)
        print(f"Fetched {len(app.reviews)} reviews.")
        if app.reviews:
            print("Sample review:", app.reviews[0])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
