import requests
import json

def test_rss():
    print("Testing App Store RSS feed...")
    url = "https://itunes.apple.com/id/rss/customerreviews/id=1330123889/json"
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            feed = data.get('feed', {})
            entries = feed.get('entry', [])
            print(f"Fetched {len(entries)} reviews via RSS.")
            if entries:
                print("Sample entry title:", entries[0].get('title', {}).get('label'))
        else:
            print(f"Failed to fetch RSS. Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_rss()
